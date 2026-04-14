"""
SpectraLoRA inference server on Modal.

    modal deploy inference/serve.py          # deploy persistent endpoint
    modal run inference/serve.py             # one-off test

Expects weights on Volume "raman-inference-weights" (run upload_weights.py first).
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
VOLUME_NAME = "raman-inference-weights"
WEIGHTS_DIR = "/weights"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(["torch==2.3.1", "numpy"])
    .pip_install(
        ["torch_scatter", "torch_cluster"],
        find_links="https://data.pyg.org/whl/torch-2.3.1+cu121.html",
    )
    .pip_install([
        "scipy",
        "rdkit",
        "peft",
        "torch_geometric",
        "e3nn==0.5.1",
        "matplotlib",
        "fastapi[standard]",
    ])
    # Bundle model source code into the image
    .add_local_dir(
        "capsule-3259363/code/detanet_model",
        remote_path="/app/detanet_model",
    )
    .add_local_dir(
        "train",
        remote_path="/app/train",
        # only need train_detanet.py
    )
    .add_local_dir(
        "third_party/ELoRA",
        remote_path="/app/third_party/ELoRA",
    )
)

weights_vol = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("raman-inference", image=image)

MINUTES = 60


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A10G",
    volumes={WEIGHTS_DIR: weights_vol},
    timeout=5 * MINUTES,
    scaledown_window=15 * MINUTES,
    memory=16384,
)
class RamanPredictor:
    """Predicts Raman spectrum from SMILES string."""

    @modal.enter()
    def load_models(self):
        import sys
        sys.path.insert(0, "/app")
        sys.path.insert(0, "/app/third_party/ELoRA")

        import argparse
        import inspect

        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from detanet_model.detanet import DetaNet
        from detanet_model.model_loader import BASE_MODEL_CONFIG, TASK_CONFIGS
        from detanet_model.constant import atom_masses

        self.device = torch.device("cuda")
        self.atom_masses = atom_masses

        # --- Physics constants ---
        self.SIGMA = 12.0
        self.TEMP = 298.0
        self.INIT_WL = 532.0
        self.FREQ_SCALE = 0.967
        self.X_GRID = np.linspace(500.0, 4000.0, 3501, dtype=np.float64)

        # --- Load config ---
        cfg = json.loads(Path(f"{WEIGHTS_DIR}/config.json").read_text())

        def _build_args(cfg, device, task):
            return argparse.Namespace(
                task=task,
                num_features=cfg.get("num_features", 160),
                num_block=cfg.get("num_block", 4),
                num_radial=cfg.get("num_radial", 32),
                attention_head=cfg.get("attention_head", 8),
                rc=cfg.get("rc", 5.0),
                dropout=cfg.get("dropout", 0.1),
                pre_layernorm=cfg.get("pre_layernorm", True),
                pre_layernorm_eps=cfg.get("pre_layernorm_eps", 1e-5),
                elora_path="/app/third_party/ELoRA",  # vendored ELoRA with LoRA mods
                device=str(device),
                use_adalora=cfg.get("use_adalora", True),
                adalora_r=cfg.get("adalora_r", 256),
                adalora_alpha=cfg.get("adalora_alpha", 512),
                adalora_dropout=cfg.get("adalora_dropout", 0.1),
                adalora_tinit=cfg.get("adalora_tinit", 10),
                adalora_tfinal=cfg.get("adalora_tfinal", 20),
                adalora_total_step=cfg.get("adalora_total_step", 1000),
                adalora_target_r=cfg.get("adalora_target_r", 128),
                adalora_rslora=cfg.get("adalora_rslora", True),
                adalora_targets=cfg.get("adalora_targets", None),
                adalora_scalar_heads=cfg.get("adalora_scalar_heads", True),
                adalora_attention=cfg.get("adalora_attention", True),
                adalora_all_linears=cfg.get("adalora_all_linears", True),
                adapter_unfreeze_initial=cfg.get("adapter_unfreeze_initial", True),
                adapter_unfreeze_prefixes=cfg.get("adapter_unfreeze_prefixes", None),
                adapter_freeze_base=cfg.get("adapter_freeze_base", True),
            )

        def _extract_sd(obj):
            if isinstance(obj, dict):
                for k in ("model", "state_dict", "module"):
                    if k in obj and isinstance(obj[k], dict):
                        obj = obj[k]
                        break
            if any(k.startswith("module.") for k in obj.keys()):
                obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
            return obj

        def _load_detanet(task, ckpt_name):
            args = _build_args(cfg, self.device, task)
            from train.train_detanet import build_model
            model = build_model(args)
            sd = _extract_sd(torch.load(
                f"{WEIGHTS_DIR}/{ckpt_name}",
                map_location=self.device, weights_only=False,
            ))
            model.load_state_dict(sd, strict=False)
            model = model.to(self.device)
            model.eval()
            print(f"[{task}] loaded {ckpt_name}")
            return model

        # --- Load 3 DetaNet models ---
        self.hi_model = _load_detanet("Hi", "Hi.pth")
        self.hij_model = _load_detanet("Hij", "Hij.pth")
        self.depolar_model = _load_detanet("depolar", "depolar.pth")

        # --- Load RefNet ---
        class FiLM(nn.Module):
            def __init__(self, cd, ch):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(cd, 128), nn.ReLU(), nn.Linear(128, 2 * ch))
            def forward(self, x, c):
                g, b = self.net(c).chunk(2, dim=-1)
                return x * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)

        class Res(nn.Module):
            def __init__(self, ch):
                super().__init__()
                self.c1 = nn.Conv1d(ch, ch, 5, padding=2)
                self.bn1 = nn.BatchNorm1d(ch)
                self.c2 = nn.Conv1d(ch, ch, 5, padding=2)
                self.bn2 = nn.BatchNorm1d(ch)
            def forward(self, x):
                return F.relu(x + self.bn2(self.c2(F.relu(self.bn1(self.c1(x))))))

        class RefNet(nn.Module):
            def __init__(self, in_len, cd=2048, drop=0.15):
                super().__init__()
                self.enc1 = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), Res(16))
                self.enc2 = nn.Sequential(nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), Res(32))
                self.enc3 = nn.Sequential(nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), Res(64))
                self.pool = nn.MaxPool1d(2)
                self.bot = Res(64)
                self.film = FiLM(cd, 64)
                self.drop = nn.Dropout(drop)
                self.u3 = nn.ConvTranspose1d(64, 64, 2, stride=2)
                self.d3 = nn.Sequential(nn.Conv1d(128, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), Res(32))
                self.u2 = nn.ConvTranspose1d(32, 32, 2, stride=2)
                self.d2 = nn.Sequential(nn.Conv1d(64, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), Res(16))
                self.u1 = nn.ConvTranspose1d(16, 16, 2, stride=2)
                self.d1 = nn.Sequential(nn.Conv1d(32, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), Res(16))
                self.head = nn.Conv1d(16, 1, 1)
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
                self.in_len = in_len

            def forward(self, s, m):
                x = s.unsqueeze(1)
                pad = (8 - x.shape[-1] % 8) % 8
                if pad:
                    x = F.pad(x, (0, pad))
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                b = self.drop(self.film(self.bot(self.pool(e3)), m))
                d3 = self.d3(torch.cat([self.u3(b)[:, :, :e3.shape[-1]], e3], 1))
                d2 = self.d2(torch.cat([self.u2(d3)[:, :, :e2.shape[-1]], e2], 1))
                d1 = self.d1(torch.cat([self.u1(d2)[:, :, :e1.shape[-1]], e1], 1))
                delta = self.head(d1).squeeze(1)
                if pad:
                    delta = delta[:, :self.in_len]
                return (s + delta).clamp(0, 1)

        L = len(self.X_GRID)
        self.refnet = RefNet(in_len=L).to(self.device)
        self.refnet.load_state_dict(torch.load(
            f"{WEIGHTS_DIR}/refnet.pth",
            map_location=self.device, weights_only=True,
        ))
        self.refnet.eval()
        print("[refnet] loaded refnet.pth")
        print("All models loaded.")

    def _smiles_to_geometry(self, smiles: str):
        """SMILES -> (pos, z) via RDKit conformer generation."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if status != 0:
            # Fallback to random coords
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
            if status != 0:
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        conf = mol.GetConformer()
        import numpy as np
        pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=np.float32)
        z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
        return pos, z, mol

    def _morgan_fingerprint(self, mol):
        """Compute 2048-bit Morgan fingerprint."""
        import numpy as np
        from rdkit.Chem import AllChem
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp, dtype=np.float32)

    def _gnn_forward(self, pos, z):
        """Run 3 DetaNet models -> (freq, raman_act) as numpy arrays."""
        import torch
        import numpy as np
        from torch_geometric.nn import radius_graph
        from detanet_model.constant import atom_masses
        from detanet_model.spectra_simulator import (
            hessfreq, chain_rule_raman, get_raman_act,
        )

        pos_t = torch.tensor(pos, dtype=torch.float32, device=self.device, requires_grad=True)
        z_t = torch.tensor(z, dtype=torch.long, device=self.device)
        edge_index = radius_graph(x=pos_t, r=5.0)

        with torch.enable_grad():
            hi = self.hi_model(pos=pos_t, z=z_t)
            hij = self.hij_model(pos=pos_t, z=z_t, edge_index=edge_index)
            dp = self.depolar_model(z=z_t, pos=pos_t)
            freq, modes = hessfreq(
                Hi=hi, Hij=hij, edge_index=edge_index,
                masses=atom_masses.to(self.device)[z_t],
                normal=False, linear=False, scale=1.0,
            )
            raman_act = get_raman_act(chain_rule_raman(dp=dp, modes=modes))

        freq = torch.nan_to_num(freq, nan=0.0, posinf=0.0, neginf=0.0)
        raman_act = torch.nan_to_num(raman_act, nan=0.0, posinf=0.0, neginf=0.0)

        freq_np = freq.detach().cpu().numpy().astype(np.float64)
        act_np = raman_act.detach().cpu().numpy().astype(np.float64)

        valid = np.isfinite(freq_np) & np.isfinite(act_np) & (freq_np > 1e-8)
        return freq_np[valid] * self.FREQ_SCALE, act_np[valid]

    def _broaden(self, freq, activity):
        """Lorentzian broadening + Raman intensity correction -> normalized spectrum."""
        import torch
        import numpy as np
        from detanet_model.spectra_simulator import Lorenz_broadening, get_raman_intensity

        if freq.size == 0:
            return np.zeros_like(self.X_GRID, dtype=np.float64)

        x_t = torch.as_tensor(self.X_GRID, dtype=torch.float64, device=self.device)
        f_t = torch.as_tensor(freq, dtype=torch.float64, device=self.device)
        a_t = torch.as_tensor(activity, dtype=torch.float64, device=self.device)

        broadened = Lorenz_broadening(f_t, a_t, c=x_t, sigma=self.SIGMA)
        spec = get_raman_intensity(x_t, broadened, temp=self.TEMP, init_wl=self.INIT_WL)
        spec = spec.detach().cpu().numpy()
        spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
        spec = np.clip(spec, 0.0, None)
        return spec / (spec.max() + 1e-12)

    def _refine(self, spectrum, morgan_fp):
        """Apply RefNet U-Net correction."""
        import torch
        import numpy as np

        s_t = torch.from_numpy(spectrum.astype(np.float32)).unsqueeze(0).to(self.device)
        m_t = torch.from_numpy(morgan_fp).unsqueeze(0).to(self.device)
        with torch.no_grad():
            refined = self.refnet(s_t, m_t).cpu().numpy()[0]
        return refined

    def _pick_peaks(self, spectrum):
        """Extract peak positions and intensities."""
        import numpy as np
        from scipy.signal import find_peaks

        peaks, props = find_peaks(spectrum, prominence=0.03, distance=8)
        positions = self.X_GRID[peaks]
        intensities = spectrum[peaks]
        # Sort by intensity descending
        order = np.argsort(intensities)[::-1]
        return positions[order].tolist(), intensities[order].tolist()

    @modal.method()
    def predict(self, smiles: str) -> dict:
        """Full inference pipeline: SMILES -> Raman spectrum."""
        import numpy as np
        import time

        t0 = time.time()

        # 1. Conformer
        pos, z, mol = self._smiles_to_geometry(smiles)
        t_conf = time.time()

        # 2. GNN forward
        freq, activity = self._gnn_forward(pos, z)
        t_gnn = time.time()

        # 3. Broaden
        spectrum_raw = self._broaden(freq, activity)
        t_broad = time.time()

        # 4. Refine
        morgan_fp = self._morgan_fingerprint(mol)
        spectrum_refined = self._refine(spectrum_raw, morgan_fp)
        t_ref = time.time()

        # 5. Peak pick
        peaks_pos, peaks_int = self._pick_peaks(spectrum_refined)
        peaks_raw_pos, peaks_raw_int = self._pick_peaks(spectrum_raw)

        return {
            "smiles": smiles,
            "n_atoms": int(len(z)),
            "n_modes": int(len(freq)),
            "x_grid": self.X_GRID.tolist(),
            "spectrum_raw": spectrum_raw.tolist(),
            "spectrum_refined": spectrum_refined.tolist(),
            "peaks": {"positions_cm": peaks_pos, "intensities": peaks_int},
            "peaks_raw": {"positions_cm": peaks_raw_pos, "intensities": peaks_raw_int},
            "freq_cm": freq.tolist(),
            "timing": {
                "conformer_s": round(t_conf - t0, 3),
                "gnn_s": round(t_gnn - t_conf, 3),
                "broaden_s": round(t_broad - t_gnn, 3),
                "refine_s": round(t_ref - t_broad, 3),
                "total_s": round(t_ref - t0, 3),
            },
        }

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(self, payload: dict) -> dict:
        """HTTP endpoint: POST {"smiles": "CCO"} -> spectrum JSON."""
        smiles = payload.get("smiles")
        if not smiles:
            return {"error": "Missing 'smiles' field"}
        return self.predict.local(smiles)


# ---------------------------------------------------------------------------
# Local entrypoint for quick testing
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(smiles: str = "c1ccccc1"):
    predictor = RamanPredictor()
    result = predictor.predict.remote(smiles)
    print(f"SMILES: {result['smiles']}")
    print(f"Atoms: {result['n_atoms']}, Modes: {result['n_modes']}")
    print(f"Timing: {result['timing']}")
    print(f"Top 5 peaks (cm-1): {result['peaks']['positions_cm'][:5]}")
