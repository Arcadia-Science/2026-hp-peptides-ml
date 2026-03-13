from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ramanchembl_pipeline.alignment_notebook_lib import (
    AlignmentDatasetBundle,
    DFTModeAlignmentDatasetBundle,
    AlignmentTrainConfig,
    _apply_peak_calibrator_to_line_predictions,
    _apply_variable_warp,
    _build_cosine_basis,
    _build_expert_masks,
    _build_rbf_basis,
    _float_cache_tag,
    _geometry_mol_features,
    _make_model,
    _mode_assignment,
    run_alignment_study,
    _select_best_learned_model,
    _split_indices,
    _split_indices_with_overrides,
)


def test_apply_variable_warp_identity():
    x = torch.linspace(500.0, 4000.0, 32)
    y = torch.linspace(0.0, 1.0, 32).unsqueeze(0)
    warp = torch.zeros_like(y)
    out = _apply_variable_warp(y, warp, x)
    assert torch.allclose(out, y, atol=1e-6)


def test_split_indices_partition_without_overlap():
    split = _split_indices(20, seed=7, val_fraction=0.2, test_fraction=0.2)
    merged = np.concatenate([split["train"], split["val"], split["test"]])
    assert len(np.unique(merged)) == 20
    assert set(merged.tolist()) == set(range(20))


def test_moe_forward_shapes():
    x_grid = np.linspace(500.0, 4000.0, 128, dtype=np.float32)
    cfg = AlignmentTrainConfig(max_epochs=2, pooled_points=32, latent_dim=16, residual_basis=8, warp_basis=4)
    model = _make_model("moe", x_grid, 16, cfg)
    y = torch.rand(4, 128)
    mol = torch.rand(4, 16)
    out = model(y, mol)
    assert out["corrected"].shape == (4, 128)
    assert out["warp_cm"].shape == (4, 128)
    assert out["delta_y"].shape == (4, 128)
    assert out["gate_weights"].shape == (4, cfg.num_experts)
    assert tuple(out["expert_names"]) == (
        "high_freq_anharmonic",
        "low_freq_damping",
        "baseline_shift",
    )
    assert torch.allclose(out["corrected"].amax(dim=1), torch.ones(4), atol=1e-5)


def test_split_indices_with_overrides_respects_requested_train_test():
    split = _split_indices_with_overrides(202, seed=11, val_fraction=0.15, test_fraction=0.15, split_override={"train": 150, "test": 50})
    assert len(split["train"]) == 150
    assert len(split["test"]) == 50
    assert len(split["val"]) == 2


def test_split_indices_with_overrides_scales_down_when_request_exceeds_cases():
    split = _split_indices_with_overrides(89, seed=11, val_fraction=0.15, test_fraction=0.15, split_override={"train": 150, "test": 50})
    assert len(split["train"]) == 57
    assert len(split["test"]) == 19
    assert len(split["val"]) == 13


def test_select_best_learned_model_prefers_higher_primary_coverage():
    summary_df = pd.DataFrame(
        [
            {"model": "uncorrected", "split": "test", "full_coverage@10": 0.50, "full_f1@10": 0.06, "full_median_abs_dnu": 3.8, "rmse": 0.14},
            {"model": "global", "split": "test", "full_coverage@10": 0.51, "full_f1@10": 0.26, "full_median_abs_dnu": 3.5, "rmse": 0.12},
            {"model": "adapter", "split": "test", "full_coverage@10": 0.512, "full_f1@10": 0.263, "full_median_abs_dnu": 3.58, "rmse": 0.119},
            {"model": "moe", "split": "test", "full_coverage@10": 0.506, "full_f1@10": 0.265, "full_median_abs_dnu": 3.40, "rmse": 0.118},
        ]
    )
    assert _select_best_learned_model(summary_df, "dft", split="test") == "adapter"


def test_float_cache_tag_is_stable_and_filename_safe():
    assert _float_cache_tag(0.967) == "0p967000"
    assert _float_cache_tag(-1.25) == "m1p250000"


def test_mode_assignment_drops_implausible_far_matches():
    pred = np.asarray([100.0, 200.0, 300.0], dtype=np.float64)
    target = np.asarray([105.0, 205.0, 800.0], dtype=np.float64)
    assign, mask = _mode_assignment(pred, target, max_match_cm=25.0)
    assert assign.tolist()[:2] == [0, 1]
    assert assign.tolist()[2] == -1
    assert mask.tolist() == [1.0, 1.0, 0.0]


def test_geometry_mol_features_are_fixed_width_and_finite():
    pos = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 1.0, 0.3]], dtype=np.float64)
    z = np.asarray([6, 1, 8], dtype=np.int64)
    feats = _geometry_mol_features(pos, z)
    assert feats.shape == (16,)
    assert np.isfinite(feats).all()


def test_peak_calibrator_line_predictions_preserve_sorted_intensity_pairing(tmp_path):
    x_grid = np.asarray([100.0, 200.0, 300.0], dtype=np.float32)
    dataset = DFTModeAlignmentDatasetBundle(
        domain="dft",
        x_grid=x_grid,
        mol_features=np.zeros((1, 16), dtype=np.float32),
        pred_freq=np.asarray([[100.0, 200.0]], dtype=np.float32),
        pred_intensity=np.asarray([[0.9, 0.2]], dtype=np.float32),
        pred_mask=np.asarray([[1.0, 1.0]], dtype=np.float32),
        target_freq=np.asarray([[200.0, 250.0]], dtype=np.float32),
        target_intensity=np.asarray([[0.2, 0.9]], dtype=np.float32),
        target_mask=np.asarray([[1.0, 1.0]], dtype=np.float32),
        match_target_idx=np.asarray([[0, 1]], dtype=np.int64),
        match_mask=np.asarray([[1.0, 1.0]], dtype=np.float32),
        y_pred_spec=np.zeros((1, 3), dtype=np.float32),
        y_target_spec=np.zeros((1, 3), dtype=np.float32),
        metadata=pd.DataFrame({"smiles": ["CC"]}),
        cache_npz=tmp_path / "dft.npz",
        cache_csv=tmp_path / "dft.csv",
    )
    calibrator = {
        "freq_delta": np.asarray([150.0, 0.0, 0.0], dtype=np.float32),
        "log_gain": np.zeros(3, dtype=np.float32),
    }
    pred = _apply_peak_calibrator_to_line_predictions(dataset, calibrator)
    assert np.allclose(pred["corrected_freq"][0], np.asarray([200.0, 250.0], dtype=np.float32))
    assert pred["corrected_intensity"][0, 0] < pred["corrected_intensity"][0, 1]


def test_run_alignment_study_smoke(tmp_path):
    x_grid = np.linspace(400.0, 1800.0, 64, dtype=np.float32)
    n_cases = 8
    centers = np.linspace(700.0, 1200.0, n_cases, dtype=np.float32)
    y_pred = []
    y_target = []
    for c in centers:
        base = np.exp(-0.5 * ((x_grid - c) / 30.0) ** 2) + 0.4 * np.exp(-0.5 * ((x_grid - (c + 180.0)) / 45.0) ** 2)
        shifted = np.exp(-0.5 * ((x_grid - (c + 8.0)) / 30.0) ** 2) + 0.45 * np.exp(-0.5 * ((x_grid - (c + 190.0)) / 48.0) ** 2)
        y_pred.append((base / base.max()).astype(np.float32))
        y_target.append((shifted / shifted.max()).astype(np.float32))
    y_pred = np.stack(y_pred).astype(np.float32)
    y_target = np.stack(y_target).astype(np.float32)
    mask = np.ones_like(y_pred, dtype=np.float32)
    mol_features = np.tile(np.linspace(0.1, 0.9, 16, dtype=np.float32), (n_cases, 1))
    metadata = pd.DataFrame({"component": [f"cmp_{i}" for i in range(n_cases)]})
    exp_bundle = AlignmentDatasetBundle(
        domain="experimental",
        x_grid=x_grid,
        y_pred=y_pred,
        y_target=y_target,
        mask=mask,
        mol_features=mol_features,
        metadata=metadata,
        cache_npz=tmp_path / "exp.npz",
        cache_csv=tmp_path / "exp.csv",
    )

    pred_freq = np.asarray([[700.0, 880.0, 1060.0]] * n_cases, dtype=np.float32)
    target_freq = pred_freq + 10.0
    pred_intensity = np.asarray([[1.0, 0.65, 0.4]] * n_cases, dtype=np.float32)
    target_intensity = np.asarray([[1.0, 0.7, 0.5]] * n_cases, dtype=np.float32)
    pred_mask = np.ones_like(pred_freq, dtype=np.float32)
    target_mask = np.ones_like(target_freq, dtype=np.float32)
    match_idx = np.asarray([[0, 1, 2]] * n_cases, dtype=np.int64)
    match_mask = np.ones_like(pred_freq, dtype=np.float32)
    dft_bundle = DFTModeAlignmentDatasetBundle(
        domain="dft",
        x_grid=x_grid,
        mol_features=mol_features,
        pred_freq=pred_freq,
        pred_intensity=pred_intensity,
        pred_mask=pred_mask,
        target_freq=target_freq,
        target_intensity=target_intensity,
        target_mask=target_mask,
        match_target_idx=match_idx,
        match_mask=match_mask,
        y_pred_spec=y_pred,
        y_target_spec=y_target,
        metadata=pd.DataFrame({"smiles": [f"C{i}" for i in range(n_cases)]}),
        cache_npz=tmp_path / "dft.npz",
        cache_csv=tmp_path / "dft.csv",
    )

    def _lines_to_spectrum(freq, inten, xg):
        y = np.zeros_like(xg, dtype=np.float32)
        for f, a in zip(freq, inten):
            y += a * np.exp(-0.5 * ((xg - f) / 18.0) ** 2)
        return y / max(float(y.max()), 1e-12)

    cfg = AlignmentTrainConfig(max_epochs=1, patience=1, batch_size=4, latent_dim=16, mol_latent_dim=8, cnn_channels=16, pooled_points=16, residual_basis=8, warp_basis=4)
    results = run_alignment_study(
        experimental_dataset=exp_bundle,
        dft_dataset=dft_bundle,
        out_dir=tmp_path / "alignment",
        device="cpu",
        train_config=cfg,
        dft_lines_to_spectrum_fn=_lines_to_spectrum,
    )

    assert (tmp_path / "alignment" / "alignment_summary.json").exists()
    assert Path(results["summary_json"]).exists()
    assert Path(results["domains"]["experimental"]["summary_csv"]).exists()
    assert Path(results["domains"]["dft"]["summary_csv"]).exists()
    exp_summary = pd.read_csv(results["domains"]["experimental"]["summary_csv"])
    dft_summary = pd.read_csv(results["domains"]["dft"]["summary_csv"])
    assert "joint_l2" in set(exp_summary["model"])
    assert "joint_l2" in set(dft_summary["model"])
