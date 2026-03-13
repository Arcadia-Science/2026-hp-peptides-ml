"""
Alignment pipeline invariants.

Defines the domain types and acceptance thresholds derived from the statistical
audit of the depolar_spectra_pipeline results (n=160 molecules, 4147 DFT peaks).

Key finding: the "5 cm^-1 median" claim is a conditional statistic on the 13.2%
of matched peaks.  The honest coverage-penalised MAE baseline is 9.34 cm^-1.
The PRIMARY goal of the alignment model is to improve coverage@10, not just
conditional accuracy.

Acceptance criteria (what "working" means):
  - F1@10 >= 0.50            (from baseline ~0.13)
  - coverage@10 >= 0.60      (from baseline 0.234)
  - cwmae@10 <= 5.0 cm^-1   (from baseline 9.34 cm^-1)
  - intensity_mae <= 0.25    (log10 units; baseline 0.412)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class RamanObservable:
    """Measured or model-predicted Raman spectrum."""
    molecule: str            # SMILES string
    wavenumbers: List[float] # cm^-1 x-axis (peaks or full grid)
    intensities: List[float] # normalised intensities


@dataclass
class DFTRamanObservable:
    """DFT ground-truth Raman data for a molecule."""
    molecule: str                # SMILES string
    structure: Dict[str, Any]    # coord, atoms, vib coord from DB blob
    frequencies: List[float]     # DFT normal-mode frequencies (cm^-1)
    raman_activities: List[float] # DFT Raman activities (Ang^4/amu)
    spectrum_x: List[float]      # broadened spectrum x-axis
    spectrum_y: List[float]      # broadened + normalised spectrum y-values


# ---------------------------------------------------------------------------
# Acceptance thresholds
# Derived from statistical audit (2026-03-12). Update when baseline improves.
# ---------------------------------------------------------------------------

# Primary (coverage-honest) — denominator is ALL DFT target peaks
# Theoretical recall ceiling = n_pred / n_target = 3474 / 4147 = 83.8%
# Target: recall@10 >= 0.80 (95.5% of predicted modes must match a DFT mode)
RECALL_AT_10_MIN   = 0.80   # recall = coverage = fraction of DFT peaks recovered
F1_AT_10_MIN       = 0.60   # F1 implied by recall=0.80 and reasonable precision
COVERAGE_AT_10_MIN = 0.80   # same as RECALL_AT_10_MIN (coverage = recall here)
CWMAE_AT_10_MAX    = 3.0    # cm^-1 — if recall=0.80, honest CWMAE must be low

# Secondary (conditional on matched pairs)
INTENSITY_MAE_MAX  = 0.25   # log10(pred/DFT) — baseline is 0.412
POINT_RMSE_MAX     = 8.0    # cm^-1 — on matched pairs only

# Baseline numbers from pre-correction depolar pipeline (for delta reporting)
BASELINE_F1_AT_10      = 0.132
BASELINE_COVERAGE_10   = 0.234
BASELINE_CWMAE_10      = 9.34   # cm^-1
BASELINE_INTENSITY_MAE = 0.412


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------

def hungarian_match(
    pred_peaks: List[tuple],
    target_peaks: List[tuple],
    tol_cm: float = 10.0,
) -> tuple:
    """
    One-to-one Hungarian matching of predicted peaks to target peaks.

    Args:
        pred_peaks:   list of (wavenumber, intensity) for predicted spectrum
        target_peaks: list of (wavenumber, intensity) for DFT ground truth
        tol_cm:       only keep pairs within this frequency tolerance

    Returns:
        (matched_pred, matched_target) — lists of (wavenumber, intensity) pairs.
    """
    if not pred_peaks or not target_peaks:
        return [], []

    pf = np.array([p[0] for p in pred_peaks], dtype=np.float64)
    tf = np.array([t[0] for t in target_peaks], dtype=np.float64)

    cost = np.abs(pf[:, None] - tf[None, :])
    row_idx, col_idx = linear_sum_assignment(cost)
    keep = cost[row_idx, col_idx] <= tol_cm

    matched_pred   = [pred_peaks[i]   for i in row_idx[keep]]
    matched_target = [target_peaks[j] for j in col_idx[keep]]
    return matched_pred, matched_target


def equivalence_check(
    obs: RamanObservable,
    dft: DFTRamanObservable,
    tol_cm: float = 10.0,
) -> Dict[str, Any]:
    """
    Coverage-honest alignment check between a predicted and DFT spectrum.

    Returns a dict with:
      n_pred, n_target, n_matched
      coverage@tol     — fraction of DFT peaks recovered
      mean_abs_dnu     — conditional mean |delta_nu| on matched pairs (cm^-1)
      cwmae_cm         — coverage-weighted MAE (unmatched penalised at tol_cm)
      passes           — True if primary acceptance criteria met
    """
    if obs.molecule != dft.molecule:
        return {"passes": False, "reason": "molecule mismatch"}

    pred_peaks   = list(zip(obs.wavenumbers, obs.intensities))
    target_peaks = list(zip(dft.spectrum_x, dft.spectrum_y))

    matched_pred, matched_target = hungarian_match(pred_peaks, target_peaks, tol_cm)

    n_pred    = len(pred_peaks)
    n_target  = len(target_peaks)
    n_matched = len(matched_pred)

    coverage = n_matched / n_target if n_target > 0 else 0.0

    if n_matched > 0:
        abs_dnu = np.abs(
            np.array([p[0] for p in matched_pred]) -
            np.array([t[0] for t in matched_target])
        )
        mean_abs_dnu = float(np.mean(abs_dnu))
    else:
        mean_abs_dnu = float(tol_cm)

    # CWMAE: unmatched targets penalised at tol_cm
    cwmae = (mean_abs_dnu * n_matched + tol_cm * (n_target - n_matched)) / max(n_target, 1)

    passes = (coverage >= COVERAGE_AT_10_MIN) and (cwmae <= CWMAE_AT_10_MAX)

    return {
        "n_pred": n_pred,
        "n_target": n_target,
        "n_matched": n_matched,
        f"coverage@{int(tol_cm)}": coverage,
        "mean_abs_dnu_cm": mean_abs_dnu,
        "cwmae_cm": cwmae,
        "passes": passes,
    }
