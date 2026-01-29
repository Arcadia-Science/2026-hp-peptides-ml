from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional


_E3NN_MODULE = None
_E3NN_PATH = None


def _register_torch_safe_globals() -> None:
    try:
        import torch

        serialization = getattr(torch, "serialization", None)
        if serialization is not None and hasattr(serialization, "add_safe_globals"):
            serialization.add_safe_globals([slice])
    except Exception:
        # Best-effort for older torch versions or restricted environments.
        pass


def _vendored_elora_root() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "third_party" / "ELoRA"
    if candidate.is_dir():
        return str(candidate)
    return None


def _resolve_elora_path(elora_path: Optional[str]) -> Optional[str]:
    if not elora_path:
        return None
    if elora_path in ("vendored", "local"):
        resolved = _vendored_elora_root()
        if resolved is None:
            raise FileNotFoundError("Vendored ELoRA not found at third_party/ELoRA.")
        return resolved
    return os.path.abspath(elora_path)


def load_e3nn(elora_path: Optional[str] = None):
    """Load e3nn, optionally from an ELoRA repo path."""
    global _E3NN_MODULE, _E3NN_PATH

    resolved_path = _resolve_elora_path(elora_path)
    if resolved_path:
        e3nn_root = os.path.join(resolved_path, "e3nn")
        if not os.path.isdir(e3nn_root):
            raise FileNotFoundError(f"Expected e3nn package at {e3nn_root}.")
        if resolved_path not in sys.path:
            sys.path.insert(0, resolved_path)

    if "e3nn" in sys.modules:
        _E3NN_MODULE = sys.modules["e3nn"]
        _E3NN_PATH = getattr(_E3NN_MODULE, "__file__", None)
        if resolved_path and _E3NN_PATH and not _E3NN_PATH.startswith(resolved_path):
            raise RuntimeError(
                "e3nn is already imported from a different path. "
                "Import DetaNet after setting elora_path to use ELoRA."
            )
    else:
        _E3NN_MODULE = importlib.import_module("e3nn")
        _E3NN_PATH = getattr(_E3NN_MODULE, "__file__", None)
        if resolved_path and _E3NN_PATH and not _E3NN_PATH.startswith(resolved_path):
            raise RuntimeError(
                "Failed to load e3nn from elora_path. "
                "Ensure elora_path points to an ELoRA checkout with an e3nn package."
            )

    _register_torch_safe_globals()

    # Ensure common submodules are available as attributes on the e3nn package.
    for submodule in ("o3", "io", "nn"):
        if not hasattr(_E3NN_MODULE, submodule):
            setattr(_E3NN_MODULE, submodule, importlib.import_module(f"e3nn.{submodule}"))

    return _E3NN_MODULE
