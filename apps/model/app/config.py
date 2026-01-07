import os
import sys

MODEL_CODE_DIR = os.getenv("MODEL_CODE_DIR", "/opt/detanet")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

if not os.path.isdir(MODEL_CODE_DIR):
    raise RuntimeError(f"MODEL_CODE_DIR does not exist: {MODEL_CODE_DIR}")

if MODEL_CODE_DIR not in sys.path:
    sys.path.insert(0, MODEL_CODE_DIR)
    os.chdir(MODEL_CODE_DIR)
