"""
One-time script to upload model weights to a Modal volume.

    python inference/upload_weights.py
"""
from pathlib import Path
import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
VOLUME_NAME = "raman-inference-weights"

WEIGHT_FILES = {
    # DetaNet checkpoints
    "depolar.pth": REPO_ROOT / "artifacts" / "spectra_queue" / "prodq-depolar-a100x8-20260219-044935" / "latest_depolar.pth",
    "Hi.pth": REPO_ROOT / "artifacts" / "hi" / "prod-hi-a10080x8-clean-20260224-182057" / "latest_Hi.pth",
    "Hij.pth": REPO_ROOT / "artifacts" / "hij" / "prod-hij-a10080x8-2ep-20260224-232300" / "latest_Hij.pth",
    # RefNet (best checkpoint)
    "refnet.pth": REPO_ROOT / "ramanchembl_pipeline" / "alignment_results" / "refinement_v9" / "es_step1400.pth",
    # Config (architecture params)
    "config.json": REPO_ROOT / "artifacts" / "spectra_queue" / "prodq-depolar-a100x8-20260219-044935" / "config.json",
}


def main():
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    for remote_name, local_path in WEIGHT_FILES.items():
        if not local_path.exists():
            print(f"MISSING: {local_path}")
            continue

    with vol.batch_upload() as upload:
        for remote_name, local_path in WEIGHT_FILES.items():
            if local_path.exists():
                upload.put_file(local_path, f"/{remote_name}")
                size_mb = local_path.stat().st_size / 1e6
                print(f"  {remote_name:20s} ({size_mb:.1f} MB) <- {local_path.name}")

    print(f"\nDone. Volume: {VOLUME_NAME}")


if __name__ == "__main__":
    main()
