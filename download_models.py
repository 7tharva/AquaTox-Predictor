import os
import gdown
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"

DRIVE_FILES = {
    "clf_models.joblib":      "15cclmVtcafuEE6pS8DpG6Z24z5AOiLrU",
    "reg_models.joblib":      "1kjnbuomMU8rPVHAT8KhQ9ZIIk1fP6AEN",
    "feature_columns.joblib": "1G_Wf63sH3459RwmjfmrEyhQ7-rK9Htgd",
}

def download_models():
    MODEL_DIR.mkdir(exist_ok=True)
    all_present = all((MODEL_DIR / f).exists() for f in DRIVE_FILES)
    if all_present:
        print("  Models already present — skipping download")
        return

    print("  Downloading models from Google Drive...")
    for filename, file_id in DRIVE_FILES.items():
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"  {filename} already exists — skipping")
            continue
        print(f"  Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest), quiet=False)
        print(f"  {filename} downloaded ✓")

    print("  All models downloaded!")

if __name__ == "__main__":
    download_models()