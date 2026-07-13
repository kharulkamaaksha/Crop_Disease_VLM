#!/usr/bin/env python3
"""
Quick environment sanity check for the PlantVillage VLM pipeline.

Run this before `streamlit run app.py` to catch missing dependencies,
missing model paths, or device issues early, with a clear report instead
of a Streamlit stack trace.

Usage:
    python check_setup.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Maps the pip package name to the name used in `import ...`
REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "Pillow": "PIL",
    "streamlit": "streamlit",
}


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_python() -> None:
    print("Python")
    v = sys.version_info
    if v >= (3, 9):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python {v.major}.{v.minor}.{v.micro} — 3.9+ is recommended")


def check_packages() -> bool:
    print("\nDependencies")
    missing = []
    for pip_name, import_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
            _ok(pip_name)
        except ImportError:
            _fail(f"{pip_name} not installed")
            missing.append(pip_name)
    if missing:
        print("\n  Run: pip install -r requirements.txt")
    return not missing


def check_device() -> None:
    print("\nCompute device")
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_properties(0).name
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        _ok(f"CUDA available — {name} ({vram:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        _warn("Apple MPS available (no CUDA GPU found) — inference will be slower than on a discrete GPU")
    else:
        _warn("No GPU detected — running a 7B model on CPU will be very slow")


def check_model_paths() -> None:
    print("\nModel configuration")
    from config import ADAPTER_PATH, BASE_MODEL_PATH, CLIP_MODEL_PATH

    if Path(ADAPTER_PATH).exists():
        _ok(f"LoRA adapter found at {ADAPTER_PATH}")
    else:
        _fail(f"LoRA adapter NOT found at {ADAPTER_PATH}")
        print("         Set VLM_ADAPTER_PATH in your .env file to your checkpoint's location.")
    _ok(f"Base model: {BASE_MODEL_PATH} (downloaded from Hugging Face on first run)")
    _ok(f"CLIP model: {CLIP_MODEL_PATH} (downloaded from Hugging Face on first run)")


def main() -> None:
    print("=" * 60)
    print(" PlantMed AI — environment check")
    print("=" * 60)

    check_python()
    packages_ok = check_packages()

    if packages_ok:
        check_device()
        check_model_paths()
    else:
        print("\nSkipping device / model checks until dependencies are installed.")

    print("\n" + "=" * 60)
    print(" Once everything above looks good, run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
