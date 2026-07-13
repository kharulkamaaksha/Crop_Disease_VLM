# config.py
"""
Central configuration for the PlantVillage VLM pipeline.

Every value below can be overridden with an environment variable (or a local
`.env` file — copy `.env.example` to `.env` and edit it) so the same code
runs unmodified across Windows, macOS and Linux, and across different
machines/hardware. Environment variables always win; the values below are
just fallbacks. See README.md for the full list of variables.
"""

import os
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """
    Minimal, dependency-free .env loader. Populates os.environ from a
    KEY=VALUE file, without overriding variables already set in the shell.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ── Model Paths ──────────────────────────────────────────────────────────────
# BASE_MODEL_PATH / CLIP_MODEL_PATH are Hugging Face model IDs by default and
# are downloaded automatically on first run. ADAPTER_PATH must point at your
# own fine-tuned LoRA checkpoint — it is not included in this repo (too large
# for git). Set VLM_ADAPTER_PATH in your .env file to your local path.
BASE_MODEL_PATH: str = _env_str("VLM_BASE_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
ADAPTER_PATH: str = _env_str(
    "VLM_ADAPTER_PATH", str(PROJECT_ROOT / "checkpoints" / "qwen25vl_plantvillage_best")
)
CLIP_MODEL_PATH: str = _env_str("VLM_CLIP_MODEL_PATH", "openai/clip-vit-base-patch32")

# ── Compute device ───────────────────────────────────────────────────────────
# "auto" prefers CUDA, then Apple Silicon (MPS), then CPU.
# Override with VLM_DEVICE=cuda | mps | cpu to force a specific device.
DEVICE_PREFERENCE: str = _env_str("VLM_DEVICE", "auto").lower()

# ── OOD Detection ────────────────────────────────────────────────────────────
OOD_THRESHOLD: float = _env_float("VLM_OOD_THRESHOLD", 0.27)
OOD_TEXT_PROMPTS: List[str] = [
    "a photo of a plant leaf",
    "a close-up of a crop leaf",
    "a healthy or diseased plant leaf",
]

# ── Generation ───────────────────────────────────────────────────────────────
MAX_NEW_TOKENS: int = _env_int("VLM_MAX_NEW_TOKENS", 200)
NUM_BEAMS: int = _env_int("VLM_NUM_BEAMS", 4)
REPETITION_PENALTY: float = _env_float("VLM_REPETITION_PENALTY", 2.5)
NO_REPEAT_NGRAM_SIZE: int = _env_int("VLM_NO_REPEAT_NGRAM_SIZE", 6)

VLM_INSTRUCTION: str = (
    "Analyze this plant leaf image and provide a structured and explainable diagnosis.\n"
    "You MUST include the following fields clearly:\n"
    "- Plant: Name of the plant\n"
    "- Condition: Disease name or 'Healthy'\n"
    "- Severity: (None / Mild / Moderate / Severe)\n"
    "- Pathogen: Cause of the disease (fungus, bacteria, virus, or unknown)\n"
    "- Symptoms: Describe the visible symptoms in the leaf "
    "(spots, discoloration, lesions, patterns, etc.)\n"
    "- Explanation: Explain how the observed symptoms lead to the diagnosis\n\n"
    "Guidelines:\n"
    "- Be precise and concise\n"
    "- Base your diagnosis ONLY on visible features\n"
    "- Do not hallucinate unknown details\n"
    "- If uncertain, state 'Unknown' instead of guessing\n\n"
    "Output format example:\n"
    "Plant: Tomato\n"
    "Condition: Late Blight\n"
    "Severity: Moderate\n"
    "Pathogen: Fungus\n"
    "Symptoms: Dark brown irregular lesions with yellow edges visible on the leaf surface\n"
    "Explanation: The presence of dark lesions with surrounding yellow halos is "
    "characteristic of fungal infection such as Late Blight"
)

# ── Supported Crops ──────────────────────────────────────────────────────────
SUPPORTED_PLANTS: List[str] = ["Tomato", "Potato", "Pepper"]

# ── Confidence Score ─────────────────────────────────────────────────────────
CONFIDENCE_OFFSET: float = _env_float("VLM_CONFIDENCE_OFFSET", 0.30)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _env_str("VLM_LOG_LEVEL", "INFO").upper()
