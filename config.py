# latest_changes/config.py
"""
Central configuration for the PlantVillage VLM pipeline.
All constants and paths are defined here — modify this file only.
"""

from typing import List

# ── Model Paths ──────────────────────────────────────────────────────────────
BASE_MODEL_PATH: str = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH: str    = r"C:\Users\Admin\Desktop\VLM\qwen\outputs\qwen25vl_plantvillage\best"
CLIP_MODEL_PATH: str = "openai/clip-vit-base-patch32"

# ── OOD Detection ────────────────────────────────────────────────────────────
OOD_THRESHOLD: float        = 0.27
OOD_TEXT_PROMPTS: List[str] = [
    "a photo of a plant leaf",
    "a close-up of a crop leaf",
    "a healthy or diseased plant leaf",
]

# ── Generation ───────────────────────────────────────────────────────────────
MAX_NEW_TOKENS: int       = 200
NUM_BEAMS: int            = 4
REPETITION_PENALTY: float = 2.5
NO_REPEAT_NGRAM_SIZE: int = 6

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
CONFIDENCE_OFFSET: float = 0.30

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"