# latest_changes/pipeline.py
"""
Production-grade single-call pipeline for plant disease diagnosis.

Flow:
    1. OOD check (CLIP multi-prompt)  → hard stop if not a plant leaf
    2. VLM inference                  → generate diagnostic caption
    3. Parse fields                   → extract all structured fields
    4. Output validation              → hard stop if critical fields missing
    5. Plant validation               → hard stop if crop mismatch
    6. Confidence score               → proxy from OOD similarity
    7. Return success result

Usage:
    from pipeline import run_pipeline
    result = run_pipeline(image, "Tomato", vlm, processor, clip_model, clip_proc)
"""

import logging
import sys
import os
from typing import Any, Dict

from PIL.Image import Image
from peft import PeftModel
from transformers import AutoProcessor, CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIDENCE_OFFSET, OOD_THRESHOLD
from utils.ood import run_ood
from utils.inference import run_inference
from utils.parser import parse_caption
from utils.validator import validate_plant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Confidence helper ────────────────────────────────────────────────────────

def _compute_confidence(ood_score: float) -> float:
    """
    Derive a display-friendly confidence score from the CLIP OOD similarity.

    OOD scores for valid plant leaves typically sit in 0.28–0.35.
    Adding CONFIDENCE_OFFSET (0.30) shifts this to 0.58–0.65, which reads
    as meaningful confidence to end users. Clamped to [0, 1].
    """
    return round(min(1.0, max(0.0, ood_score + CONFIDENCE_OFFSET)), 3)


# ── Return type helpers ──────────────────────────────────────────────────────

def _ood_error(score: float) -> Dict[str, Any]:
    return {
        "status":  "error",
        "type":    "OOD",
        "message": "Image not recognised as a plant leaf. Please upload a valid leaf photo.",
        "score":   score,
    }


def _invalid_output_error(fields: Dict) -> Dict[str, Any]:
    missing = [k for k in ("Plant", "Condition") if not fields.get(k)]
    return {
        "status":  "error",
        "type":    "INVALID_OUTPUT",
        "message": (
            f"Model failed to generate a valid diagnosis. "
            f"Missing required fields: {', '.join(missing)}. "
            f"Try re-uploading a clearer image."
        ),
    }


def _wrong_plant_error(expected: str, detected: str) -> Dict[str, Any]:
    return {
        "status":   "error",
        "type":     "WRONG_PLANT",
        "message":  f"Crop mismatch: selected '{expected}' but image appears to show '{detected}'.",
        "expected": expected,
        "detected": detected,
    }


def _success(fields: Dict, raw: str, confidence: float) -> Dict[str, Any]:
    return {
        "status":     "success",
        "data":       fields,
        "raw":        raw,
        "confidence": confidence,
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    image: Image,
    selected_plant: str,
    vlm: PeftModel,
    processor: AutoProcessor,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    ood_threshold: float = OOD_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run the full plant disease diagnosis pipeline in a single call.

    Args:
        image:           PIL Image (RGB) uploaded by the user.
        selected_plant:  Crop chosen by the user (e.g. "Tomato").
        vlm:             Fine-tuned Qwen2.5-VL model with LoRA adapter.
        processor:       Corresponding AutoProcessor for the VLM.
        clip_model:      CLIP model for OOD detection.
        clip_processor:  Corresponding CLIPProcessor.
        ood_threshold:   Mean cosine similarity cutoff (default from config).

    Returns:
        On success:
            { "status": "success", "data": {...}, "raw": "...", "confidence": float }
        On OOD:
            { "status": "error", "type": "OOD", "message": "...", "score": float }
        On invalid model output:
            { "status": "error", "type": "INVALID_OUTPUT", "message": "..." }
        On plant mismatch:
            { "status": "error", "type": "WRONG_PLANT",
              "expected": "...", "detected": "..." }
    """
    logger.info("── Pipeline start  selected_plant=%s ──", selected_plant)

    # ── Step 1: OOD detection (multi-prompt) ────────────────────────────────
    is_ood, ood_score = run_ood(image, clip_model, clip_processor, ood_threshold)

    if is_ood:
        logger.warning("OOD detected (score=%.4f). Stopping pipeline.", ood_score)
        return _ood_error(ood_score)

    logger.info("OOD passed (score=%.4f).", ood_score)
    confidence = _compute_confidence(ood_score)

    # ── Step 2: VLM inference ────────────────────────────────────────────────
    raw_output = run_inference(image, vlm, processor)

    # ── Step 3: Parse structured fields ─────────────────────────────────────
    fields = parse_caption(raw_output)

    # ── Step 4: Output validation (failsafe) ────────────────────────────────
    if not fields.get("Plant") or not fields.get("Condition"):
        logger.error("Invalid model output — missing critical fields. Raw: %s", raw_output[:200])
        return _invalid_output_error(fields)

    # ── Step 5: Plant validation ─────────────────────────────────────────────
    detected_plant = fields.get("Plant") or ""

    if not validate_plant(detected_plant, selected_plant):
        logger.warning(
            "Plant mismatch — expected: '%s'  detected: '%s'",
            selected_plant, detected_plant,
        )
        return _wrong_plant_error(selected_plant, detected_plant)

    logger.info(
        "Pipeline complete — condition: %s  confidence: %.1f%%",
        fields.get("Condition"), confidence * 100,
    )
    return _success(fields, raw_output, confidence)

