# latest_changes/models/loader.py
"""
Loads the Qwen2.5-VL (+ LoRA adapter) and CLIP models onto GPU.
Call load_vlm() and load_clip() once at startup; pass the returned
objects to run_pipeline() on every request.
"""

import logging
from typing import Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADAPTER_PATH, BASE_MODEL_PATH, CLIP_MODEL_PATH

logger = logging.getLogger(__name__)

# Type aliases
VLMTuple  = Tuple[PeftModel, AutoProcessor]
CLIPTuple = Tuple[CLIPModel, CLIPProcessor]


def load_vlm() -> VLMTuple:
    """
    Load the fine-tuned Qwen2.5-VL-7B-Instruct model with LoRA adapter.

    Returns:
        (model, processor) — both ready for inference on CUDA.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading VLM processor from %s", ADAPTER_PATH)

    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

    logger.info("Loading base model from %s", BASE_MODEL_PATH)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Loading LoRA adapter from %s", ADAPTER_PATH)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if device == "cuda" else 0
    logger.info("VLM ready on %s  |  VRAM: %.1f GB", device, vram_gb)

    return model, processor


def load_clip() -> CLIPTuple:
    """
    Load CLIP ViT-B/32 for OOD detection.

    Returns:
        (clip_model, clip_processor) — model on CUDA if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading CLIP from %s", CLIP_MODEL_PATH)

    clip_model = CLIPModel.from_pretrained(
        CLIP_MODEL_PATH, use_safetensors=True
    ).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    clip_model.eval()

    logger.info("CLIP ready on %s", device)
    return clip_model, clip_processor
