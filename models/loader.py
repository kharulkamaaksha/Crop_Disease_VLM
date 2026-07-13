# models/loader.py
"""
Loads the Qwen2.5-VL (+ LoRA adapter) and CLIP models onto the best available
device. Call load_vlm() and load_clip() once at startup; pass the returned
objects to run_pipeline() on every request.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADAPTER_PATH, BASE_MODEL_PATH, CLIP_MODEL_PATH, DEVICE_PREFERENCE

logger = logging.getLogger(__name__)

# Type aliases
VLMTuple  = Tuple[PeftModel, AutoProcessor]
CLIPTuple = Tuple[CLIPModel, CLIPProcessor]


def resolve_device() -> str:
    """
    Pick the best available compute device.

    VLM_DEVICE=auto (default) prefers CUDA, then Apple Silicon (MPS), then
    CPU. Set VLM_DEVICE=cuda|mps|cpu in .env to force a specific device.
    """
    if DEVICE_PREFERENCE in {"cuda", "mps", "cpu"}:
        return DEVICE_PREFERENCE
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for(device: str) -> torch.dtype:
    """bfloat16 on CUDA, float16 on MPS (better supported than bf16 there), float32 on CPU."""
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _check_adapter_path() -> None:
    if not Path(ADAPTER_PATH).exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at '{ADAPTER_PATH}'. "
            "Set VLM_ADAPTER_PATH in your .env file to the folder containing "
            "your fine-tuned checkpoint (see .env.example / README.md)."
        )


def load_vlm() -> VLMTuple:
    """
    Load the fine-tuned Qwen2.5-VL-7B-Instruct model with LoRA adapter.

    Returns:
        (model, processor) — both ready for inference on the resolved device.

    Raises:
        FileNotFoundError: if ADAPTER_PATH does not exist locally.
    """
    _check_adapter_path()

    device = resolve_device()
    dtype = _dtype_for(device)
    logger.info("Loading VLM processor from %s", ADAPTER_PATH)

    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

    logger.info(
        "Loading base model from %s (device=%s, dtype=%s)", BASE_MODEL_PATH, device, dtype
    )
    load_kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        # Let accelerate shard across GPU(s) automatically.
        load_kwargs["device_map"] = "auto"

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH, **load_kwargs
    )
    if device != "cuda":
        base_model = base_model.to(device)

    logger.info("Loading LoRA adapter from %s", ADAPTER_PATH)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("VLM ready on cuda  |  VRAM: %.1f GB", vram_gb)
    else:
        logger.info("VLM ready on %s", device)

    return model, processor


def load_clip() -> CLIPTuple:
    """
    Load CLIP ViT-B/32 for OOD detection.

    Returns:
        (clip_model, clip_processor) — model on the resolved device.
    """
    device = resolve_device()
    logger.info("Loading CLIP from %s (device=%s)", CLIP_MODEL_PATH, device)

    clip_model = CLIPModel.from_pretrained(
        CLIP_MODEL_PATH, use_safetensors=True
    ).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    clip_model.eval()

    logger.info("CLIP ready on %s", device)
    return clip_model, clip_processor
