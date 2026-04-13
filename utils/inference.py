# latest_changes/utils/inference.py
"""
Runs Qwen2.5-VL inference using the chat-template format.
Only the newly generated tokens are returned (prompt tokens excluded).
"""

import logging
from typing import Optional

import torch
from PIL.Image import Image
from peft import PeftModel
from transformers import AutoProcessor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAX_NEW_TOKENS,
    NO_REPEAT_NGRAM_SIZE,
    NUM_BEAMS,
    REPETITION_PENALTY,
    VLM_INSTRUCTION,
)

logger = logging.getLogger(__name__)


def run_inference(
    image: Image,
    model: PeftModel,
    processor: AutoProcessor,
    instruction: Optional[str] = None,
) -> str:
    """
    Run Qwen2.5-VL inference on a single image.

    Args:
        image:       PIL Image (RGB).
        model:       Fine-tuned Qwen2.5-VL model with LoRA adapter.
        processor:   Corresponding AutoProcessor.
        instruction: Optional override for the text prompt.

    Returns:
        Generated caption string (prompt tokens excluded).
    """
    prompt = instruction or VLM_INSTRUCTION

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    # Build text using the model's chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # IMPORTANT: No truncation=True — breaks Qwen2.5-VL image token counts
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )

    # Strip prompt tokens — return only what the model generated
    prompt_len = inputs["input_ids"].shape[1]
    generated  = output_ids[0][prompt_len:]
    raw_text   = processor.decode(generated, skip_special_tokens=True).strip()

    logger.info("VLM output: %s", raw_text[:120])
    return raw_text
