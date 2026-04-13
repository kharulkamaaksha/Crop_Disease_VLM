# latest_changes/utils/ood.py
"""
Out-of-Distribution (OOD) detection using CLIP cosine similarity.

Uses MULTI-PROMPT averaging for much stronger filtering:
  - Single prompt ("a photo of a plant leaf") lets random green objects pass.
  - Averaging across 3 complementary prompts raises the bar significantly,
    forcing the image to align with the concept of a plant leaf from
    multiple semantic angles.

Images that are not plant leaves are blocked here before VLM inference runs.
"""

import logging
from typing import List, Tuple

import torch
from PIL.Image import Image
from transformers import CLIPModel, CLIPProcessor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OOD_TEXT_PROMPTS

logger = logging.getLogger(__name__)


def run_ood(
    image: Image,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    threshold: float,
) -> Tuple[bool, float]:
    """
    Compute averaged CLIP cosine similarity between the image and multiple
    plant-leaf reference prompts. Returns a (is_ood, score) tuple.

    Multi-prompt strategy:
        score = mean( cosine_sim(img, prompt_i) for prompt_i in PROMPTS )

    This is significantly harder to fool than a single prompt — a random
    green object might score well against one prompt but poorly against
    the others, pulling the average below the threshold.

    Args:
        image:          PIL Image (RGB).
        clip_model:     Loaded CLIPModel (on CUDA).
        clip_processor: Loaded CLIPProcessor.
        threshold:      Mean similarity cutoff; below this → OOD.

    Returns:
        is_ood (bool):  True if the image is NOT a plant leaf.
        score  (float): Averaged cosine similarity (rounded to 4 dp).
    """
    device = next(clip_model.parameters()).device

    inputs = clip_processor(
        text=OOD_TEXT_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)

        # Shape: (1, D) and (num_prompts, D)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds

        # L2-normalise both
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        # img_emb: (1, D)  txt_emb: (num_prompts, D)
        # Matrix multiply → (1, num_prompts), then take mean
        per_prompt_sim = (img_emb @ txt_emb.T).squeeze(0)   # (num_prompts,)
        similarity     = per_prompt_sim.mean().item()

    is_ood = similarity < threshold
    score  = round(similarity, 4)

    logger.info(
        "OOD check — avg_score: %.4f  per_prompt: %s  threshold: %.2f  ood: %s",
        score,
        [round(s, 4) for s in per_prompt_sim.tolist()],
        threshold,
        is_ood,
    )

    return is_ood, score
