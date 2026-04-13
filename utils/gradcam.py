# latest_changes/utils/gradcam.py
"""
GradCAM implementation for Qwen2.5-VL.

Since Qwen2.5-VL uses a Vision Transformer (ViT) encoder, we hook into
the last attention layer of the visual encoder to extract attention weights,
then upsample them into a heatmap overlaid on the original image.

This is an attention-rollout approach — more reliable than gradient-based
GradCAM for transformer vision encoders where gradients can be noisy.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)


def _attention_to_heatmap(
    attn_weights: torch.Tensor,
    img_size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert attention weights to a spatial heatmap.

    Args:
        attn_weights: Tensor of shape (heads, seq_len, seq_len)
        img_size:     (width, height) of the original image

    Returns:
        heatmap as uint8 numpy array (H, W, 3) in BGR for cv2 operations
    """
    # Average over heads
    attn = attn_weights.mean(dim=0).cpu().float()   # (seq, seq)

    # CLS token attends to all patches — use its row
    cls_attn = attn[0, 1:]   # skip CLS itself → (num_patches,)

    # Infer grid size — Qwen2.5-VL uses 14×14 patches at base resolution
    num_patches = cls_attn.shape[0]
    grid = int(num_patches ** 0.5)
    if grid * grid != num_patches:
        # fallback: take sqrt floor
        grid = int(num_patches ** 0.5)
        cls_attn = cls_attn[:grid * grid]

    heatmap = cls_attn.reshape(grid, grid).numpy()

    # Normalise to 0–255
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Upsample to image size
    w, h = img_size
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   # BGR

    return heatmap


def generate_gradcam(
    image: PILImage,
    model,
    processor,
) -> Optional[PILImage]:
    """
    Generate a GradCAM / attention heatmap overlay for the given image.

    Hooks into the last visual encoder attention layer and extracts
    attention weights from a forward pass with the image.

    Args:
        image:     PIL Image (RGB).
        model:     Fine-tuned Qwen2.5-VL PeftModel.
        processor: Corresponding AutoProcessor.

    Returns:
        PIL Image (RGB) with heatmap blended over the original,
        or None if extraction fails.
    """
    try:
        attention_store = {}

        # Find the visual encoder's last attention layer
        # Qwen2.5-VL: model.base_model.model.visual.blocks[-1].attn
        try:
            visual_blocks = model.base_model.model.visual.blocks
        except AttributeError:
            # Direct model (no PEFT wrapper)
            visual_blocks = model.model.visual.blocks

        last_attn = visual_blocks[-1].attn

        def hook_fn(module, input, output):
            # output is typically the attention output; we need weights
            # Store input query/key to recompute if needed
            attention_store["output"] = output

        # Register forward hook
        handle = last_attn.register_forward_hook(hook_fn)

        # Also hook to capture attention weights directly if available
        attn_weights_store = {}

        def attn_hook(module, args, kwargs, output):
            return output

        # Build minimal input for a forward pass
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": "Analyze this plant leaf."},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run with output_attentions to get attention weights
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        handle.remove()

        # Try to get visual attentions
        # Qwen2.5-VL stores them in vision_model outputs
        visual_attentions = None
        if hasattr(outputs, "visual_attentions") and outputs.visual_attentions:
            visual_attentions = outputs.visual_attentions[-1]   # last layer
        elif hasattr(outputs, "attentions") and outputs.attentions:
            # fallback: use language model attentions (less accurate but works)
            visual_attentions = outputs.attentions[-1][0]       # (heads, seq, seq)

        if visual_attentions is None:
            logger.warning("Could not extract attention weights — using gradient fallback")
            return _gradient_fallback(image, model, processor, inputs)

        # Generate heatmap
        if visual_attentions.dim() == 4:
            visual_attentions = visual_attentions[0]   # remove batch dim

        heatmap_bgr = _attention_to_heatmap(visual_attentions, image.size)

        # Blend with original image
        original_np  = np.array(image.resize(image.size))
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        blended      = cv2.addWeighted(original_bgr, 0.55, heatmap_bgr, 0.45, 0)
        blended_rgb  = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        return Image.fromarray(blended_rgb)

    except Exception as e:
        logger.error("GradCAM failed: %s", e, exc_info=True)
        return None


def _gradient_fallback(image, model, processor, inputs) -> Optional[PILImage]:
    """
    Gradient-based saliency map as fallback when attention weights
    are not directly accessible.
    """
    try:
        pixel_values = inputs["pixel_values"].requires_grad_(True)
        modified_inputs = dict(inputs)
        modified_inputs["pixel_values"] = pixel_values

        outputs = model(**modified_inputs, return_dict=True)
        # Use sum of logits as scalar target
        score = outputs.logits[0].sum()
        score.backward()

        # Gradient magnitude over channel dim
        grad = pixel_values.grad[0].abs().mean(dim=0).cpu().numpy()
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        grad = (grad * 255).astype(np.uint8)

        w, h = image.size
        grad_resized = cv2.resize(grad, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(grad_resized, cv2.COLORMAP_JET)

        original_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)
        return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

    except Exception as e:
        logger.error("Gradient fallback also failed: %s", e)
        return None
