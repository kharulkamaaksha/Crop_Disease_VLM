# latest_changes/utils/validator.py
"""
Validates that the plant detected by the VLM matches the user-selected crop.

Normalisation strips noise words ("leaf", "plant", "crop") before comparing,
so edge cases like "Cherry tomato leaf" or "Tomato plant infected..." still
match correctly against "Tomato".
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Words that add no discriminative information for crop-type matching
_NOISE_WORDS = re.compile(
    r"\b(leaf|leaves|plant|plants|crop|crops|image|photo|picture)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Strip noise words and whitespace for a clean comparison."""
    cleaned = _NOISE_WORDS.sub("", text)
    return " ".join(cleaned.lower().split())   # collapse extra spaces


def validate_plant(detected: Optional[str], selected: str) -> bool:
    """
    Check whether the VLM-detected plant matches the user's selection.

    Comparison is done after normalising both strings (lowercase + noise
    word removal), so minor phrasing differences don't cause false failures.

    Args:
        detected: Plant name extracted from the VLM output (may be None).
        selected: Plant name chosen by the user in the UI.

    Returns:
        True if they match; False otherwise.
    """
    if not detected:
        logger.warning("No plant detected in VLM output — validation failed.")
        return False

    norm_selected = _normalize(selected)
    norm_detected = _normalize(detected)

    match = norm_selected in norm_detected

    logger.info(
        "Plant validation — selected: '%s' → '%s'  detected: '%s' → '%s'  match: %s",
        selected, norm_selected, detected, norm_detected, match,
    )
    return match
