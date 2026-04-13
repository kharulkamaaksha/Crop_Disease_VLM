# latest_changes/utils/parser.py
"""
Extracts structured fields from the VLM's free-text output using regex.
Fields: Plant, Condition, Severity, Pathogen (optional).
"""

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Fields to extract and their regex patterns.
# Symptoms and Explanation allow longer values (up to the next field header or end of string).
_FIELD_PATTERNS: Dict[str, str] = {
    "Plant":       r"Plant\s*[:\-]\s*([^.\n]+)",
    "Condition":   r"Condition\s*[:\-]\s*([^.\n]+)",
    "Severity":    r"Severity\s*[:\-]\s*([^.\n]+)",
    "Pathogen":    r"Pathogen\s*[:\-]\s*([^.\n]+)",
    "Symptoms":    r"Symptoms\s*[:\-]\s*(.+?)(?=\n[A-Z][a-z]+\s*[:\-]|\Z)",
    "Explanation": r"Explanation\s*[:\-]\s*(.+?)(?=\n[A-Z][a-z]+\s*[:\-]|\Z)",
}


def parse_caption(text: str) -> Dict[str, Optional[str]]:
    """
    Parse structured fields from VLM-generated text.

    Args:
        text: Raw generated string from the VLM.

    Returns:
        Dictionary with keys: Plant, Condition, Severity, Pathogen,
        Symptoms, Explanation. Missing fields are set to None.
    """
    fields: Dict[str, Optional[str]] = {}

    for field, pattern in _FIELD_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip().rstrip(".,;")
            fields[field] = value
        else:
            fields[field] = None
            logger.debug("Field '%s' not found in output.", field)

    logger.info("Parsed fields: %s", fields)
    return fields
