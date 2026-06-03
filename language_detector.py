"""
Language detection for Sri Lankan social media comments.

Detects:
  - "english"  → English (or Singlish) only, no Sinhala script
  - "sinhala"  → Sinhala script only
  - "mixed"    → both Sinhala script and English characters present
"""

import re

_SINHALA_CHAR_RE = re.compile(r'[඀-෿]')


def _sinhala_char_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    sinhala_chars = [c for c in chars if _SINHALA_CHAR_RE.match(c)]
    return len(sinhala_chars) / len(chars)


class LanguageDetector:
    """
    Returns 'english', 'sinhala', or 'mixed' based on the characters present.
    """

    def detect(self, text: str) -> str:
        if not text or not text.strip():
            return "english"

        sinhala_ratio = _sinhala_char_ratio(text)

        # --- Has Sinhala characters ---
        has_sinhala = sinhala_ratio > 0
        has_latin = bool(re.search(r'[a-zA-Z]', text))

        if has_sinhala and has_latin:
            return "mixed"
        if has_sinhala:
            return "sinhala"
        return "english"
