"""
Text normalization for Sri Lankan social media comments.
Handles repeated characters, punctuation, whitespace — while keeping emojis intact.
"""

import re
import unicodedata


# Sinhala Unicode block: U+0D80 – U+0DFF
_SINHALA_RE = re.compile(r'[඀-෿]')

# Match any run of the SAME letter repeated more than 2 times
# e.g. "goooood" → "good", "elaaaa" → "elaa"
# We keep up to 2 repetitions so "cool" stays "cool" not "col"
_REPEATED_CHARS_RE = re.compile(r'(.)\1{2,}')

# Collapse runs of the same punctuation mark (e.g. "!!!" → "!")
_REPEATED_PUNCT_RE = re.compile(r'([!?.,;:])\1+')

# Collapse multiple spaces into one
_MULTI_SPACE_RE = re.compile(r' {2,}')


def _normalize_sinhala_repeated(text: str) -> str:
    """
    For Sinhala script, collapse repeated Unicode characters.
    Example: "සුපිරියීීී" → "සුපිරියී"
    """
    result = []
    prev = None
    count = 0
    for char in text:
        if _SINHALA_RE.match(char):
            if char == prev:
                count += 1
                if count <= 2:
                    result.append(char)
            else:
                result.append(char)
                prev = char
                count = 1
        else:
            result.append(char)
            prev = None
            count = 0
    return "".join(result)


class TextNormalizer:
    """
    Cleans raw social media text before sentiment analysis.

    What it does:
    - Lowercases ASCII/Latin characters (Sinhala script is unchanged)
    - Strips leading/trailing whitespace
    - Collapses repeated ASCII letters: "goooood" → "good"
    - Collapses repeated Sinhala characters
    - Collapses repeated punctuation: "!!!" → "!"
    - Preserves emojis exactly as-is
    """

    def normalize(self, text: str) -> str:
        if not text or not text.strip():
            return text

        # 1. Strip outer whitespace
        text = text.strip()

        # 2. Lowercase (only affects ASCII/Latin — Sinhala is unaffected)
        text = text.lower()

        # 3. Normalize Sinhala repeated chars (before ASCII repeated-char pass)
        text = _normalize_sinhala_repeated(text)

        # 4. Collapse repeated ASCII letters (keep max 2 in a row)
        #    "goooood" → "good", "elaaaa" → "elaa"
        text = _REPEATED_CHARS_RE.sub(r'\1\1', text)

        # 5. Collapse repeated punctuation
        text = _REPEATED_PUNCT_RE.sub(r'\1', text)

        # 6. Collapse multiple spaces
        text = _MULTI_SPACE_RE.sub(' ', text)

        return text
