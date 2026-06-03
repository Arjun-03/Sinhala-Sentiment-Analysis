"""
Sri Lankan slang normalization dictionary.
Maps common Sinhala/Singlish internet slang to plain English equivalents.
Add new entries here to extend coverage — no code changes needed.
"""

SLANG_DICT: dict[str, str] = {
    # Positive / praise
    "patta":    "excellent",
    "ela":      "great",
    "elakiri":  "excellent",
    "maru":     "awesome",
    "supiri":   "super",
    "hodai":    "good",
    "hoda":     "good",
    "wasi":     "awesome",
    "adurei":   "amazing",
    "pissu":    "crazy good",
    "baas":     "boss",
    "niyamai":  "nice",
    "niyama":   "nice",
    "lassana":  "beautiful",
    "wahey":    "wow great",

    # Negative / bad
    "boru":     "fake",
    "kunu":     "dirty bad",
    "yakko":    "idiot",
    "honda na": "not good",
    "narak":    "ugly bad",
    "berikam":  "stupid",
    "geri":     "bad quality",

    # Neutral / filler
    "machan":   "friend",
    "malli":    "brother",
    "akka":     "sister",
    "aiya":     "brother",
    "ane":      "oh my",
    "aiyo":     "oh no",
    "ammo":     "oh wow",
    "ada":      "today",
    "mokada":   "what is",
    "kohomada": "how is",
    "oyage":    "your",
    "api":      "we",
}


class SlangNormalizer:
    """Replaces Sri Lankan slang tokens with plain English equivalents."""

    def __init__(self, custom_dict: dict[str, str] | None = None):
        self._dict = {**SLANG_DICT, **(custom_dict or {})}
        # Pre-build a lowercase version for case-insensitive matching
        self._lower_dict = {k.lower(): v for k, v in self._dict.items()}

    def normalize(self, text: str) -> str:
        """
        Replace slang words/phrases in text.
        Works on whole tokens and multi-word phrases.
        Returns the modified text.
        """
        # Try multi-word phrases first (longest match wins)
        result = text
        for phrase in sorted(self._lower_dict, key=len, reverse=True):
            replacement = self._lower_dict[phrase]
            # Case-insensitive whole-phrase replacement
            import re
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            result = pattern.sub(replacement, result)
        return result
