"""
Emoji detection, scoring, and text conversion for Sri Lankan social media.

Three responsibilities:
  1. Detect emoji-only comments
  2. Score emoji-only comments with a weighted dictionary
  3. Convert emojis to words when text + emojis are mixed
"""

import re
import emoji


# ---------------------------------------------------------------------------
# Weighted emoji sentiment scores
# Positive = +N, Negative = -N, Neutral = 0
# ---------------------------------------------------------------------------
EMOJI_SCORES: dict[str, int] = {
    # Strong positive
    "❤️": 3, "🥰": 3, "😍": 3, "🤩": 3, "💯": 3, 
    "👑️": 2, "👑": 2, "👍": 2, "🔥": 2, "🎉": 2, 
    "🙏": 2, "😊": 2, "😁": 2, "🙌": 2, "✨": 2, 
    "💖": 2, "👏": 2, "🥳": 2, "😎": 2, "💪": 2,
    "🌷": 1, "❤": 1, "💪": 1,

    # Mild positive
    "🙂":  1, "😄": 1, "😀": 1, "💙": 1, "💚": 1,
    "🤗":  1, "😇": 1, "🌟": 1, "⭐": 1,

    # Neutral
    "😐":  0, "🤔": 0, "😶": 0, "👀": 0, "🤷": 0,

    # Mild negative
    "😕": -1, "😞": -1, "😔": -1, "🙁": -1,

    # Negative
    "😭": -2, "😢": -2, "😤": -2, "😠": -2, "👎": -2,
    "💔": -2, "😒": -2, "🤦": -2, "😂": -2, "🤣": -2,

    # Strong negative
    "😡": -3, "🤬": -3, "🤮": -4, "🤢": -3, "💀": -3,
    "🤡": -3,  # sarcasm marker — treated mildly negative
}

# Sarcasm-associated emojis (used in sarcasm detection heuristic)
SARCASM_EMOJIS: set[str] = {"💀", "🤡", "😂", "🤣"}

# Regex that matches any non-whitespace, non-emoji character
# Used to decide if a string is "emoji only"
_NON_EMOJI_TEXT_RE = re.compile(r'[^\s]')


def _extract_emojis(text: str) -> list[str]:
    """Return a list of every emoji character found in text."""
    return [token['emoji'] for token in emoji.emoji_list(text)]


def is_emoji_only(text: str) -> bool:
    """
    Return True if the comment contains only emojis (and whitespace).
    Pure emoji examples: "😂😂😂", "❤️🔥"
    """
    if not text.strip():
        return False
    # Remove all emojis; if nothing non-whitespace remains → emoji-only
    stripped = emoji.replace_emoji(text, replace='')
    return not _NON_EMOJI_TEXT_RE.search(stripped)


def score_emojis(text: str) -> int:
    """
    Compute a weighted sentiment score for emojis in text.
    Repeated emojis accumulate (🔥🔥🔥 = +6).
    Returns an integer score (positive, negative, or 0).
    """
    total = 0
    for e in _extract_emojis(text):
        total += EMOJI_SCORES.get(e, 0)
    return total


def emoji_score_to_sentiment(score: int) -> str:
    """Map a numeric emoji score to a sentiment label."""
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def emojis_to_words(text: str) -> str:
    """
    Replace emojis with their English description words.
    Example: "සුපිරි 🔥❤️" → "සුපිරි fire red heart"
    Uses emoji.demojize() then strips the :colon: wrappers.
    """
    demojized = emoji.demojize(text, delimiters=(" ", " "))
    # demojize adds spaces + underscores; clean up for readability
    demojized = demojized.replace("_", " ")
    # Collapse multiple spaces that may have appeared
    demojized = re.sub(r' {2,}', ' ', demojized).strip()
    return demojized


def has_sarcasm_emoji(text: str) -> bool:
    """Return True if the text contains a sarcasm-associated emoji."""
    emojis_in_text = set(_extract_emojis(text))
    return bool(emojis_in_text & SARCASM_EMOJIS)
