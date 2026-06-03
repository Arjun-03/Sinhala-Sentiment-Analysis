"""
Main sentiment analysis pipeline for Sri Lankan social media comments.

Routing logic:
  1. Media-only (GIF/sticker) → infer from metadata
  2. Emoji-only               → emoji weighted scoring
  3. Text (possibly + emojis):
       a. Normalize text
       b. Apply slang normalization
       c. Detect language
       d. If pure Sinhala → translate → XLM-R
       e. If English / mixed → XLM-R directly
  4. Sarcasm heuristic applied after model prediction
  5. Confidence threshold: low confidence → neutral
"""

import logging
from dataclasses import dataclass, field

from normalization import TextNormalizer
from slang_dictionary import SlangNormalizer
from emoji_handler import (
    is_emoji_only,
    score_emojis,
    emoji_score_to_sentiment,
    emojis_to_words,
    has_sarcasm_emoji,
)
from language_detector import LanguageDetector
from translator import Translator
from sentiment_model import SentimentModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Media metadata sentiment inference
# ---------------------------------------------------------------------------

_MEDIA_SENTIMENT_MAP: dict[str, str] = {
    # Positive
    "laughing": "positive", "heart": "positive", "thumbs_up": "positive",
    "fire": "positive", "love": "positive", "clap": "positive",
    "wow": "positive", "celebrate": "positive", "smile": "positive",
    "cool": "positive",
    # Negative
    "angry": "negative", "sad": "negative", "cry": "negative",
    "thumbs_down": "negative", "disgusted": "negative", "sick": "negative",
    # Neutral
    "thinking": "neutral", "shrug": "neutral",
}


def _infer_media_sentiment(metadata: dict) -> str:
    """Return sentiment from sticker/GIF metadata fields."""
    keys_to_check = ["sticker_name", "gif_tags", "alt_text", "reaction_type"]
    for key in keys_to_check:
        value = metadata.get(key, "")
        if isinstance(value, list):
            value = " ".join(value)
        value_lower = str(value).lower()
        for keyword, sentiment in _MEDIA_SENTIMENT_MAP.items():
            if keyword in value_lower:
                return sentiment
    return "neutral"


# ---------------------------------------------------------------------------
# Sarcasm heuristic
# ---------------------------------------------------------------------------

_SARCASM_POSITIVE_WORDS = {"great", "nice", "good", "wow", "amazing", "excellent", "brilliant"}


def _apply_sarcasm_heuristic(text: str, model_sentiment: str) -> str:
    """
    Flip positive → negative when sarcasm signals are present:
    - sarcasm-associated emoji (💀 🤡 😂 🤣) combined with a positive word
    - pattern: positive model result but text looks sarcastic
    """
    if model_sentiment != "positive":
        return model_sentiment

    if not has_sarcasm_emoji(text):
        return model_sentiment

    # Check if the text also has a positive-sounding word — classic sarcasm pattern
    words = set(text.lower().split())
    if words & _SARCASM_POSITIVE_WORDS:
        logger.debug("Sarcasm heuristic triggered on: %s", text)
        return "negative"

    return model_sentiment


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    original_text: str
    normalized_text: str
    detected_language: str
    translated_text: str
    emoji_score: int | None
    model_sentiment: str
    confidence: float
    final_sentiment: str

    def to_dict(self) -> dict:
        return {
            "original_text":    self.original_text,
            "normalized_text":  self.normalized_text,
            "detected_language": self.detected_language,
            "translated_text":  self.translated_text,
            "emoji_score":      self.emoji_score,
            "model_sentiment":  self.model_sentiment,
            "confidence":       self.confidence,
            "final_sentiment":  self.final_sentiment,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SentimentPipeline:
    """
    End-to-end sentiment analysis pipeline.

    Components are lazy-loaded — models are only downloaded/loaded
    when the pipeline first processes a comment.
    """

    def __init__(self, confidence_threshold: float = 0.55):
        self._normalizer    = TextNormalizer()
        self._slang         = SlangNormalizer()
        self._lang_detector = LanguageDetector()
        self._translator    = Translator()
        self._model         = SentimentModel(confidence_threshold=confidence_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str, media_metadata: dict | None = None) -> SentimentResult:
        """
        Analyze a single comment and return a SentimentResult.

        Args:
            text:           The raw comment string (may be empty for media-only).
            media_metadata: Optional dict with keys like sticker_name, gif_tags, etc.
        """
        # Sanitize surrogate characters that Windows stdin can introduce
        original = (text or "").encode("utf-8", errors="replace").decode("utf-8")

        # ── 1. Media-only (no text) ─────────────────────────────────────────
        if media_metadata and not original.strip():
            sentiment = _infer_media_sentiment(media_metadata)
            return SentimentResult(
                original_text=original,
                normalized_text="",
                detected_language="media",
                translated_text="",
                emoji_score=None,
                model_sentiment=sentiment,
                confidence=1.0,
                final_sentiment=sentiment,
            )

        # ── 2. Emoji-only ───────────────────────────────────────────────────
        if is_emoji_only(original):
            score = score_emojis(original)
            sentiment = emoji_score_to_sentiment(score)
            return SentimentResult(
                original_text=original,
                normalized_text=original,
                detected_language="emoji-only",
                translated_text="",
                emoji_score=score,
                model_sentiment=sentiment,
                confidence=1.0,
                final_sentiment=sentiment,
            )

        # ── 3. Text (with or without emojis) ────────────────────────────────

        # 3a. Normalize
        normalized = self._normalizer.normalize(original)

        # 3b. Slang normalization
        normalized = self._slang.normalize(normalized)

        # 3c. Convert emojis to words for NLP (keeps Sinhala script intact)
        text_for_model = emojis_to_words(normalized)

        # 3d. Detect language (on emoji-free version)
        import emoji as _emoji_lib
        text_no_emoji = _emoji_lib.replace_emoji(normalized, replace=' ').strip()
        detected_lang = self._lang_detector.detect(text_no_emoji)

        # 3e. Compute emoji score (informational)
        emoji_score = score_emojis(original)

        # 3f. Route by language
        translated_text = ""

        if detected_lang == "sinhala":
            # Translate Sinhala → English, then classify
            translated_text = self._translator.translate(text_for_model)
            classify_input  = translated_text
        else:
            # English or mixed → classify directly
            classify_input = text_for_model

        # 3g. Model classification
        model_result   = self._model.classify(classify_input)
        model_sentiment = model_result["sentiment"]
        confidence      = model_result["confidence"]

        # 3h. Sarcasm heuristic (uses original text for emoji detection)
        final_sentiment = _apply_sarcasm_heuristic(original, model_sentiment)

        return SentimentResult(
            original_text=original,
            normalized_text=normalized,
            detected_language=detected_lang,
            translated_text=translated_text,
            emoji_score=emoji_score,
            model_sentiment=model_sentiment,
            confidence=round(confidence, 4),
            final_sentiment=final_sentiment,
        )
