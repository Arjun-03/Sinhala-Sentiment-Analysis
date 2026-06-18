"""
Main sentiment analysis pipeline for Sri Lankan social media comments.

Routing logic:
  1. Emoji-only               → emoji weighted scoring
  2. Text (possibly + emojis):
       a. Normalize text
       b. Apply slang normalization
       c. Detect language
       d. If pure Sinhala or mixed → translate → XLM-R
       e. If pure English → XLM-R directly
  3. Sarcasm heuristic applied after model prediction
  4. Confidence threshold: low confidence → neutral
"""

import logging
from dataclasses import dataclass

import emoji as _emoji_lib

from normalization import TextNormalizer
from slang_dictionary import SlangNormalizer
from emoji_handler import (
    is_emoji_only,
    score_emojis,
    emoji_score_to_sentiment,
    emoji_confidence,
    has_sarcasm_emoji,
)
from language_detector import LanguageDetector
from translator import Translator
from sentiment_model import SentimentModel

logger = logging.getLogger(__name__)


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

    def analyze(self, text: str) -> SentimentResult:
        """Analyze a single comment and return a SentimentResult."""
        # Sanitize surrogate characters that Windows stdin can introduce
        original = (text or "").encode("utf-8", errors="replace").decode("utf-8")

        # ── 1. Emoji-only ───────────────────────────────────────────────────
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
                confidence=emoji_confidence(score),
                final_sentiment=sentiment,
            )

        # ── 2. Text (with or without emojis) ────────────────────────────────

        # 2a. Normalize
        normalized = self._normalizer.normalize(original)

        # 2b. Slang normalization
        normalized = self._slang.normalize(normalized)

        text_for_model = normalized

        # 2c. Detect language (on emoji-free version)
        text_no_emoji = _emoji_lib.replace_emoji(normalized, replace=' ').strip()
        detected_lang = self._lang_detector.detect(text_no_emoji)

        # 2d. Compute emoji score (informational)
        emoji_score = score_emojis(original)

        # 2e. Route by language
        translated_text = ""

        if detected_lang in ("sinhala", "mixed"):
            # Preserve emojis before translation — the tokenizer drops them
            emojis_in_text = "".join(tok['emoji'] for tok in _emoji_lib.emoji_list(text_for_model))
            translated_text = self._translator.translate(text_for_model)
            classify_input  = (translated_text + " " + emojis_in_text).strip()
            translated_text = classify_input
        else:
            # Pure English → classify directly
            classify_input = text_for_model

        # 2f. Model classification
        model_result   = self._model.classify(classify_input)
        model_sentiment = model_result["sentiment"]
        confidence      = model_result["confidence"]

        # 2g. Sarcasm heuristic (uses original text for emoji detection)
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
