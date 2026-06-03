"""
Sentiment classification using XLM-RoBERTa fine-tuned on Twitter data.

Model: cardiffnlp/twitter-xlm-roberta-base-sentiment
  - Multilingual (100 languages)
  - Fine-tuned on tweets → handles informal, emoji-heavy text well
  - Outputs: positive / negative / neutral + softmax confidence scores

Confidence thresholding:
  - If the top-label confidence < threshold → override with "neutral"
  - Default threshold: 0.55 (configurable)
"""

import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_DEFAULT_CONFIDENCE_THRESHOLD = 0.55

# The model uses these label IDs — map them to standard names
_LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral",
    # Some versions of the model use numeric or short labels:
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


class SentimentModel:
    """
    Wrapper around the XLM-RoBERTa sentiment classifier.

    Lazy-loaded: the model is only downloaded/loaded when classify() is
    first called, so startup is fast.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self._model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline

        logger.info("Loading sentiment model (first use — may take a moment)…")

        device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
        self._pipeline = pipeline(
            "text-classification",
            model=self._model_name,
            tokenizer=self._model_name,
            top_k=None,
            device=device,
        )

        logger.info("Sentiment model loaded.")

    def classify(self, text: str) -> dict:
        """
        Classify text and return:
          {
            "sentiment": "positive" | "negative" | "neutral",
            "confidence": float (0.0 – 1.0),
            "all_scores": {"positive": float, "negative": float, "neutral": float}
          }
        Falls back to neutral on any error.
        """
        if not text or not text.strip():
            return {"sentiment": "neutral", "confidence": 1.0, "all_scores": {}}

        self._load()

        try:
            # The pipeline returns [[{"label": ..., "score": ...}, ...]]
            results = self._pipeline(text[:512])  # truncate to model max
            scores_list = results[0]

            # Build a clean dict of scores
            all_scores = {}
            for item in scores_list:
                label = _LABEL_MAP.get(item["label"], item["label"].lower())
                all_scores[label] = round(item["score"], 4)

            # Pick the highest-scoring label
            best_label = max(all_scores, key=all_scores.__getitem__)
            best_score = all_scores[best_label]

            # Apply confidence threshold
            if best_score < self.confidence_threshold:
                return {
                    "sentiment": "neutral",
                    "confidence": best_score,
                    "all_scores": all_scores,
                }

            return {
                "sentiment": best_label,
                "confidence": best_score,
                "all_scores": all_scores,
            }

        except Exception as exc:
            logger.warning("Sentiment classification failed (%s)", exc)
            return {"sentiment": "neutral", "confidence": 0.0, "all_scores": {}}
