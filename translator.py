"""
Sinhala → English translation using Facebook NLLB-200.

Model: facebook/nllb-200-distilled-600M
  - Supports 200 languages including Sinhala (sin_Sinh)
  - Distilled version is faster and smaller (~2.4 GB)
  - Runs on CPU or GPU

The model is loaded once and reused (lazy loading on first call).
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# NLLB language codes
_SINHALA_CODE = "sin_Sinh"
_ENGLISH_CODE = "eng_Latn"

_MODEL_NAME = "facebook/nllb-200-distilled-600M"


class Translator:
    """
    Translates Sinhala text to English using NLLB-200.

    The model is loaded lazily (only when translate() is first called)
    so startup time is fast when translation is not needed.
    """

    def __init__(self, model_name: str = _MODEL_NAME):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        """Load tokenizer and model on first use."""
        if self._model is not None:
            return  # already loaded

        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        logger.info("Loading NLLB translation model (first use — may take a moment)…")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()

        logger.info("Translation model loaded on %s", self._device)

    def translate(self, text: str, src_lang: str = _SINHALA_CODE, tgt_lang: str = _ENGLISH_CODE) -> str:
        """
        Translate text from src_lang to tgt_lang.
        Returns the translated English string, or the original text on failure.
        """
        if not text or not text.strip():
            return text

        self._load()

        import torch

        try:
            self._tokenizer.src_lang = src_lang
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            target_lang_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    forced_bos_token_id=target_lang_id,
                    max_new_tokens=256,
                    max_length=None,
                    num_beams=4,
                )

            translated = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return translated[0] if translated else text

        except Exception as exc:
            logger.warning("Translation failed (%s); using original text.", exc)
            return text
