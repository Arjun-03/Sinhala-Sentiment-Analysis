"""
FastAPI for the Sri Lankan Sentiment Analysis pipeline.
Run with: uvicorn api:app --reload

Endpoint:
  POST /analyze
  Body:  { "comment": "your text here" }
  Returns the sentiment result JSON.
"""

import os
import warnings
import logging

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import SentimentPipeline

app = FastAPI(title="Sri Lankan Sentiment Analysis API")

# Load pipeline once when the server starts
pipeline = SentimentPipeline(confidence_threshold=0.55)


class CommentRequest(BaseModel):
    comment: str


@app.post("/analyze")
def analyze(request: CommentRequest):
    result = pipeline.analyze(request.comment)
    return result.to_dict()
