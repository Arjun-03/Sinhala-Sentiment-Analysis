"""
Interactive console for the Sri Lankan Sentiment Analysis pipeline.
Run with: python main.py
Type a comment and press Enter to analyze. Type 'quit' to exit.
"""

import sys
import json
import logging
import os
import warnings

# Silence HuggingFace and other noisy libraries
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)

# Force UTF-8 on Windows for stdin, stdout, and stderr
if hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from pipeline import SentimentPipeline


def print_result(result) -> None:
    output = json.dumps(result.to_dict(), ensure_ascii=False, indent=0)
    sys.stdout.buffer.write((output + "\n").encode("utf-8"))


def main():
    pipeline = SentimentPipeline(confidence_threshold=0.55)

    print("\n" + "=" * 60)
    print("   Sri Lankan Sentiment Analysis")
    print("   Type a comment and press Enter. Type 'quit' to exit.")
    print("=" * 60)
    print("(Models will load on the first comment — please wait)\n")

    while True:
        try:
            text = input("Comment: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not text:
            continue

        result = pipeline.analyze(text)
        print_result(result)


if __name__ == "__main__":
    main()
