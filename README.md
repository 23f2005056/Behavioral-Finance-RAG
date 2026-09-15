# Behavioral Finance RAG Bot

A retrieval-augmented Q&A system over behavioral finance PDFs (including
charts/figures), with streaming answers and lightweight conversation memory.

## 1. What does this do?

Answers questions about behavioral finance concepts by retrieving the most
relevant passages (including text pulled from charts via OCR) and reasoning
from them -- including application/scenario questions where the answer
requires identifying a concept, not just repeating a definition. It also
remembers your last few questions so follow-ups like "what is this bias?"
resolve correctly, without slowing down standalone questions.

## 2. How do I run it?

```bash
pip install -r requirements.txt
```

**Install Tesseract OCR itself (a system binary, not just a Python package):**
- Windows: download from the Tesseract GitHub releases page, then set
  `TESSERACT_CMD=C:\path\to\tesseract.exe` in your `.env` (don't hardcode
  it in the code -- that's what broke portability last time)
- Mac: `brew install tesseract`
- Linux/Colab: `sudo apt-get install tesseract-ocr` / `!apt-get install -y tesseract-ocr`

**Then:**

```bash
cp .env.example .env
# fill in NVIDIA_API_KEY and TESSERACT_CMD in .env

mkdir -p data
# put all your PDFs in data/

python pipeline.py          # step 1: build the cache (run once)
python chatbot.py           # step 2: ask questions -- answers stream live


```

```

## What changed for speed, and why

- **Streaming answers.** `chatbot.py` now shows tokens as they arrive
  instead of waiting for the full response -- this is the single biggest
  perceived-speed fix, since NVIDIA's free shared endpoint can genuinely
  take 600-1500ms just for the first token under load, which no amount of
  local code optimization removes. Streaming makes that wait feel like
  "it's typing" instead of "it's frozen."
- **Fixed `CHAT_MODEL` placeholder.** Your version had `CHAT_MODEL` default
  to the literal string `"..."`, which isn't a real model and would fail
  every call if `.env` didn't override it. Now defaults to
  `meta/llama-3.1-8b-instruct`, a genuinely fast, small instruct model --
  verify it's still current on build.nvidia.com before relying on it.
- **Lower `max_tokens` (400, was 512).** Shorter generations finish faster;
  raise it back if answers feel cut off.
- **Embedding model warm-up at startup.** `chatbot.py` now loads the local
  embedding model right after "Loading cached embeddings..." instead of
  lazily on your first question, so the one-time load delay doesn't look
  like your first question is unusually slow.
- **History is opt-in per question**, not sent on every turn -- most
  questions are standalone, so most requests stay on the fast, no-history
  path. Only questions that look referential ("this bias", "that concept",
  "continue") pay the (small) extra prompt-size cost.



