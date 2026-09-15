import os
import re
import json
import hashlib
from collections import deque
import time
import numpy as np
import openai
import pdfplumber
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------


EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # local, free, no API key needed
MIN_OCR_CHARS = 15                             # below this, OCR = "no useful text"
USE_VISION_FALLBACK = False                    # keep False first run -- see README
MAX_TOKENS = 500                          # lower = faster generation; raise if answers feel cut off

# Tesseract path -- set TESSERACT_CMD in .env if your install location differs.
# Wrapped in a check so a missing/wrong path warns instead of crashing on import.
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"D:\JRE\tesseract.exe")
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    print(f"[warn] TESSERACT_CMD not found at '{TESSERACT_CMD}' -- OCR will fail until this "
          f"points at your real tesseract.exe (set TESSERACT_CMD in .env).")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# IMPORTANT: "..." is not a real model and will fail every API call. This default
# is a genuinely valid, fast, small instruct model -- override in .env if you want
# a different one, but don't leave CHAT_MODEL unset/placeholder.
CHAT_MODEL="openai/gpt-oss-20b"
if CHAT_MODEL.strip() in ("", "...", "TODO", "your_model_here"):
    print(f"[warn] CHAT_MODEL looks like a placeholder ('{CHAT_MODEL}') -- "
          f"set a real model ID in .env, e.g. meta/llama-3.1-8b-instruct")

VISION_MODEL = os.getenv("VISION_MODEL", "meta/llama-3.2-11b-vision-instruct")
# NOTE: verify CHAT_MODEL / VISION_MODEL are still valid names on build.nvidia.com
# before relying on them -- NVIDIA's hosted model catalog changes over time.
# Also: NVIDIA's free shared endpoint can see 600-1500ms just for the first token
# under load -- that's normal for the free tier, not something your code controls.
# Streaming (below) makes this feel fast regardless of that fixed latency.

client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)


# ---------------------------------------------------------------------------
# CACHE HELPERS
# ---------------------------------------------------------------------------

def file_hash(path):
    """Hash the PDF's contents so we can detect if the source file changed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def hash_directory(pdf_dir):
    """
    Hash based on every PDF's filename + contents in the folder, so the
    cache invalidates if any file is added, removed, or changed -- not
    just if one specific file changes.
    """
    pdf_files = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    h = hashlib.sha256()
    for fname in pdf_files:
        h.update(fname.encode())
        h.update(file_hash(os.path.join(pdf_dir, fname)).encode())
    return h.hexdigest()[:16]


def cache_is_valid(pdf_dir, cache_dir="cache"):
    """
    Returns True only if embeddings + chunks already exist AND the cached
    hash matches the current set of PDFs in pdf_dir. If any file changed,
    was added, or was removed, this returns False so pipeline.py knows to
    rebuild instead of silently using stale data.
    """
    meta_path = os.path.join(cache_dir, "meta.json")
    chunks_path = os.path.join(cache_dir, "chunks.json")
    embeddings_path = os.path.join(cache_dir, "embeddings.npz")

    if not (os.path.exists(meta_path) and os.path.exists(chunks_path) and os.path.exists(embeddings_path)):
        return False

    with open(meta_path, "r") as f:
        meta = json.load(f)

    current_hash = hash_directory(pdf_dir)
    if meta.get("dir_hash") != current_hash:
        print("WARNING: cached data doesn't match current PDFs in the folder (files changed). Rebuilding.")
        return False

    return True


# ---------------------------------------------------------------------------
# TEXT + IMAGE EXTRACTION
# ---------------------------------------------------------------------------

def clean_text(text):
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_images_from_page(page, page_num, output_dir, doc_title="doc"):
    """
    Extracts embedded images from a single page via bbox-crop.
    Tested and confirmed working on standard embedded raster images
    (e.g. matplotlib-style charts). Wrapped in try/except because some
    PDFs use unusual image formats (CMYK, indexed color) that can fail
    to crop cleanly -- a failed image is skipped, not a pipeline crash.

    doc_title is prefixed into the filename so images from different
    documents never collide (e.g. two docs both having a "page1_img1").
    """
    os.makedirs(output_dir, exist_ok=True)  # defensive: don't rely on caller to have made this
    saved = []
    for img_idx, img in enumerate(page.images, start=1):
        try:
            bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
            cropped = page.within_bbox(bbox).to_image(resolution=200)
            filename = f"{doc_title}_page{page_num}_img{img_idx}.png"
            filepath = os.path.join(output_dir, filename)
            cropped.save(filepath)
            saved.append({"doc_title": doc_title, "page_num": page_num, "filepath": filepath})
        except Exception as e:
            print(f"  [warn] skipped one image on page {page_num} of {doc_title}: {e}")
    return saved


def run_ocr(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img).strip()


def ocr_has_useful_text(text, min_chars=MIN_OCR_CHARS):
    return len(text.strip()) >= min_chars


def vision_fallback_caption(image_path):
    """
    Only called when OCR found little/no text AND USE_VISION_FALLBACK=True.
    Wrapped in try/except so one bad API call doesn't kill a 150-page run --
    on failure, returns an empty string and the image is simply treated as
    having no extractable content.
    """
    import base64
    try:
        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what this chart or figure shows, "
                                              "including any key numbers, trends, or labeled "
                                              "categories. Be concise (2-3 sentences)."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            }],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [warn] vision fallback failed on {image_path}: {e}")
        return ""


# ---------------------------------------------------------------------------
# CHUNKING (paragraph-based, fixed-size overlap fallback, tables excluded)
# ---------------------------------------------------------------------------



def fixed_size_chunks(text, chunk_size_words=250, overlap_words=45):
    words = text.split()
    if len(words) <= chunk_size_words:
        return [text]
    chunks, step = [], chunk_size_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def chunk_text(text, source_type, page_num, doc_title, chunk_id_start=0,
                max_paragraph_words=300, chunk_size_words=250, overlap_words=45):
    """
    Chunks one page's merged text (narrative + image-derived), dropping
    table-like chunks entirely. source_type is "text" or "image" so you
    can later see which kind of content actually gets retrieved.
    doc_title identifies which of your PDFs this chunk came from --
    essential once page numbers repeat across documents.
    """
    chunks = []
    chunk_id = chunk_id_start

    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        word_count = len(para.split())
        sub_chunks = [para] if word_count <= max_paragraph_words else \
            fixed_size_chunks(para, chunk_size_words, overlap_words)

        for sub in sub_chunks:
            chunk_id += 1
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:05d}",
                "doc_title": doc_title,
                "page_num": page_num,
                "source_type": source_type,
                "text": sub,
            })

    return chunks, chunk_id


# ---------------------------------------------------------------------------
# EMBEDDING + RETRIEVAL
# ---------------------------------------------------------------------------

_embed_model = None

def get_embed_model():
    """
    Loaded once, reused for every call -- this is already cached correctly,
    but chatbot.py now calls this explicitly at startup (see chatbot.py)
    so the one-time load delay happens during "Loading..." instead of
    silently making your first question look unusually slow.
    """
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def embed_chunks(chunks):
    model = get_embed_model()
    texts = [c["text"] for c in chunks]
    return np.array(model.encode(texts, batch_size=32, show_progress_bar=True,
                                  normalize_embeddings=True))


def embed_query(query):
    model = get_embed_model()
    instruction = "Represent this sentence for searching relevant passages: "
    return model.encode([instruction + query], normalize_embeddings=True)[0]


def retrieve(query, embeddings, chunks, top_k=5):
    query_vec = embed_query(query)
    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        c = chunks[idx]
        results.append({
            "score": float(scores[idx]),
            "chunk_id": c["chunk_id"],
            "doc_title": c["doc_title"],
            "page_num": c["page_num"],
            "source_type": c["source_type"],
            "text": c["text"],
        })
    return results


# ---------------------------------------------------------------------------
# CACHE LOAD/SAVE
# ---------------------------------------------------------------------------

def save_cache(pdf_dir, chunks, embeddings, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(cache_dir, "embeddings.npz"), embeddings=embeddings)
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({"dir_hash": hash_directory(pdf_dir), "n_chunks": len(chunks)}, f, indent=2)
    print(f"Cached {len(chunks)} chunks + embeddings to {cache_dir}/")


def load_cache(cache_dir="cache"):
    with open(os.path.join(cache_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    data = np.load(os.path.join(cache_dir, "embeddings.npz"))
    embeddings = data["embeddings"]
    return chunks, embeddings


# ---------------------------------------------------------------------------
# CONVERSATION MEMORY
#   - last MAX_EXACT_HISTORY exchanges kept verbatim
#   - anything older folded into a lightweight text summary (no LLM call,
#     so this never adds latency)
#   - a fast keyword check decides whether the CURRENT question actually
#     needs any of this -- most questions are standalone and skip it
#     entirely, which keeps the common case fast
#   - separately, the full session is appended to a small saved log on
#     disk (last MAX_SAVED_HISTORY exchanges), for your own record
# ---------------------------------------------------------------------------

MAX_EXACT_HISTORY = 5

MAX_SUMMARY_CHARS = 500
HISTORY_FILE = "chat_history.json"

# Cheap, fast, no-LLM-call heuristic. Not perfect -- a known, documented
# limitation rather than something worth an extra API call to classify.
_REFERENTIAL_PATTERNS = [
    r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b", r"\bit\b",
    r"\bprevious(ly)?\b", r"\bearlier\b", r"\babove\b", r"\bagain\b",
    r"\bfurther\b", r"\bmore on\b", r"\bcontinue\b", r"\bsame\b",
    r"\bthe (bias|concept|topic|thing|example|case)\b",
]
_REFERENTIAL_RE = re.compile("|".join(_REFERENTIAL_PATTERNS), re.IGNORECASE)


# def needs_conversation_history(query):
#     """True if the question looks like it depends on earlier conversation
#     (e.g. 'what is this bias?'). Otherwise the fast, stateless RAG path runs."""
#     return bool(_REFERENTIAL_RE.search(query))


class ConversationMemory:
    """Lives for one chatbot.py session. exact = last 5 exchanges verbatim.
    summary = a running, lightweight text digest of anything older."""

    def __init__(self):
        self.exact = deque(maxlen=MAX_EXACT_HISTORY)
        self.summary = ""

    def add_exchange(self, user_msg, assistant_msg):
        if len(self.exact) == self.exact.maxlen:
            self._fold_into_summary(self.exact[0])
        self.exact.append({"user": user_msg, "assistant": assistant_msg})

    def _fold_into_summary(self, exchange):
        snippet = f"Q: {exchange['user'][:100]} | A: {exchange['assistant'][:100]}"
        self.summary = (self.summary + " " + snippet).strip()
        if len(self.summary) > MAX_SUMMARY_CHARS:
            self.summary = self.summary[-MAX_SUMMARY_CHARS:]

    def build_context_block(self):
        """Returns a text block for the prompt, or None if there's no history yet."""
        if not self.exact and not self.summary:
            return None
        parts = []
        if self.summary:
            parts.append(f"Summary of earlier conversation: {self.summary}")
        if self.exact:
            recent = "\n\n".join(f"User: {e['user']}\nAssistant: {e['assistant']}"
                                  for e in self.exact)
            parts.append(f"Most recent {len(self.exact)} exchange(s):\n{recent}")
        return "\n\n".join(parts)


def load_saved_history(path=HISTORY_FILE):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

#max last 5 conversation exchanges are saved to disk for your own record, but the LLM only sees the last 5 exchanges in memory (see ConversationMemory above).
def save_history_entry(user_msg, assistant_msg, path=HISTORY_FILE):
    """Appends to a persisted log capped at the last `max_saved` exchanges."""
    history = load_saved_history(path)
    history.append({"user": user_msg, "assistant": assistant_msg})
    history = history[-5:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# GENERATION (RAG-augmented + no-RAG baseline)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a behavioral finance study assistant. Use the retrieved context "
    "as your PRIMARY EVIDENCE, but you may use reliable general knowledge when "
    "the context does not fully answer the question.\n\n"

    "if the question is not related to behavioral finance, politely decline to answer and say you are a behavioral finance study assistant.\n\n"
    " EXCEPTION TO ABOVE IS WHEN IN THE CONTEXT, A BEHAVIORAL FINANCE CONCEPT IS MENTIONED, BUT THE QUESTION IS ABOUT A SPECIFIC APPLICATION OR SCENARIO. IN THAT CASE, REASON FROM THE CONCEPT TO ANSWER THE QUESTION.\n\n"
    "ONE MORE EXCEPTION IS WHEN IN THE QUESTION THERE IS MENTION ABOUT MORGAN STANLEY, IN THAT CASE, REASON FROM THE CONTEXT TO ANSWER THE QUESTION.\n\n"

    "Answering rules:\n"
    "- If the retrieved context directly answers the question, prioritize and "
    "accurately reflect that information.\n"
    "- For application, scenario, or case-study questions, reason from the "
    "concepts in the context to identify the relevant behavioral finance "
    "concept or bias, even when the exact scenario is not stated in the context.\n"
    "- You may explain a concept using simple examples or reasoning when useful, "
    "but do not present information as if it came from the retrieved context "
    "unless it actually did.\n"
    "- If the question refers to 'this', 'that', 'the above', or a previously "
    "discussed bias, use the Conversation history only to resolve the reference. "
    "Do not treat conversation history as evidence.\n"
    "- If the retrieved context is insufficient, use reliable general knowledge "
    "when appropriate. If you are genuinely uncertain, say so rather than guessing.\n"
    "- Never fabricate facts, examples, source claims, citations, or evidence.\n"
    "- Distinguish clearly between information supported by the context and "
    "your own reasoning when necessary.\n"
    "- Answer the question directly and avoid unnecessary background.\n"
    "- Be concise, clear, and suitable for a behavioral finance student."
)


def build_rag_prompt(query, retrieved_chunks, history_context=None):
    context_block = "\n\n".join(
        f"[{c['doc_title']}, page {c['page_num']}, source: {c['source_type']}]\n{c['text']}"
        for c in retrieved_chunks
    )
    history_block = f"\n\nConversation history (for resolving references only):\n{history_context}" \
        if history_context else ""
    return f"Context:\n{context_block}{history_block}\n\nQuestion: {query}\n\nAnswer using the context above."



def stream_rag_answer(query, embeddings, chunks, top_k=5, history_context=None):
    """
    Streaming version for the interactive chatbot.
    Returns retrieved chunks and a generator that yields the answer
    as text arrives from the API.
    """
    t1 = time.perf_counter()

    retrieved = retrieve(query, embeddings, chunks, top_k=top_k)

    t2 = time.perf_counter()

    prompt = build_rag_prompt(query, retrieved, history_context=history_context)

    t3 = time.perf_counter()

    print(f"[Retrieval time: {t2-t1:.3f}s]")
    print(f"[Prompt building time: {t3-t2:.3f}s]")

    def token_generator():
        t4 = time.perf_counter()
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        t5 = time.perf_counter()

        print(f"[API request setup: {t5-t4:.3f}s]")

        for event in stream:
            # Some streaming events may contain no choices.
            if not getattr(event, "choices", None):
                continue

            choice = event.choices[0]
            delta = getattr(choice, "delta", None)

            if delta is None:
                continue

            content = getattr(delta, "content", None)

            if content:
                yield content

    # IMPORTANT: this line must be present and outside token_generator()
    return retrieved, token_generator()


