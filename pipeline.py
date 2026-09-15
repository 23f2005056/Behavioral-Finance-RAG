"""
pipeline.py

Run this ONCE (or whenever your PDFs change) to build the cache: extract
text + images from every PDF in PDF_DIR, OCR the images, chunk everything,
embed, and save to cache/. chatbot.py and evaluate.py both just read from
that cache afterward -- they never re-embed anything.

Usage:
    python pipeline.py

Before running on the full 50-page document, set PAGE_LIMIT below to a
small number (e.g. 10) and confirm the whole thing runs end to end first.
That's your "smallest possible end-to-end version" for hour 1. The other
9 files are already small (<=10 pages), so PAGE_LIMIT mostly matters for
your one 50-page document.
"""

import os
import json
import pdfplumber

import rag_utils as ru

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PDF_DIR = "data"       # folder containing all of your PDFs
PAGE_LIMIT = 10         # <-- start small! Set to None to process full documents.


def process_pdf(pdf_path, doc_title, page_limit, chunk_id_counter):
    """Processes a single PDF, returns (chunks, image_log, updated chunk_id_counter)."""
    all_chunks = []
    image_extraction_log = []

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:page_limit] if page_limit else pdf.pages
        total = len(pages)

        for i, page in enumerate(pages, start=1):
            page_num = i
            print(f"  [{doc_title}] page {page_num}/{total}...")

            raw_text = page.extract_text() or ""
            page_text = ru.clean_text(raw_text)

            images = ru.extract_images_from_page(page, page_num, "cache/images", doc_title=doc_title)
            image_texts = []

            for img in images:
                ocr_text = ru.run_ocr(img["filepath"])
                useful = ru.ocr_has_useful_text(ocr_text)

                final_text = ocr_text
                method = "ocr"

                if not useful and ru.USE_VISION_FALLBACK:
                    caption = ru.vision_fallback_caption(img["filepath"])
                    if caption:
                        final_text = caption
                        method = "vision"

                image_extraction_log.append({
                    "doc_title": doc_title,
                    "page_num": page_num,
                    "filepath": img["filepath"],
                    "ocr_text": ocr_text,
                    "method_used": method,
                    "final_text": final_text,
                })

                if final_text:
                    image_texts.append(final_text)

            if page_text:
                text_chunks, chunk_id_counter = ru.chunk_text(
                    page_text, source_type="text", page_num=page_num, doc_title=doc_title,
                    chunk_id_start=chunk_id_counter
                )
                all_chunks.extend(text_chunks)

            if image_texts:
                image_text_combined = "\n\n".join(image_texts)
                img_chunks, chunk_id_counter = ru.chunk_text(
                    image_text_combined, source_type="image", page_num=page_num, doc_title=doc_title,
                    chunk_id_start=chunk_id_counter
                )
                all_chunks.extend(img_chunks)

    return all_chunks, image_extraction_log, chunk_id_counter


def process_all_pdfs(pdf_dir, page_limit=None):
    """Loops over every PDF in pdf_dir, tagging each chunk with its doc_title."""
    pdf_files = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    print(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}: {pdf_files}")

    all_chunks = []
    all_image_logs = []
    chunk_id_counter = 0

    for filename in pdf_files:
        doc_title = os.path.splitext(filename)[0]
        pdf_path = os.path.join(pdf_dir, filename)
        print(f"\nProcessing {filename}...")

        chunks, image_log, chunk_id_counter = process_pdf(pdf_path, doc_title, page_limit, chunk_id_counter)
        all_chunks.extend(chunks)
        all_image_logs.extend(image_log)

    with open("cache/image_extractions.json", "w", encoding="utf-8") as f:
        json.dump(all_image_logs, f, indent=2, ensure_ascii=False)

    n_ocr_useful = sum(1 for r in all_image_logs if r["method_used"] == "ocr" and r["final_text"])
    n_vision = sum(1 for r in all_image_logs if r["method_used"] == "vision")
    n_empty = sum(1 for r in all_image_logs if not r["final_text"])
    print(f"\nImages processed across all documents: {len(all_image_logs)} total | "
          f"{n_ocr_useful} via OCR | {n_vision} via vision fallback | {n_empty} yielded nothing")

    return all_chunks


if __name__ == "__main__":
    if ru.cache_is_valid(PDF_DIR):
        print("Cache already exists and matches the PDFs in data/ -- nothing to do.")
        print("Delete the cache/ folder if you want to force a rebuild.")
    else:
        print(f"Building cache from PDFs in {PDF_DIR}/ (page_limit={PAGE_LIMIT})...")
        chunks = process_all_pdfs(PDF_DIR, page_limit=PAGE_LIMIT)

        n_text = sum(1 for c in chunks if c["source_type"] == "text")
        n_image = sum(1 for c in chunks if c["source_type"] == "image")
        n_docs = len(set(c["doc_title"] for c in chunks))
        print(f"\n{len(chunks)} total chunks from {n_docs} document(s) "
              f"({n_text} from text, {n_image} from images)")

        print("Embedding chunks...")
        embeddings = ru.embed_chunks(chunks)

        ru.save_cache(PDF_DIR, chunks, embeddings)
        print("\nDone. Run chatbot.py to ask questions, or evaluate.py to run your test set.")
