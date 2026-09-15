"""
chatbot.py

Ask questions against your cached, embedded corpus. Loads whatever
pipeline.py already built -- never re-embeds anything.

Speed notes:
- Answers stream token-by-token instead of waiting for the full response.
- The embedding model is warmed up once at startup (see below) so the
  first question isn't misleadingly slower than the rest.
- The last 5 conversation exchanges are always sent to the LLM,
  allowing follow-up questions to use recent conversation context.

Usage:
    python chatbot.py
    (then type questions at the prompt; type 'exit' to quit)
"""

import rag_utils as ru
import time
TOP_K = 5  



def ask(query, embeddings, chunks, memory):

    history_context = memory.build_context_block()
    start_time = time.perf_counter()

    retrieved, token_stream = ru.stream_rag_answer(
        query, embeddings, chunks, top_k=TOP_K, history_context=history_context
    )

    print("\nAnswer:\n", end="", flush=True)
    full_answer = ""
    for token in token_stream:
        print(token, end="", flush=True)
        full_answer += token
    print("\n")

    print("Sources used:")
    for r in retrieved:
        print(f"  [score {r['score']:.3f}] {r['doc_title']}, page {r['page_num']} ({r['source_type']})")
        print(f"    {r['text'][:90]}...")
    print()

    memory.add_exchange(query, full_answer)
    ru.save_history_entry(query, full_answer)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(total_time)


if __name__ == "__main__":
    print("Loading cached embeddings...")
    chunks, embeddings = ru.load_cache()
    ru.get_embed_model()  # load once now, not silently on your first question
    print("Ready.\n")

    memory = ru.ConversationMemory()
    
    while True:
        query = input("Ask a question (or 'exit' or 'quit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        ask(query, embeddings, chunks, memory)
