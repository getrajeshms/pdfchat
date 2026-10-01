# Zero-cost PDF Chat (RAG)

Chat with your PDFs and get answers that cite file names and page numbers, without paying for anything.

## Why it costs nothing

| Stage | Tool | Cost |
|---|---|---|
| PDF text extraction | `pypdf` | Free, local |
| Embeddings | `fastembed` + `BAAI/bge-small-en-v1.5` (ONNX, CPU, no PyTorch) | Free, local |
| Vector search | NumPy cosine similarity | Free, local |
| Answer generation | Ollama (local) **or** Groq / Gemini free tiers **or** none | Free |
| Hosting | Your laptop, or Streamlit Community Cloud | Free |

The previous app re-embedded every PDF on every question through a paid API. Here, embedding is local and happens **once per set of files** (cached), so the only possible external call is the final answer, and even that can be local.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the small embedding model once.

## Pick how answers are generated (sidebar)

1. **Ollama (fully offline, $0 forever)**
   - Install from https://ollama.com, then run `ollama pull llama3.2:3b`.
   - Runs on a normal laptop with about 8 GB RAM. With more RAM, try a larger model, such as an 8B one.
2. **Groq (free tier)**: create a key at https://console.groq.com. Very fast.
3. **Gemini (free tier)**: create a key at https://aistudio.google.com.
4. **No LLM (search only)**: shows the best-matching passages with page numbers. No model at all.

Model names change over time. If a default name stops working, type the provider's current model name in the sidebar.

## Free deployment

Push `app.py` and `requirements.txt` to GitHub and deploy on Streamlit Community Cloud. Ollama runs only on your own machine, so the hosted version should use Groq, Gemini, or search-only mode.

## Tuning

- `CHUNK_SIZE` / `CHUNK_OVERLAP`: smaller chunks give more precise citations, larger chunks give more context.
- `Passages to retrieve` slider: more passages give better recall but use more tokens.
- Scanned PDFs have no text layer. Run `ocrmypdf in.pdf out.pdf` (free) first.

## When to upgrade (still free)

- **More than ~50k chunks**: replace the NumPy search with `faiss-cpu` or `chromadb`.
- **Exact terms (IDs, codes, names)**: add BM25 keyword search (`rank-bm25`) and merge it with vector results (hybrid search).
- **Better ranking**: re-rank the top 20 results with a small local cross-encoder (fastembed has rerankers).
- **Index persistence across restarts**: save `records` and `vecs` with `np.save` or JSON.
