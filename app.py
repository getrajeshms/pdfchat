"""
Zero-cost PDF chat (RAG)
------------------------
- Text extraction: pypdf (free, local)
- Embeddings:      fastembed + BAAI/bge-small-en-v1.5 (free, local, CPU, no PyTorch)
- Vector search:   NumPy cosine similarity (no vector DB needed for a few thousand chunks)
- Answering:       any OpenAI-compatible endpoint -> Ollama (local, free),
                   Groq or Gemini (free tiers), or "No LLM" (search only, $0)

Run:  pip install -r requirements.txt  &&  streamlit run app.py
"""

import hashlib
import io
import re

import numpy as np
import streamlit as st
from pypdf import PdfReader

# ----------------------------- settings ------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, small, strong for its size
CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150     # characters shared between neighbouring chunks
DEFAULT_TOP_K = 5       # passages sent to the LLM
HISTORY_TURNS = 4       # previous messages sent to the LLM (keeps tokens low)

# Every provider below speaks the OpenAI chat API, so one code path serves all.
# Model names change over time: they are editable in the sidebar.
PROVIDERS = {
    "Ollama (local, free)": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2:3b",
        "needs_key": False,
    },
    "Groq (free tier)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "needs_key": True,
    },
    "Gemini (free tier)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "needs_key": True,
    },
    "No LLM (search only, $0)": None,
}

SYSTEM_PROMPT = """You answer questions using ONLY the numbered excerpts from the user's PDFs.
Rules:
- Cite sources inline like [1] or [2][3], using the excerpt numbers.
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Be concise. Use bullet points when listing several items."""


# ----------------------------- ingestion -----------------------------------
def extract_pages(data: bytes):
    """Yield (page_number, text) for each page that has extractable text."""
    reader = PdfReader(io.BytesIO(data))
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""  # scanned pages return None/empty
        if text.strip():
            yield page_no, text


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring paragraph/sentence breaks."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:  # only break early if it isn't too early
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


@st.cache_resource(show_spinner="Loading embedding model (one-time small download)...")
def get_embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL)


def normalize(vecs: np.ndarray) -> np.ndarray:
    return vecs / (np.linalg.norm(vecs, axis=-1, keepdims=True) + 1e-12)


@st.cache_data(show_spinner=False, max_entries=5)
def build_index(key: str, _files):
    """Build (records, vectors) once per unique set of files. `key` is a content hash."""
    records = []
    for name, data in _files:
        for page_no, text in extract_pages(data):
            for chunk in split_text(text):
                records.append({"file": name, "page": page_no, "text": chunk})

    if not records:
        return [], np.zeros((0, 384), dtype=np.float32)

    embedder = get_embedder()
    vecs = np.array(list(embedder.passage_embed([r["text"] for r in records])), dtype=np.float32)
    return records, normalize(vecs)


# ----------------------------- retrieval -----------------------------------
def search(query: str, records, vecs, k: int):
    q = np.array(next(iter(get_embedder().query_embed([query]))), dtype=np.float32)
    scores = vecs @ normalize(q)
    top = np.argsort(-scores)[:k]
    return [{**records[i], "score": float(scores[i])} for i in top]


def build_context(hits) -> str:
    return "\n\n".join(
        f"[{n}] ({h['file']}, page {h['page']})\n{h['text']}" for n, h in enumerate(hits, start=1)
    )


# ----------------------------- generation ----------------------------------
def stream_answer(provider, model, api_key, question, hits, history):
    from openai import OpenAI

    client = OpenAI(base_url=provider["base_url"], api_key=api_key or "ollama")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-HISTORY_TURNS:]]
    messages.append(
        {"role": "user", "content": f"Excerpts:\n\n{build_context(hits)}\n\nQuestion: {question}"}
    )

    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2, stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ----------------------------- UI ------------------------------------------
def show_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for n, s in enumerate(sources, start=1):
            st.markdown(f"**[{n}] {s['file']}, page {s['page']}** · relevance {s['score']:.2f}")
            st.text(s["text"])  # st.text: shown as plain text, no HTML/markdown injection


def main():
    st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")
    st.title("📄 Chat with your PDFs")
    st.caption("Free local embeddings · free LLM of your choice · answers cite page numbers")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

        provider_name = st.selectbox("Answer with", list(PROVIDERS))
        provider = PROVIDERS[provider_name]
        model = api_key = None
        if provider:
            model = st.text_input("Model name", provider["model"])
            if provider["needs_key"]:
                api_key = st.text_input("API key (free tier)", type="password")

        top_k = st.slider("Passages to retrieve", 2, 10, DEFAULT_TOP_K)
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if not uploads:
        st.info("Upload one or more PDFs in the sidebar to start.")
        return

    files = tuple((f.name, f.getvalue()) for f in uploads)
    key = hashlib.sha256(
        b"".join(n.encode() + hashlib.sha256(d).digest() for n, d in files)
    ).hexdigest()

    with st.spinner("Indexing PDFs (only happens once per set of files)..."):
        records, vecs = build_index(key, files)

    if not records:
        st.error("No text found. Scanned/image PDFs need OCR first (e.g. `ocrmypdf in.pdf out.pdf`).")
        return
    st.sidebar.success(f"Indexed {len(files)} file(s) · {len(records)} chunks")

    # Replay the conversation
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    question = st.chat_input("Ask a question about your PDFs")
    if not question:
        return

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = search(question, records, vecs, top_k)

        if provider is None:
            answer = "Search-only mode: here are the most relevant passages. Open **Sources** below."
            st.markdown(answer)
        elif provider["needs_key"] and not api_key:
            answer = f"Add your {provider_name.split(' (')[0]} API key in the sidebar, or switch to Ollama / search-only."
            st.warning(answer)
        else:
            try:
                answer = st.write_stream(
                    stream_answer(provider, model, api_key, question, hits, history)
                )
            except Exception as e:  # network, bad key, model not pulled, rate limit...
                hint = " Is Ollama running, and did you `ollama pull` the model?" if not provider["needs_key"] else ""
                answer = f"LLM error: {e}.{hint}"
                st.error(answer)

        show_sources(hits)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})


if __name__ == "__main__":
    main()
