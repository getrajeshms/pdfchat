"""
Zero-cost PDF chat (RAG)
------------------------
- Text extraction: pypdf (free, local)
- Embeddings:      fastembed + BAAI/bge-small-en-v1.5 (free, local, CPU, no PyTorch)
- Vector search:   NumPy cosine similarity (no vector DB needed for a few thousand chunks)
- Answering:       any OpenAI-compatible endpoint -> Ollama (local, free),
                   Groq or Gemini (free tiers), or "No LLM" (search only, $0)

Run locally:  pip install -r requirements.txt  &&  streamlit run app.py

Settings (optional) can come from .streamlit/secrets.toml, Streamlit Cloud
"Secrets", or environment variables:
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    GROQ_API_KEY    = "..."
    GEMINI_API_KEY  = "..."
"""

import hashlib
import io
import json
import os
import re
import urllib.request

import numpy as np
import streamlit as st
from pypdf import PdfReader

# ----------------------------- settings ------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, small, strong for its size
EMBED_DIM = 384
CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150     # characters shared between neighbouring chunks
DEFAULT_TOP_K = 5       # passages sent to the LLM
HISTORY_TURNS = 4       # previous messages sent to the LLM (keeps tokens low)
LLM_TIMEOUT = 180       # seconds; local CPU models can be slow on the first call


def get_setting(name: str, default=None):
    """Read a setting from Streamlit secrets, then environment, then default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file at all
        pass
    return os.environ.get(name, default)


# Every provider below speaks the OpenAI chat API, so one code path serves all.
# Model names change over time: they are editable in the sidebar.
PROVIDERS = {
    "Ollama (local, free)": {
        "base_url": get_setting("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model": "llama3.2:3b",
        "key_name": None,
        "is_ollama": True,
    },
    "Groq (free tier)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "key_name": "GROQ_API_KEY",
        "is_ollama": False,
    },
    "Gemini (free tier)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",  # check Google AI Studio for current model names
        "key_name": "GEMINI_API_KEY",
        "is_ollama": False,
    },
    "No LLM (search only, $0)": None,
}

SYSTEM_PROMPT = """You answer questions using ONLY the numbered excerpts from the user's PDFs.
Rules:
- Cite sources inline like [1] or [2][3], using the excerpt numbers.
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Be concise. Use bullet points when listing several items."""


# ----------------------------- ollama check --------------------------------
@st.cache_data(ttl=15, show_spinner=False)
def ollama_status(base_url: str):
    """Return (reachable, [installed model names]) for an Ollama server."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(root + "/api/tags", timeout=3) as resp:
            data = json.load(resp)
        return True, sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return False, []


# ----------------------------- ingestion -----------------------------------
def extract_pages(data: bytes):
    """Yield (page_number, text) for each page that has extractable text."""
    reader = PdfReader(io.BytesIO(data))
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""  # scanned pages return None/empty
        except Exception:
            text = ""  # a malformed page shouldn't break the whole file
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
        try:
            for page_no, text in extract_pages(data):
                for chunk in split_text(text):
                    records.append({"file": name, "page": page_no, "text": chunk})
        except Exception as e:
            st.warning(f"Could not read {name}: {e}")

    if not records:
        return [], np.zeros((0, EMBED_DIM), dtype=np.float32)

    embedder = get_embedder()
    vecs = np.array(list(embedder.passage_embed([r["text"] for r in records])), dtype=np.float32)
    return records, normalize(vecs)


# ----------------------------- retrieval -----------------------------------
def search(query: str, records, vecs, k: int):
    q = np.array(next(iter(get_embedder().query_embed([query]))), dtype=np.float32)
    scores = vecs @ normalize(q)
    top = np.argsort(-scores)[: min(k, len(records))]
    return [{**records[i], "score": float(scores[i])} for i in top]


def build_context(hits) -> str:
    return "\n\n".join(
        f"[{n}] ({h['file']}, page {h['page']})\n{h['text']}" for n, h in enumerate(hits, start=1)
    )


# ----------------------------- generation ----------------------------------
def stream_answer(provider, model, api_key, question, hits, history):
    from openai import OpenAI

    client = OpenAI(
        base_url=provider["base_url"], api_key=api_key or "ollama", timeout=LLM_TIMEOUT
    )
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
            st.text(s["text"])  # plain text, no HTML/markdown injection


def sidebar():
    """Draw the sidebar. Returns (uploads, provider_name, provider, model, api_key, top_k, ready)."""
    with st.sidebar:
        uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

        provider_name = st.selectbox("Answer with", list(PROVIDERS))
        provider = PROVIDERS[provider_name]
        model = api_key = None
        ready = True

        if provider and provider["is_ollama"]:
            reachable, installed = ollama_status(provider["base_url"])
            if reachable:
                st.caption(f"✅ Ollama connected at {provider['base_url']}")
                if installed:
                    default = provider["model"] if provider["model"] in installed else installed[0]
                    model = st.selectbox("Model", installed, index=installed.index(default))
                else:
                    st.warning("Ollama has no models. Run `ollama pull llama3.2:3b`.")
                    ready = False
            else:
                ready = False
                st.error(
                    f"Can't reach Ollama at {provider['base_url']}.\n\n"
                    "• On your PC: open the Ollama app, then run this app with "
                    "`streamlit run app.py`.\n\n"
                    "• On Streamlit Cloud: Ollama on your PC can't be reached. "
                    "Pick Groq, Gemini or search-only instead."
                )
        elif provider:
            model = st.text_input("Model name", provider["model"])
            secret_key = get_setting(provider["key_name"])
            if secret_key:
                api_key = secret_key
                st.caption("🔑 Using API key from secrets")
            else:
                api_key = st.text_input("API key (free tier)", type="password")

        top_k = st.slider("Passages to retrieve", 2, 10, DEFAULT_TOP_K)
        if st.button("Clear chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    return uploads, provider_name, provider, model, api_key, top_k, ready


def main():
    st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")
    st.title("📄 Chat with your PDFs")
    st.caption("Free local embeddings · free LLM of your choice · answers cite page numbers")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploads, provider_name, provider, model, api_key, top_k, ready = sidebar()

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
            if m.get("error"):
                st.error(m["content"])
            else:
                st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    question = st.chat_input("Ask a question about your PDFs")
    if not question:
        return

    # Only send successful turns to the LLM as history
    history = [m for m in st.session_state.messages if not m.get("error")]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    is_error = False
    with st.chat_message("assistant"):
        hits = search(question, records, vecs, top_k)

        if provider is None:
            answer = "Search-only mode: here are the most relevant passages. Open **Sources** below."
            st.markdown(answer)
        elif not ready:
            answer = "The selected LLM isn't available (see the sidebar). Showing matching passages only."
            st.warning(answer)
        elif provider["key_name"] and not api_key:
            answer = f"Add your {provider_name.split(' (')[0]} API key in the sidebar, or switch to search-only."
            st.warning(answer)
        else:
            try:
                answer = st.write_stream(
                    stream_answer(provider, model, api_key, question, hits, history)
                )
                if not isinstance(answer, str):
                    answer = "".join(str(a) for a in answer)
                if not answer.strip():
                    answer = "The model returned an empty answer. Try again or pick another model."
                    st.warning(answer)
            except Exception as e:  # network, bad key, model not found, rate limit...
                is_error = True
                hint = (
                    " Is Ollama running, and is the model pulled?"
                    if provider["is_ollama"]
                    else " Check the API key and model name."
                )
                answer = f"LLM error: {e}.{hint}"
                st.error(answer)

        show_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": hits, "error": is_error}
    )


if __name__ == "__main__":
    main()
