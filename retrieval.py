"""
Better retrieval for PDF chat
-----------------------------
1. Query rewriting:   the LLM turns the user's question (plus chat history) into a
                      standalone search query and extra keywords
                      ("objective of this paper" -> "study objectives aims hypothesis").
2. Hybrid search:     BM25 keyword search (with light stemming, so "objective" matches
                      "objectives") + vector search, merged with Reciprocal Rank Fusion.
3. Document-level:    whole-paper questions ("what is this paper about / its objective")
                      also get the first page (title + abstract) of each file.

No new dependencies: BM25 and the stemmer are implemented with NumPy and plain Python.
Everything falls back gracefully: if the LLM rewrite fails, the original question is used.
"""

import json
import math
import re
from collections import Counter, defaultdict

import numpy as np

# ----------------------------- tokenizing ----------------------------------
STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into
    is it its itself just me more most my no nor not now of off on once only or other
    our out over own same she should so some such than that the their them then there
    these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your paper article document pdf tell
    please give explain describe""".split()
)


def stem(word: str) -> str:
    """Tiny suffix-stripping stemmer: objectives/objective -> objectiv, studies/study -> study."""
    if len(word) > 5 and word.endswith("eses"):  # hypotheses -> hypothesis
        word = word[:-2] + "is"
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str):
    # PDFs often use ligatures ("ﬁ") and hyphenated line breaks ("scal-\nable")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"-\s*\n\s*", "", text).lower()
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text) if w not in STOPWORDS and len(w) > 1]


# ----------------------------- BM25 ----------------------------------------
def build_bm25(texts, k1: float = 1.5, b: float = 0.75):
    """Return a small picklable BM25 index: postings term -> (doc_ids, tfs)."""
    docs = [tokenize(t) for t in texts]
    lengths = np.array([len(d) for d in docs], dtype=np.float32)
    postings = defaultdict(lambda: ([], []))
    for doc_id, toks in enumerate(docs):
        for term, tf in Counter(toks).items():
            postings[term][0].append(doc_id)
            postings[term][1].append(tf)
    n = len(docs)
    index = {}
    for term, (ids, tfs) in postings.items():
        df = len(ids)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        index[term] = (np.array(ids, dtype=np.int32), np.array(tfs, dtype=np.float32), idf)
    return {"index": index, "lengths": lengths, "avgdl": float(lengths.mean() or 1.0), "k1": k1, "b": b}


def bm25_scores(bm25, query: str) -> np.ndarray:
    scores = np.zeros(len(bm25["lengths"]), dtype=np.float32)
    k1, b, avgdl = bm25["k1"], bm25["b"], bm25["avgdl"]
    for term in set(tokenize(query)):
        if term not in bm25["index"]:
            continue
        ids, tfs, idf = bm25["index"][term]
        norm = k1 * (1 - b + b * bm25["lengths"][ids] / avgdl)
        scores[ids] += idf * tfs * (k1 + 1) / (tfs + norm)
    return scores


# ----------------------------- fusion --------------------------------------
def rrf(rankings, k: int = 60):
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[int(doc_id)] += 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)


# ----------------------------- query rewriting -----------------------------
REWRITE_PROMPT = """You turn a user's question about their PDF documents into search queries.
Use the conversation so far to resolve words like "it", "they" or "that study".

Return ONLY a JSON object, no other text:
{"query": "<standalone question in the wording a research paper would use>",
 "keywords": "<6-12 extra search words and synonyms, space separated>",
 "scope": "<'document' if the question is about the whole paper (its aim, objective, summary, main findings, conclusion), else 'specific'>"}

Example: "what is the primary objective of this paper?" ->
{"query": "What were the primary objectives and hypotheses of this study?",
 "keywords": "objectives aims purpose hypothesis study tested goal",
 "scope": "document"}"""

DOC_LEVEL = re.compile(
    r"\b(this|the) (paper|study|article|document|report|pdf)\b|\bsummar|\bmain (aim|objective|finding|point|idea)"
    r"|\b(objective|aim|purpose|conclusion|key finding)s?\b",
    re.I,
)


def parse_rewrite(text: str, question: str):
    """Pull the JSON object out of the model's reply; fall back safely if it is malformed."""
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    query = str(data.get("query") or question).strip()
    keywords = str(data.get("keywords") or "").strip()
    scope = str(data.get("scope") or "").strip().lower()
    if scope not in ("document", "specific"):
        scope = "document" if DOC_LEVEL.search(question) else "specific"
    return {"query": query, "keywords": keywords, "scope": scope}


def rewrite_query(client, model, question, history, turns: int = 4):
    """Ask the LLM for a standalone query + keywords. Never raises: falls back to the question."""
    fallback = {
        "query": question,
        "keywords": "",
        "scope": "document" if DOC_LEVEL.search(question) else "specific",
        "rewritten": False,
    }
    if client is None:
        return fallback
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-turns:])
    user = (f"Conversation so far:\n{convo}\n\n" if convo else "") + f"Question: {question}"
    try:
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=200,
        )
        result = parse_rewrite(reply.choices[0].message.content, question)
        result["rewritten"] = True
        return result
    except Exception:  # model down, rate limit, bad key: retrieval still works
        return fallback


# ----------------------------- hybrid search -------------------------------
def hybrid_search(question, plan, records, vecs, bm25, embed_query, k=5, candidates=30, doc_pages=2):
    """
    plan:        output of rewrite_query()
    embed_query: function str -> normalized 1-D vector
    Returns up to k hits plus, for document-level questions, first-page chunks of each file.
    """
    if not records:
        return []
    queries = list(dict.fromkeys([question, plan["query"]]))  # dedupe, keep order
    rankings = []
    vec_scores = None
    for q in queries:
        s = vecs @ embed_query(q)
        vec_scores = s if vec_scores is None else np.maximum(vec_scores, s)
        rankings.append(np.argsort(-s)[:candidates])
    keyword_text = " ".join(queries + [plan.get("keywords", "")])
    kw = bm25_scores(bm25, keyword_text)
    rankings.append([i for i in np.argsort(-kw)[:candidates] if kw[i] > 0])

    # Each retriever's single best hit is always kept (plain RRF can drop a chunk that is
    # #1 for keywords but mediocre semantically); the remaining slots are filled by RRF.
    chosen = []
    for ranking in rankings:
        if len(ranking) and int(ranking[0]) not in chosen:
            chosen.append(int(ranking[0]))
    for doc_id in rrf(rankings):
        if len(chosen) >= k:
            break
        if doc_id not in chosen:
            chosen.append(doc_id)

    if plan.get("scope") == "document":
        # Title + abstract usually live in the first chunks of page 1 of each file.
        seen_files = defaultdict(int)
        for i, r in enumerate(records):
            if r["page"] == 1 and seen_files[r["file"]] < doc_pages and i not in chosen:
                chosen.append(i)
                seen_files[r["file"]] += 1

    return [
        {**records[i], "score": float(vec_scores[i]), "keyword": float(kw[i])}
        for i in chosen
    ]
