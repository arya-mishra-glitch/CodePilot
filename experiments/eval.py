"""
Layer 6 — Evaluation (Recall@K and MRR vs. Naive Chunking Baseline)
====================================================================
Measures whether your AST-aware retrieval system finds the right code
more often than a naive text-chunking approach.

Architecture (v2)
-----------------
The original eval had three compounding problems:

  1. Query generation was unfair: the LLM wrote a question for symbol X,
     but the question often had multiple valid answers in the codebase.
     Your system retrieved equally correct symbol Y and got marked a miss.

  2. The judge checked symbol identity, not answer quality.  A correct
     symbol at rank 2 (after call-graph expansion shuffled the order)
     was invisibly penalised.

  3. source_symbol was stored but never shown to the judge, so the judge
     had no anchor for what "correct" means on ambiguous questions.

v2 fixes all three with a two-phase design:

  Phase 1 — generate_queries (unchanged interface, extended output)
      For each sampled symbol the LLM writes:
        • question       — natural developer question (no function name)
        • reference_answer — one-sentence answer derived from that symbol's code

  Phase 2 — judging (new: context-quality, not symbol-identity)
      At eval time, ALL top-K retrieved symbols are concatenated into a
      single context block and passed to the judge with the question and
      reference_answer.  The judge answers ONE question:
        "Does this context contain enough information to arrive at
         the reference answer?"
      This means:
        • Call-graph expansion gets credit when combined context answers
          the question even if no single symbol does.
        • Ambiguous questions are anchored to what the source symbol
          actually does, not whatever the judge guesses "correct" means.
        • Only one LLM call per query (K times cheaper than before).

What this file does
-------------------
  1. naive_baseline     — fixed N-line window chunking, pure FAISS.
  2. generate_queries   — auto-generates (question, reference_answer,
                          source_symbol) triples from your index.
  3. evaluate           — runs both systems, measures Recall@K and MRR.
  4. plot               — side-by-side bar chart.
  5. CLI                — run everything from the terminal.

Relevance judging modes
-----------------------
  hint-only (default)
      A symbol is a hit if any hint string appears as a substring in its
      name, display_name, or file path.  Fast and free.  Use with
      hand-written query sets that have "relevant" hint lists.

  LLM-as-judge (--judge-provider gemini|groq|auto)   ← recommended
      Passes the full retrieved context + reference_answer to the LLM.
      Required when using --generate-queries (those queries have no hints).
      Falls back to hint matching as a free fast-path when hints exist.

Usage
-----
    # Recommended: auto-generate queries, judge with LLM
    python eval.py ./repo_index --repo /path/to/repo \\
        --generate-queries --judge-provider auto

    # Save generated queries for reuse
    python eval.py ./repo_index --repo /path/to/repo \\
        --generate-queries --save-queries generated.json --judge-provider auto

    # Reuse saved queries
    python eval.py ./repo_index --repo /path/to/repo \\
        --queries generated.json --judge-provider auto

    # Hint-only mode with a hand-written query file
    python eval.py ./repo_index --repo /path/to/repo --queries my_queries.json

    # Show built-in sample queries
    python eval.py --show-queries

    # Adjust K (default 5), save JSON results, skip plot
    python eval.py ./repo_index --repo /path/to/repo \\
        --generate-queries --judge-provider auto --k 10 \\
        --save-json results.json --no-plot

Query JSON format
-----------------
    [
      {
        "question":         "where is authentication handled?",
        "reference_answer": "JWT tokens are verified in auth/middleware.js",
        "source_symbol":    "authMiddleware",   // debug only, not used by judge
        "relevant":         ["auth", "login"]   // optional; used by hint-only mode
      },
      ...
    ]

    "reference_answer" is used by LLM-as-judge.
    "relevant"         is used by hint-only mode.
    Either can be omitted depending on which judging mode you use.

Metrics
-------
  Recall@K   — fraction of queries where the retrieved context contains
                a relevant answer in the top-K results.
  MRR        — Mean Reciprocal Rank: 1/rank of first relevant symbol,
                averaged across queries.

Dependencies
------------
    pip install sentence-transformers faiss-cpu matplotlib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ── Layer imports ──────────────────────────────────────────────────────────────
try:
    import index_manager
    import embedder as emb
    from query_engine import _expand_with_call_graph, _PROVIDERS
except ImportError as e:
    print(
        f"[eval] ImportError: {e}\n"
        "Make sure index_manager.py, embedder.py, and query_engine.py are on sys.path.",
        file=sys.stderr,
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in sample query set  (hand-written, hint-only mode)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    {
        "question": "how is user authentication handled?",
        "relevant": ["auth", "login", "authenticate", "verifyToken", "middleware"],
    },
    {
        "question": "how are passwords hashed or verified?",
        "relevant": ["hash", "bcrypt", "compare", "password"],
    },
    {
        "question": "where is the database connection established?",
        "relevant": ["connect", "pool", "db", "database", "mongoose", "sequelize", "knex"],
    },
    {
        "question": "how is user registration handled?",
        "relevant": ["register", "signup", "createUser", "create"],
    },
    {
        "question": "how are JWT tokens created or verified?",
        "relevant": ["jwt", "sign", "verify", "token", "secret"],
    },
    {
        "question": "how does the app send emails or notifications?",
        "relevant": ["email", "mail", "send", "notify", "notification", "smtp"],
    },
    {
        "question": "where are API routes defined?",
        "relevant": ["router", "route", "app.get", "app.post", "endpoint"],
    },
    {
        "question": "how is error handling done in the application?",
        "relevant": ["error", "catch", "handler", "next", "status"],
    },
    {
        "question": "how is file upload handled?",
        "relevant": ["upload", "multer", "file", "storage", "stream"],
    },
    {
        "question": "how is user profile or account data fetched?",
        "relevant": ["profile", "account", "getUser", "user", "fetch"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Hint-only relevance  (fast path, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _is_relevant_hint(sym: dict, relevant_hints: list[str]) -> bool:
    """
    True if any hint string appears as a case-insensitive substring in the
    symbol's name, display_name, or file path.
    """
    name    = (sym.get("name")   or "").lower()
    parent  = (sym.get("parent") or "").lower()
    file    = (sym.get("file")   or "").lower()
    display = f"{parent}.{name}" if parent else name

    candidates = [name, display, file]
    for hint in relevant_hints:
        h = hint.lower()
        if any(h in c for c in candidates):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-judge  (v2: context-quality, not symbol-identity)
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_block(symbols: list[dict], max_code_chars: int = 600) -> str:
    """
    Concatenate top-K symbols into a single readable context block for the judge.
    Each symbol gets a header and its first max_code_chars characters of code.
    """
    parts = []
    for i, sym in enumerate(symbols, 1):
        name   = sym.get("name", "unknown")
        parent = sym.get("parent", "")
        file   = sym.get("file", "")
        code   = (sym.get("code") or "")[:max_code_chars]
        display = f"{parent}.{name}" if parent else name
        parts.append(
            f"--- [{i}] {display}  ({file}) ---\n{code}"
        )
    return "\n\n".join(parts)


def _judge_context_quality(
    question:         str,
    reference_answer: str,
    retrieved:        list[dict],
    llm_fn,
) -> tuple[bool, int]:
    """
    Ask the LLM whether the retrieved context contains enough information to
    arrive at the reference answer.

    Returns (is_relevant: bool, best_rank: int).
    best_rank is the index (1-based) of the first symbol the LLM considers
    relevant when checked individually — used for MRR.  If the context as a
    whole passes but no individual symbol is pinpointed, rank defaults to 1.

    Strategy
    --------
    Step 1: Whole-context check (one LLM call).
        Pass all retrieved symbols as a combined block.  If the judge says NO,
        the query is a miss — no need to check individual symbols.

    Step 2: If YES, find the first individually relevant symbol for MRR
        (up to 5 symbols, one call each).  This is skipped if there is only
        one retrieved symbol.
    """
    if not retrieved:
        return False, 0

    context = _build_context_block(retrieved)

    whole_prompt = (
        f"You are evaluating a code retrieval system.\n\n"
        f"Developer question:\n{question}\n\n"
        f"Expected answer (derived from the source function):\n{reference_answer}\n\n"
        f"Retrieved code context:\n{context}\n\n"
        f"Does the retrieved context contain enough information to arrive at "
        f"the expected answer above?\n"
        f"Consider the context as a whole — individual functions may each "
        f"contribute part of the answer.\n"
        f"Reply with exactly one word: YES or NO."
    )

    try:
        whole_verdict = llm_fn(whole_prompt).strip().upper().startswith("YES")
    except Exception:
        return False, 0

    if not whole_verdict:
        return False, 0

    # Whole context passed — find first relevant symbol for MRR
    if len(retrieved) == 1:
        return True, 1

    for rank, sym in enumerate(retrieved, 1):
        sym_context = _build_context_block([sym])
        sym_prompt = (
            f"Developer question:\n{question}\n\n"
            f"Expected answer:\n{reference_answer}\n\n"
            f"Retrieved symbol:\n{sym_context}\n\n"
            f"Does this single symbol directly help answer the question "
            f"(even partially)?\n"
            f"Reply YES or NO only."
        )
        try:
            sym_verdict = llm_fn(sym_prompt).strip().upper().startswith("YES")
        except Exception:
            continue
        if sym_verdict:
            return True, rank

    # Context passed as a whole but no single symbol pinpointed — credit rank 1
    return True, 1


# ─────────────────────────────────────────────────────────────────────────────
# Auto query generation  (v2: stores reference_answer)
# ─────────────────────────────────────────────────────────────────────────────

def generate_queries(
    code_syms: list[dict],
    llm_fn,
    n:    int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    Auto-generate a query set from the index.

    For each sampled symbol the LLM writes TWO things:
      • question         — natural developer question (no function name)
      • reference_answer — one-sentence answer derived from that symbol's code

    Storing a reference_answer alongside each question fixes the judge's
    "what does correct mean?" problem: on ambiguous questions, the judge has
    a concrete anchor instead of making a free-form guess.

    The result is a list of dicts:
        {
            "question":         "...",
            "reference_answer": "...",
            "source_symbol":    "...",   // for debug tracing only
            "source_file":      "...",
            "relevant":         []       // empty; LLM judge is used at eval time
        }

    Parameters
    ----------
    code_syms : list of symbol dicts from Layer 1 / index_manager.load()
    llm_fn    : callable(prompt) -> str
    n         : number of queries to generate (default 20)
    seed      : random seed for reproducible sampling
    """
    import random
    rng = random.Random(seed)

    candidates = [
        s for s in code_syms
        if s.get("kind") in ("function", "method", "arrow_function")
        and len((s.get("code") or "")) > 80
    ]

    if not candidates:
        raise ValueError(
            "No function/method symbols found. Cannot generate queries — "
            "run Layer 1 first."
        )

    sample = rng.sample(candidates, min(n, len(candidates)))
    queries: list[dict] = []

    print(
        f"\n  [generate] Generating {len(sample)} queries from index symbols ...\n",
        file=sys.stderr,
    )

    for i, sym in enumerate(sample, 1):
        name   = sym.get("name", "unknown")
        kind   = sym.get("kind", "function")
        file   = sym.get("file", "")
        code   = (sym.get("code") or "")[:600]
        docstr = (sym.get("docstring") or "").strip()

        context_line = f"Docstring: {docstr}\n" if docstr else ""

        # Ask for BOTH question and reference_answer in one call (cheaper)
        prompt = (
            f"A developer is browsing an unfamiliar codebase and encounters "
            f"this {kind}:\n\n"
            f"File: {file}\n"
            f"{context_line}"
            f"Code:\n{code}\n\n"
            f"Your task: produce a JSON object with exactly two keys.\n\n"
            f"1. \"question\": ONE natural English question a developer would "
            f"type into a search box to find this code.\n"
            f"   Rules:\n"
            f"   - Do NOT mention the function name ({name}) or filename.\n"
            f"   - Be conceptual, not implementation-specific.\n"
            f"   - Bad: 'what does {name} do' or 'find {name}'\n"
            f"   - Good: 'where is password hashing handled'\n\n"
            f"2. \"reference_answer\": ONE sentence that directly answers that "
            f"question, based only on this code.\n"
            f"   - Name the specific mechanism, table, condition, or field "
            f"the code uses.\n"
            f"   - Do NOT just say 'it is handled in {name}'.\n"
            f"   - Bad: 'Password hashing is handled in the hashPassword function.'\n"
            f"   - Good: 'Passwords are hashed using bcrypt with a salt round "
            f"of 10 before being stored in the users table.'\n\n"
            f"Respond with ONLY valid JSON, no markdown, no explanation:\n"
            f"{{\"question\": \"...\", \"reference_answer\": \"...\"}}"
        )

        try:
            raw = llm_fn(prompt).strip()
            # Strip markdown fences if the model wraps in ```json ... ```
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            question         = parsed.get("question", "").strip().strip('"\'')
            reference_answer = parsed.get("reference_answer", "").strip().strip('"\'')
        except Exception as e:
            print(
                f"  [generate] LLM/parse error for {name}: {e}",
                file=sys.stderr,
            )
            continue

        if not question or not reference_answer:
            print(
                f"  [generate] Skipping {name}: empty question or reference_answer",
                file=sys.stderr,
            )
            continue

        queries.append({
            "question":         question,
            "reference_answer": reference_answer,
            "source_symbol":    name,
            "source_file":      file,
            "relevant":         [],   # LLM judge used at eval time
        })

        print(
            f"  [{i:02d}] {name:40s}\n"
            f"        Q: {question}\n"
            f"        A: {reference_answer}\n",
            file=sys.stderr,
        )

    print(
        f"  [generate] Done. {len(queries)} queries generated.\n",
        file=sys.stderr,
    )
    return queries


# ─────────────────────────────────────────────────────────────────────────────
# Naive chunking baseline
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_LINES   = 30
_CHUNK_OVERLAP = 5

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".next", "target", "out",
    "test", "tests", "examples",
}
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
}


def _chunk_file(file_path: str) -> list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []

    chunks: list[dict] = []
    step = _CHUNK_LINES - _CHUNK_OVERLAP
    for start in range(0, max(1, len(lines)), max(1, step)):
        end  = min(start + _CHUNK_LINES, len(lines))
        code = "".join(lines[start:end])
        if not code.strip():
            continue
        chunks.append({
            "name":       f"chunk_{start + 1}",
            "file":       file_path,
            "start_line": start + 1,
            "end_line":   end,
            "code":       code,
            "kind":       "chunk",
            "language":   "unknown",
            "parent":     "",
        })
        if end == len(lines):
            break
    return chunks


def build_naive_index(
    repo_path:  str,
    model_name: str = emb.DEFAULT_MODEL,
    batch_size: int = 64,
) -> tuple[faiss.Index, list[dict], SentenceTransformer]:
    repo   = Path(repo_path)
    chunks: list[dict] = []

    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in _CODE_EXTENSIONS:
                chunks.extend(_chunk_file(os.path.join(dirpath, fname)))

    print(f"  [naive] {len(chunks)} chunks from {repo}", file=sys.stderr)

    if not chunks:
        raise ValueError(f"No code files found under '{repo}'")

    model = SentenceTransformer(model_name)
    texts = [
        f"chunk in {c['file']} lines {c['start_line']}-{c['end_line']}\n{c['code'][:800]}"
        for c in chunks
    ]

    print(f"  [naive] Embedding {len(texts)} chunks ...", file=sys.stderr)
    vecs: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)  # type: ignore
    print(f"  [naive] FAISS index built: {index.ntotal} vectors", file=sys.stderr)
    return index, chunks, model


def query_naive(
    question: str,
    index:    faiss.Index,
    chunks:   list[dict],
    model:    SentenceTransformer,
    top_k:   int = 5,
) -> list[dict]:
    q_vec = model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    k = min(top_k, index.ntotal)
    _, ids = index.search(q_vec, k)  # type: ignore
    return [chunks[i] for i in ids[0].tolist() if i < len(chunks)]


# ─────────────────────────────────────────────────────────────────────────────
# Your system's retrieval
# ─────────────────────────────────────────────────────────────────────────────

def query_your_system(
    question:    str,
    faiss_code:  faiss.Index,
    bm25_code,
    code_syms:   list[dict],
    call_graph:  dict[str, list[str]],
    sym_by_name: dict[str, dict],
    model_name:  str,
    top_k:       int = 5,
    faiss_style: Optional[faiss.Index] = None,
    style_syms:  Optional[list[dict]] = None,
) -> list[dict]:
    retrieved = emb.query(
        question,
        faiss_code,
        bm25_code,
        code_syms,
        model_name=model_name,
        top_k=top_k,
        faiss_style=faiss_style,
        style_syms=style_syms or [],
    )
    return _expand_with_call_graph(retrieved, call_graph, sym_by_name)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring  (v2: unified hint + context-quality judge)
# ─────────────────────────────────────────────────────────────────────────────

def _score_results(
    results:          list[dict],
    relevant_hints:   list[str],
    k:                int,
    question:         str = "",
    reference_answer: str = "",
    llm_fn=None,
) -> tuple[int, float]:
    """
    Return (hit: 0|1, reciprocal_rank: float) for a single query.

    Judging strategy:

    1. Hint fast-path (free, no LLM):
       Walk results[:k] symbol by symbol.  If any hint matches, it's a hit
       at that rank.  Used when relevant_hints is non-empty regardless of
       whether llm_fn is set.

    2. LLM context-quality judge (v2):
       If hint fast-path didn't fire and llm_fn is provided, pass the FULL
       top-K context block to the judge in ONE call (see _judge_context_quality).
       The judge compares the context against reference_answer.
       Returns the rank of the first individually relevant symbol for MRR.

    3. No match: (0, 0.0).
    """
    clipped = results[:k]

    # Fast-path: hint matching (per-symbol, preserves rank for MRR)
    if relevant_hints:
        for rank, sym in enumerate(clipped, start=1):
            if _is_relevant_hint(sym, relevant_hints):
                return 1, 1.0 / rank

    # LLM context-quality judge
    if llm_fn and question and clipped:
        hit, best_rank = _judge_context_quality(
            question, reference_answer, clipped, llm_fn
        )
        if hit:
            return 1, 1.0 / best_rank

    return 0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    system_name: str
    recall_at_k: float
    mrr:         float
    k:           int
    per_query:   list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.system_name:30s}  "
            f"Recall@{self.k}={self.recall_at_k:.3f}  "
            f"MRR={self.mrr:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    queries:         list[dict],
    # Your system
    faiss_code:      faiss.Index,
    bm25_code,
    code_syms:       list[dict],
    call_graph:      dict[str, list[str]],
    sym_by_name:     dict[str, dict],
    model_name:      str,
    # Naive baseline
    naive_index:     faiss.Index,
    naive_chunks:    list[dict],
    naive_model:     SentenceTransformer,
    # Config
    k:               int = 5,
    faiss_style:     Optional[faiss.Index] = None,
    style_syms:      Optional[list[dict]] = None,
    judge_provider:  Optional[str] = None,
) -> tuple[EvalResult, EvalResult]:
    """
    Run both systems over all queries and return (your_result, naive_result).

    Each query dict may contain:
        question         — required
        reference_answer — used by LLM judge (from generate_queries)
        relevant         — used by hint fast-path (from hand-written sets)
    """
    llm_fn = _PROVIDERS.get(judge_provider) if judge_provider else None

    if llm_fn:
        print(
            f"  [judge] LLM-as-judge enabled (provider={judge_provider}) — "
            "using context-quality scoring with reference_answer anchor.",
            file=sys.stderr,
        )

    your_hits,  your_rr   = [], []
    naive_hits, naive_rr  = [], []
    your_per_query  = []
    naive_per_query = []

    print(
        f"\n  Running evaluation over {len(queries)} queries (K={k}) ...\n",
        file=sys.stderr,
    )

    for i, q in enumerate(queries, 1):
        question         = q.get("question", "")
        relevant_hints   = q.get("relevant", [])
        reference_answer = q.get("reference_answer", "")

        # ── Your system ───────────────────────────────────────────────────────
        y_results = query_your_system(
            question, faiss_code, bm25_code, code_syms,
            call_graph, sym_by_name, model_name,
            top_k=k, faiss_style=faiss_style, style_syms=style_syms,
        )
        y_hit, y_rr = _score_results(
            y_results, relevant_hints, k,
            question=question,
            reference_answer=reference_answer,
            llm_fn=llm_fn,
        )
        your_hits.append(y_hit)
        your_rr.append(y_rr)

        # ── Naive baseline ────────────────────────────────────────────────────
        n_results = query_naive(question, naive_index, naive_chunks, naive_model, top_k=k)
        n_hit, n_rr = _score_results(
            n_results, relevant_hints, k,
            question=question,
            reference_answer=reference_answer,
            llm_fn=llm_fn,
        )
        naive_hits.append(n_hit)
        naive_rr.append(n_rr)

        # ── Per-query debug record ────────────────────────────────────────────
        y_sym  = (
            f"{y_results[0].get('parent','')}.{y_results[0]['name']}"
            if y_results else "—"
        )
        n_file = (
            f"{n_results[0]['file']}:{n_results[0]['start_line']}"
            if n_results else "—"
        )
        ref_snippet = (reference_answer[:80] + "…") if len(reference_answer) > 80 \
                      else reference_answer

        print(
            f"  [{i:02d}] {question}\n"
            f"        Ref: {ref_snippet}\n"
            f"        Your:  {'✓' if y_hit else '✗'}  top={y_sym[:80]}\n"
            f"        Naive: {'✓' if n_hit else '✗'}  top={n_file[:80]}\n",
            file=sys.stderr,
        )

        your_per_query.append({
            "question":         question,
            "reference_answer": reference_answer,
            "source_symbol":    q.get("source_symbol", ""),
            "hit":              bool(y_hit),
            "rr":               round(y_rr, 4),
            "top_results": [
                f"{s.get('parent','')}.{s['name']}  ({s.get('file','')})"
                for s in y_results[:k]
            ],
        })
        naive_per_query.append({
            "question":         question,
            "reference_answer": reference_answer,
            "source_symbol":    q.get("source_symbol", ""),
            "hit":              bool(n_hit),
            "rr":               round(n_rr, 4),
            "top_results": [
                f"{c['file']} L{c['start_line']}-{c['end_line']}"
                for c in n_results[:k]
            ],
        })

    n = len(queries)
    your_result = EvalResult(
        system_name="Code Intelligence (yours)",
        recall_at_k=sum(your_hits) / n,
        mrr=sum(your_rr) / n,
        k=k,
        per_query=your_per_query,
    )
    naive_result = EvalResult(
        system_name="Naive chunking (baseline)",
        recall_at_k=sum(naive_hits) / n,
        mrr=sum(naive_rr) / n,
        k=k,
        per_query=naive_per_query,
    )
    return your_result, naive_result


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(
    your_result:  EvalResult,
    naive_result: EvalResult,
    out_path:     str = "eval_results.png",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[eval] matplotlib not installed. Run: pip install matplotlib",
              file=sys.stderr)
        return

    k       = your_result.k
    metrics = [f"Recall@{k}", "MRR"]
    your_vals  = [your_result.recall_at_k,  your_result.mrr]
    naive_vals = [naive_result.recall_at_k, naive_result.mrr]

    x     = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars_your  = ax.bar(x - width / 2, your_vals,  width, color="#4f8ef7", zorder=3)
    bars_naive = ax.bar(x + width / 2, naive_vals, width, color="#f74f4f", zorder=3)

    for bar in bars_your:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", color="white", fontsize=11, fontweight="bold",
        )
    for bar in bars_naive:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", color="white", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="white", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="#aaaaaa", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.yaxis.grid(True, color="#222222", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    patch_your  = mpatches.Patch(color="#4f8ef7", label="Code Intelligence (yours)")
    patch_naive = mpatches.Patch(color="#f74f4f", label="Naive chunking (baseline)")
    ax.legend(
        handles=[patch_your, patch_naive],
        facecolor="#1a1d27", edgecolor="#333333",
        labelcolor="white", fontsize=10,
    )
    ax.set_title(
        f"Code Intelligence vs. Naive Chunking  "
        f"(n={len(your_result.per_query)} queries)",
        color="white", fontsize=13, pad=14,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Layer 6 — Evaluation: Recall@K and MRR vs. naive chunking baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
          # Auto-generate queries, judge with LLM (recommended):
          python eval.py ./repo_index --repo /path/to/repo \\
              --generate-queries --judge-provider auto

          # Save generated queries for reuse:
          python eval.py ./repo_index --repo /path/to/repo \\
              --generate-queries --save-queries generated.json --judge-provider auto

          # Reuse saved queries:
          python eval.py ./repo_index --repo /path/to/repo \\
              --queries generated.json --judge-provider auto

          # Hand-written query file with hint matching (no LLM):
          python eval.py ./repo_index --repo /path/to/repo --queries my_queries.json

          # Show built-in sample queries:
          python eval.py --show-queries

        Query JSON format
        -----------------
          [
            {
              "question":         "where is authentication handled?",
              "reference_answer": "JWT tokens are verified in auth middleware",
              "relevant":         ["auth", "login"]
            },
            ...
          ]
          "reference_answer" is used by --judge-provider.
          "relevant"         is used by hint-only mode.
        """),
    )
    ap.add_argument("index_dir", nargs="?",
                    help="Path to the index directory built by index_manager.py")
    ap.add_argument("--repo",    metavar="REPO_PATH",
                    help="Root of the source repo (for naive baseline chunking)")
    ap.add_argument("--queries", metavar="JSON_FILE",
                    help="Path to a JSON query set (default: built-in samples)")
    ap.add_argument("--generate-queries", action="store_true",
                    help="Auto-generate queries from the index using the LLM.")
    ap.add_argument("--generate-n", type=int, default=20, metavar="N",
                    help="Number of queries to generate (default: 20)")
    ap.add_argument("--save-queries", metavar="PATH",
                    help="Save generated queries to a JSON file for reuse")
    ap.add_argument("--k",      type=int, default=5,
                    help="Retrieval depth K (default: 5)")
    ap.add_argument("--model",  default=emb.DEFAULT_MODEL,
                    help=f"sentence-transformers model (default: {emb.DEFAULT_MODEL})")
    ap.add_argument("--plot",   default="eval_results.png",
                    help="Output path for the comparison chart")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip chart generation")
    ap.add_argument("--judge-provider",
                    choices=["gemini", "groq", "auto"], default=None,
                    metavar="PROVIDER",
                    help="Enable LLM-as-judge (gemini/groq/auto). "
                         "Required for --generate-queries.")
    ap.add_argument("--save-json", metavar="PATH",
                    help="Save per-query results to a JSON file")
    ap.add_argument("--show-queries", action="store_true",
                    help="Print the built-in query set as JSON and exit")
    args = ap.parse_args()

    if args.show_queries:
        print(json.dumps(SAMPLE_QUERIES, indent=2))
        return

    if not args.index_dir:
        ap.error("index_dir is required (unless using --show-queries)")
    if not args.repo:
        ap.error("--repo is required")
    if args.generate_queries and not args.judge_provider:
        ap.error("--generate-queries requires --judge-provider (gemini/groq/auto)")

    llm_fn = _PROVIDERS.get(args.judge_provider) if args.judge_provider else None

    print(f"\n  Loading index from '{args.index_dir}' ...", file=sys.stderr)
    faiss_code, bm25_code, faiss_style, code_syms, style_syms, call_graph = \
        index_manager.load(args.index_dir)
    print(
        f"  Index loaded: {len(code_syms)} code symbols, "
        f"{len(call_graph)} call graph nodes",
        file=sys.stderr,
    )

    sym_by_name: dict[str, dict] = {}
    for sym in code_syms:
        name    = sym.get("name",   "")
        parent  = sym.get("parent", "")
        display = f"{parent}.{name}" if parent else name
        sym_by_name[display] = sym
        sym_by_name[name]    = sym

    # ── Load or generate query set ────────────────────────────────────────────
    if args.generate_queries:
        queries = generate_queries(code_syms, llm_fn, n=args.generate_n)
        if args.save_queries:
            with open(args.save_queries, "w", encoding="utf-8") as f:
                json.dump(queries, f, indent=2)
            print(f"  Generated queries saved → {args.save_queries}", file=sys.stderr)
    elif args.queries:
        with open(args.queries, "r", encoding="utf-8") as f:
            queries: list[dict] = json.load(f)
        print(f"  Loaded {len(queries)} queries from {args.queries}", file=sys.stderr)
    else:
        queries = SAMPLE_QUERIES
        print(
            f"  Using {len(queries)} built-in sample queries.\n"
            "  (Run --show-queries to see/edit them, or use --generate-queries.)\n",
            file=sys.stderr,
        )

    # ── Build naive baseline ──────────────────────────────────────────────────
    print(f"\n  Building naive baseline for '{args.repo}' ...", file=sys.stderr)
    naive_index, naive_chunks, naive_model = build_naive_index(
        args.repo, model_name=args.model,
    )

    # ── Run evaluation ────────────────────────────────────────────────────────
    your_result, naive_result = evaluate(
        queries=queries,
        faiss_code=faiss_code,
        bm25_code=bm25_code,
        code_syms=code_syms,
        call_graph=call_graph,
        sym_by_name=sym_by_name,
        model_name=args.model,
        naive_index=naive_index,
        naive_chunks=naive_chunks,
        naive_model=naive_model,
        k=args.k,
        faiss_style=faiss_style,
        style_syms=style_syms,
        judge_provider=args.judge_provider,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Evaluation Results  (n={len(queries)} queries, K={args.k})")
    print(f"{'='*62}")
    print(f"  {your_result}")
    print(f"  {naive_result}")

    delta_recall = your_result.recall_at_k - naive_result.recall_at_k
    delta_mrr    = your_result.mrr          - naive_result.mrr
    sign_r = "+" if delta_recall >= 0 else ""
    sign_m = "+" if delta_mrr    >= 0 else ""
    print(f"\n  Delta  Recall@{args.k}: {sign_r}{delta_recall:.3f}   MRR: {sign_m}{delta_mrr:.3f}")
    print(f"{'='*62}\n")

    # ── Save per-query JSON ───────────────────────────────────────────────────
    if args.save_json:
        out = {
            "k":     args.k,
            "model": args.model,
            "your_system": {
                "recall_at_k": your_result.recall_at_k,
                "mrr":         your_result.mrr,
                "per_query":   your_result.per_query,
            },
            "naive_baseline": {
                "recall_at_k": naive_result.recall_at_k,
                "mrr":         naive_result.mrr,
                "per_query":   naive_result.per_query,
            },
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"  Per-query results saved → {args.save_json}", file=sys.stderr)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot(your_result, naive_result, out_path=args.plot)


if __name__ == "__main__":
    main()