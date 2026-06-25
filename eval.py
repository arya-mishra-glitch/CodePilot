"""
Layer 6 — Evaluation (Recall@K and MRR vs. Naive Chunking Baseline)
====================================================================
Measures whether your AST-aware retrieval system actually finds the right code
more often than a naive text-chunking approach.

What this file does
-------------------
  1. naive_baseline  — chunks every source file into fixed N-line windows,
                       embeds them with the same model as Layer 3, and answers
                       queries via pure FAISS (no call graph, no BM25).
  2. evaluate        — runs both systems over a query set, measures Recall@K
                       and MRR for each, and returns a results dict.
  3. plot            — draws a side-by-side bar chart and saves it to disk.
  4. CLI             — lets you run everything from the terminal, and also lets
                       you print/edit the built-in sample query set.

What is a "query set"?
----------------------
A JSON file (or Python list) of dicts, each with:

    {
        "question": "where is authentication handled?",
        "relevant": ["AuthManager.login", "verifyToken", "auth/middleware.js"]
    }

"relevant" is a list of strings — a retrieved symbol is counted as a hit if
any of its fields (name, display_name, file) contains any of these strings as
a substring.  This is intentionally lenient: you don't need exact matches.

Metrics
-------
  Recall@K   — fraction of queries where at least one relevant symbol appears
                in the top-K results.  The primary metric.

  MRR        — Mean Reciprocal Rank: 1/rank of the first relevant result,
                averaged across queries.  Rewards finding the right answer
                first, not just somewhere in the top K.

Usage
-----
    # Run evaluation with the built-in sample queries on a repo:
    python eval.py ./repo_index --repo /path/to/repo

    # Use your own query set:
    python eval.py ./repo_index --queries my_queries.json

    # Show the built-in sample queries (edit them to match your repo):
    python eval.py --show-queries

    # Adjust K (default: 5):
    python eval.py ./repo_index --k 10

    # Save the plot to a custom path:
    python eval.py ./repo_index --plot eval_results.png

Dependencies
------------
    pip install sentence-transformers faiss-cpu matplotlib
    (index_manager.py, embedder.py, call_graph.py, ast_parser.py also required)
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


# ── Layer imports ─────────────────────────────────────────────────────────────
try:
    import index_manager
    import embedder as emb
    from query_engine import _expand_with_call_graph
except ImportError as e:
    print(
        f"[eval] ImportError: {e}\n"
        "Make sure index_manager.py, embedder.py, and query_engine.py are on sys.path.",
        file=sys.stderr,
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in sample query set
#
# These are written for a generic Node/Express + React codebase.
# EDIT THESE to match your actual repo before running eval.
# Run `python eval.py --show-queries` to print them as JSON.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_QUERIES: list[dict] = [
    {
        "question": "where is user authentication handled?",
        "relevant": ["login", "authenticate", "verifyToken", "auth"],
    },
    {
        "question": "how is the database connection set up?",
        "relevant": ["connect", "pool", "db", "database", "mongoose", "sequelize"],
    },
    {
        "question": "where are passwords hashed or compared?",
        "relevant": ["hash", "bcrypt", "compare", "password"],
    },
    {
        "question": "how are errors handled in middleware?",
        "relevant": ["errorHandler", "error", "middleware", "next"],
    },
    {
        "question": "where is the JWT token created or verified?",
        "relevant": ["jwt", "sign", "verify", "token"],
    },
    {
        "question": "how does the app handle user registration?",
        "relevant": ["register", "signup", "createUser", "newUser"],
    },
    {
        "question": "where are API routes defined?",
        "relevant": ["router", "route", "app.get", "app.post"],
    },
    {
        "question": "how is input validation done?",
        "relevant": ["validate", "sanitize", "check", "joi", "zod"],
    },
    {
        "question": "where is the user session managed?",
        "relevant": ["session", "cookie", "passport", "req.user"],
    },
    {
        "question": "how does the app send emails?",
        "relevant": ["email", "mail", "nodemailer", "sendMail", "smtp"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Relevance matching
# ─────────────────────────────────────────────────────────────────────────────

def _is_relevant(sym: dict, relevant_hints: list[str]) -> bool:
    """
    Return True if this symbol is a plausible hit for the given query.

    A symbol is relevant if any of the hint strings appears (case-insensitive)
    in its name, display_name, or file path.  This is intentionally lenient —
    you're measuring whether the system found something reasonable, not whether
    it found exactly the one function you had in mind.
    """
    name    = sym.get("name",   "").lower()
    parent  = sym.get("parent", "").lower()
    file    = sym.get("file",   "").lower()
    display = f"{parent}.{name}" if parent else name

    candidates = [name, display, file]
    for hint in relevant_hints:
        h = hint.lower()
        if any(h in c for c in candidates):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Naive chunking baseline
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_LINES   = 30   # lines per chunk
_CHUNK_OVERLAP = 5    # overlap between consecutive chunks

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".next", "target", "out",
    "test", "tests", "examples"
}
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp",
}


def _chunk_file(file_path: str) -> list[dict]:
    """
    Split a source file into overlapping fixed-size line windows.
    Returns a list of chunk dicts: {file, start_line, end_line, code, name}.
    """
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
    repo_path: str,
    model_name: str = emb.DEFAULT_MODEL,
    batch_size: int = 64,
) -> tuple[faiss.Index, list[dict], SentenceTransformer]:
    """
    Build a naive baseline index: chunk every source file into fixed-size
    windows and embed them — no AST, no call graph, no BM25.

    Returns
    -------
    (faiss_index, chunks, model)
    """
    repo = Path(repo_path)
    chunks: list[dict] = []

    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in _CODE_EXTENSIONS:
                full = os.path.join(dirpath, fname)
                chunks.extend(_chunk_file(full))

    print(f"  [naive] {len(chunks)} chunks from {repo}", file=sys.stderr)

    if not chunks:
        raise ValueError(f"No code files found under '{repo}'")

    model = SentenceTransformer(model_name)

    # Simple text: just the code (mirrors what emb._build_text does for code)
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

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs) #type:ignore
    print(f"  [naive] FAISS index built: {index.ntotal} vectors", file=sys.stderr)

    return index, chunks, model


def query_naive(
    question: str,
    index:    faiss.Index,
    chunks:   list[dict],
    model:    SentenceTransformer,
    top_k:    int = 5,
) -> list[dict]:
    """Retrieve top_k chunks for a question using the naive FAISS index."""
    q_vec = model.encode([question], normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
    k     = min(top_k, index.ntotal)
    _, ids = index.search(q_vec, k) #type:ignore
    return [chunks[i] for i in ids[0].tolist() if i < len(chunks)]


# ─────────────────────────────────────────────────────────────────────────────
# Your system's retrieval (wrapper around embedder + call graph)
# ─────────────────────────────────────────────────────────────────────────────

def query_your_system(
    question:        str,
    faiss_code:      faiss.Index,
    bm25_code,
    code_syms:       list[dict],
    call_graph:      dict[str, list[str]],
    sym_by_name:     dict[str, dict],
    model_name:      str,
    top_k:           int = 5,
    faiss_style:     Optional[faiss.Index] = None,
    style_syms:      Optional[list[dict]] = None,
) -> list[dict]:
    """
    Retrieve via FAISS + BM25 (RRF fusion) + call-graph expansion.
    This is exactly what QueryEngine.query() does, extracted for eval use.
    """
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
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    system_name:  str
    recall_at_k:  float          # fraction of queries with a hit in top-K
    mrr:          float          # Mean Reciprocal Rank
    k:            int
    per_query:    list[dict] = field(default_factory=list)   # debug detail

    def __str__(self) -> str:
        return (
            f"{self.system_name:30s}  "
            f"Recall@{self.k}={self.recall_at_k:.3f}  "
            f"MRR={self.mrr:.3f}"
        )


def _score_results(
    results:         list[dict],
    relevant_hints:  list[str],
    k:               int,
) -> tuple[int, float]:
    """
    For a single query, return (hit, reciprocal_rank).

    hit              : 1 if any of the top-k results is relevant, else 0
    reciprocal_rank  : 1/rank of the first relevant result (0 if none in top-k)
    """
    for rank, sym in enumerate(results[:k], start=1):
        if _is_relevant(sym, relevant_hints):
            return 1, 1.0 / rank
    return 0, 0.0


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
) -> tuple[EvalResult, EvalResult]:
    """
    Run both systems over all queries and return (your_result, naive_result).
    """
    your_hits, your_rr   = [], []
    naive_hits, naive_rr = [], []

    your_per_query  = []
    naive_per_query = []

    print(f"\n  Running evaluation over {len(queries)} queries (K={k}) ...\n",
          file=sys.stderr)

    for i, q in enumerate(queries, 1):
        question = q["question"]
        relevant = q["relevant"]

        # ── Your system ───────────────────────────────────────────────────────
        y_results = query_your_system(
            question, faiss_code, bm25_code, code_syms,
            call_graph, sym_by_name, model_name,
            top_k=k, faiss_style=faiss_style, style_syms=style_syms,
        )
        y_hit, y_rr = _score_results(y_results, relevant, k)
        your_hits.append(y_hit)
        your_rr.append(y_rr)

        # ── Naive baseline ────────────────────────────────────────────────────
        n_results = query_naive(question, naive_index, naive_chunks, naive_model, top_k=k)
        n_hit, n_rr = _score_results(n_results, relevant, k)
        naive_hits.append(n_hit)
        naive_rr.append(n_rr)

        your_per_query.append({
            "question": question,
            "hit": bool(y_hit),
            "rr":  round(y_rr, 4),
            "top_results": [
                f"{s.get('parent','')}.{s['name']}  ({s.get('file','')})"
                for s in y_results[:k]
            ],
        })
        naive_per_query.append({
            "question": question,
            "hit": bool(n_hit),
            "rr":  round(n_rr, 4),
            "top_results": [
                f"{c['file']} L{c['start_line']}-{c['end_line']}"
                for c in n_results[:k]
            ],
        })

        # Progress
        y_sym = f"{y_results[0].get('parent','')}.{y_results[0]['name']}" if y_results else "—"
        n_file = f"{n_results[0]['file']}:{n_results[0]['start_line']}" if n_results else "—"
        hit_char_y = "✓" if y_hit else "✗"
        hit_char_n = "✓" if n_hit else "✗"
        print(
            f"  [{i:02d}] {question[:50]:<50s}\n"
            f"         Your: {hit_char_y}  top={y_sym[:50]}\n"
            f"        Naive: {hit_char_n}  top={n_file[:50]}\n",
            file=sys.stderr,
        )

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
    """
    Draw a side-by-side bar chart comparing the two systems.
    Saves to out_path and attempts to open it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — works in Colab and CI
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[eval] matplotlib not installed. Run: pip install matplotlib",
              file=sys.stderr)
        return

    k = your_result.k
    metrics = [f"Recall@{k}", "MRR"]
    your_vals  = [your_result.recall_at_k,  your_result.mrr]
    naive_vals = [naive_result.recall_at_k, naive_result.mrr]

    x     = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Bars
    bars_your  = ax.bar(x - width / 2, your_vals,  width, color="#4f8ef7", zorder=3)
    bars_naive = ax.bar(x + width / 2, naive_vals, width, color="#f74f4f", zorder=3)

    # Value labels on bars
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

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="white", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="#aaaaaa", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.yaxis.grid(True, color="#222222", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    patch_your  = mpatches.Patch(color="#4f8ef7", label="Code Intelligence (yours)")
    patch_naive = mpatches.Patch(color="#f74f4f", label="Naive chunking (baseline)")
    ax.legend(handles=[patch_your, patch_naive], facecolor="#1a1d27",
              edgecolor="#333333", labelcolor="white", fontsize=10)

    # Title
    ax.set_title(
        f"Code Intelligence vs. Naive Chunking  (n={len(your_result.per_query)} queries)",
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
          # Run with built-in sample queries (edit them first to match your repo):
          python eval.py ./repo_index --repo /path/to/repo

          # Use a custom query JSON file:
          python eval.py ./repo_index --repo /path/to/repo --queries my_queries.json

          # Show the built-in query set (copy & edit):
          python eval.py --show-queries

          # Evaluate at K=10, save plot to a custom path:
          python eval.py ./repo_index --repo /path/to/repo --k 10 --plot chart.png

        Query JSON format
        -----------------
          [
            {
              "question": "where is user authentication handled?",
              "relevant": ["login", "authenticate", "auth/middleware.js"]
            },
            ...
          ]

          A symbol is counted as a hit if any hint string appears as a
          substring (case-insensitive) in its name, display_name, or file path.
        """),
    )
    ap.add_argument("index_dir", nargs="?",
                    help="Path to the index directory built by index_manager.py")
    ap.add_argument("--repo",    metavar="REPO_PATH",
                    help="Root of the source repo (for naive baseline chunking)")
    ap.add_argument("--queries", metavar="JSON_FILE",
                    help="Path to a JSON query set (default: built-in samples)")
    ap.add_argument("--k",       type=int, default=5,
                    help="Retrieval depth K (default: 5)")
    ap.add_argument("--model",   default=emb.DEFAULT_MODEL,
                    help=f"sentence-transformers model (default: {emb.DEFAULT_MODEL})")
    ap.add_argument("--plot",    default="eval_results.png",
                    help="Output path for the comparison chart (default: eval_results.png)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip chart generation")
    ap.add_argument("--save-json", metavar="PATH",
                    help="Save per-query results to a JSON file")
    ap.add_argument("--show-queries", action="store_true",
                    help="Print the built-in query set as JSON and exit")
    args = ap.parse_args()

    # ── --show-queries ────────────────────────────────────────────────────────
    if args.show_queries:
        print(json.dumps(SAMPLE_QUERIES, indent=2))
        print(
            "\n# Copy the above JSON to a file, edit the questions and relevant hints\n"
            "# to match your repo, then run:\n"
            "#   python eval.py ./repo_index --repo /path/to/repo --queries my_queries.json",
            file=sys.stderr,
        )
        return

    # ── Validate args ─────────────────────────────────────────────────────────
    if not args.index_dir:
        ap.error("index_dir is required (unless using --show-queries)")
    if not args.repo:
        ap.error("--repo is required (path to the source repo for naive baseline)")

    # ── Load query set ────────────────────────────────────────────────────────
    if args.queries:
        with open(args.queries, "r", encoding="utf-8") as f:
            queries: list[dict] = json.load(f)
        print(f"  Loaded {len(queries)} queries from {args.queries}", file=sys.stderr)
    else:
        queries = SAMPLE_QUERIES
        print(f"  Using {len(queries)} built-in sample queries.", file=sys.stderr)
        print("  (Run --show-queries to see and edit them.)\n", file=sys.stderr)

    # ── Load your system's index ──────────────────────────────────────────────
    print(f"\n  Loading index from '{args.index_dir}' ...", file=sys.stderr)
    faiss_code, bm25_code, faiss_style, code_syms, style_syms, call_graph = \
        index_manager.load(args.index_dir)
    print(f"  Index loaded: {len(code_syms)} code symbols, {len(call_graph)} call graph nodes",
          file=sys.stderr)

    # Build name → symbol map for call-graph expansion
    sym_by_name: dict[str, dict] = {}
    for sym in code_syms:
        name    = sym.get("name",   "")
        parent  = sym.get("parent", "")
        display = f"{parent}.{name}" if parent else name
        sym_by_name[display] = sym
        sym_by_name[name]    = sym

    # ── Build naive baseline ──────────────────────────────────────────────────
    print(f"\n  Building naive baseline index for '{args.repo}' ...", file=sys.stderr)
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
            "k": args.k,
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