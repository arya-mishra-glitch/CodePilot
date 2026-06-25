"""
Layer 3 — Search Index Builder (Embeddings + FAISS + BM25)
===========================================================
Reads the tagged symbols.json produced by Layer 2 and builds two parallel
search indexes:

    1. FAISS  — dense vector index (semantic similarity)
    2. BM25   — sparse keyword index (exact / token-overlap search)

Both indexes cover the same set of symbols: every entry whose
search_tier == "code".  CSS symbols (search_tier == "style") go into a
separate, lighter FAISS index so style questions can be answered too.

Output (written to --out directory, default ./repo_index)
---------
    faiss_code.index    ← FAISS index for code symbols
    faiss_style.index   ← FAISS index for CSS/style symbols  (may be absent)
    bm25_code.pkl       ← BM25 index for code symbols
    index_meta.json     ← id → symbol mapping consumed by Layers 4 & 5

Usage
-----
    # Build index from symbols.json produced by Layer 2:
    python embedder.py repo_index/symbols.json

    # Custom output directory:
    python embedder.py repo_index/symbols.json --out repo_index

    # Use a larger model (slower, better quality):
    python embedder.py repo_index/symbols.json --model all-mpnet-base-v2

    # Query the index interactively after building:
    python embedder.py repo_index/symbols.json --query "where is auth handled"

    # Only query (skip rebuild if index already exists):
    python embedder.py repo_index/symbols.json --query "login flow" --no-rebuild

Dependencies
------------
    pip install sentence-transformers faiss-cpu rank_bm25
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional
from typing import cast

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Default model: fast + good for code.  Swap for all-mpnet-base-v2 for ~5%
# better retrieval at ~3× the encoding time.
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# How many results each sub-index returns before the merge step
_FAISS_TOP_K = 15
_BM25_TOP_K  = 15

# Fusion weight: final_score = FAISS_WEIGHT * faiss_score + BM25_WEIGHT * bm25_score
# Both scores are normalised to [0, 1] before combining.
FAISS_WEIGHT = 0.6
BM25_WEIGHT  = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Text representation
#
# The embedding quality is the biggest lever you have in Layer 3.
# A function whose code is just "const x = () => {}" embeds poorly on its own.
# We build a richer text blob by combining:
#   • display_name  — unique identifier that the model can tokenise meaningfully
#   • docstring     — natural-language description when present  (0 in your current
#                     corpus, but leave the hook here for when you add them)
#   • file path     — gives context: "auth/middleware.js" signals authentication
#   • first N lines of code — enough structure for the model, avoids padding waste
#
# CSS symbols get a leaner representation since their "code" is already concise.
# ─────────────────────────────────────────────────────────────────────────────

_CODE_PREVIEW_LINES = 35   # embed this many lines of source; rest is noise

import re
def _tokenize_query(question: str) -> list[str]:
    raw = re.split(r"[\s\(\)\{\}\[\]<>,.;:\"'/\\|=\-+*&^%$#@!~`]+", question)
    expanded = []
    for tok in raw:
        if tok:
            sub = re.sub(r"([a-z])([A-Z])", r"\1 \2", tok).replace("_", " ")
            expanded.extend(sub.split())
    return [t.lower() for t in expanded if len(t) > 1]


"""
    Build the text that will be embedded for a single symbol.

    We intentionally keep this readable rather than terse:
    the model was trained on natural English + code, so a
    sentence-like header helps it anchor the meaning.
"""

def _build_text(sym: dict, callers: list[str] | None = None, callees: list[str] | None = None) -> str:
    parts: list[str] = []

    kind    = sym.get("kind",   "symbol")
    name    = sym.get("name",   "unknown")
    parent  = sym.get("parent", "")
    file    = sym.get("file",   "")
    display = f"{parent}.{name}" if parent else name

    parts.append(f"{kind} {display} in {file}")

    doc = sym.get("docstring", "")
    if doc:
        parts.append(doc.strip())

    # ← NEW: structural context
    if callers:
        parts.append(f"Called by: {', '.join(callers[:5])}")
    if callees:
        parts.append(f"Calls: {', '.join(callees[:5])}")

    tables = sym.get("tables_used", [])
    if tables:
        parts.append(f"SQL tables: {', '.join(tables)}")

    code = sym.get("code", "")
    if code:
        lines = code.splitlines()[:_CODE_PREVIEW_LINES]
        parts.append("\n".join(lines))

    return "\n".join(parts)

def _build_bm25_tokens(sym: dict) -> list[str]:
    """
    Tokenise a symbol for BM25.

    BM25 is a bag-of-words model, so we want every meaningful token:
    split on whitespace AND common code punctuation so that
    "getUserById" becomes ["getUserById", "get", "User", "By", "Id"]
    giving both exact-name matches and sub-word matches.
    """
    import re
    text = _build_text(sym)
    # Split on whitespace + code separators
    raw_tokens = re.split(r"[\s\(\)\{\}\[\]<>,.;:\"'/\\|=\-+*&^%$#@!~`]+", text)
    # camelCase / PascalCase split: "getUserById" → ["get", "User", "By", "Id"]
    expanded: list[str] = []
    for tok in raw_tokens:
        if tok:
            # Insert space before each uppercase run following a lowercase
            sub = re.sub(r"([a-z])([A-Z])", r"\1 \2", tok)
            # Split snake_case
            sub = sub.replace("_", " ")
            expanded.extend(sub.split())
    return [t.lower() for t in expanded if len(t) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# Index building
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    symbols: list[dict],
    call_graph = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> tuple[
    faiss.Index,             # FAISS index for code symbols
    BM25Okapi,               # BM25 index for code symbols
    Optional[faiss.Index],   # FAISS index for style symbols (or None)
    list[dict],              # code symbols in index order
    list[dict],              # style symbols in index order
]:
    """
    Build all search indexes from a flat symbol list.

    Parameters
    ----------
    symbols    : flat list of symbol dicts from Layer 2's symbols.json
    model_name : sentence-transformers model name
    batch_size : number of texts to embed per forward pass

    Returns
    -------
    (faiss_code, bm25_code, faiss_style, code_syms, style_syms)

    The *_syms lists are the ordered lists of symbols whose row i in the
    FAISS index corresponds to symbol i in the list — critical for lookup.
    """
    # ── Partition symbols ────────────────────────────────────────────────────
    # build reverse map
    callers_map: dict[str, list[str]] = {}
    if call_graph:
        for caller, callees in call_graph.items():
            for callee in callees:
                callers_map.setdefault(callee, []).append(caller)

    code_syms  = [s for s in symbols if s.get("search_tier") == "code"]
    style_syms = [s for s in symbols if s.get("search_tier") == "style"]

    print(f"    Symbols to embed: {len(code_syms)} code, {len(style_syms)} style",
          file=sys.stderr)

    if not code_syms:
        raise ValueError(
            "No symbols with search_tier='code' found. "
            "Did you run call_graph.py with tag_symbols() first?"
        )

    def _get_display(s):
        p = s.get("parent", "")
        n = s.get("name", "")
        return f"{p}.{n}" if p else n

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"    Loading model '{model_name}' ...", file=sys.stderr)
    model = SentenceTransformer(model_name)
    dim = model.get_embedding_dimension()
    print(f"    Embedding dimension: {dim}", file=sys.stderr)

    # ── Embed code symbols (with call-graph context) ─────────────────────────
    code_texts = []
    for s in code_syms:
        display = _get_display(s)
        callers = callers_map.get(display, [])
        callees = call_graph.get(display, []) if call_graph else []
        code_texts.append(_build_text(s, callers=callers, callees=callees))
    print(f"    Encoding {len(code_texts)} code symbols ...", file=sys.stderr)
    code_vecs: np.ndarray = model.encode(
        code_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )                                 # shape: (N, dim)

    # ── Build FAISS index (code) ─────────────────────────────────────────────
    # IndexFlatIP = exact inner-product search.  With normalised vectors this
    # is equivalent to cosine similarity.  For >100k symbols, swap to
    # IndexIVFFlat for ~10× speed at tiny accuracy cost.
    faiss_code = faiss.IndexFlatIP(dim)
    faiss_code.add(code_vecs.astype(np.float32)) # type: ignore
    print(f"    FAISS code index: {faiss_code.ntotal} vectors", file=sys.stderr)

    # ── Build BM25 index (code) ──────────────────────────────────────────────
    tokenised = [_build_bm25_tokens(s) for s in code_syms]
    bm25_code = BM25Okapi(tokenised)
    print(f"    BM25 code index built.", file=sys.stderr)

    # ── Embed + index style symbols (optional) ───────────────────────────────
    faiss_style: Optional[faiss.Index] = None
    if style_syms:
        style_texts = [_build_text(s) for s in style_syms]
        print(f"    Encoding {len(style_texts)} style symbols ...", file=sys.stderr)
        style_vecs: np.ndarray = model.encode(
            style_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        faiss_style = faiss.IndexFlatIP(dim)
        faiss_style.add(style_vecs.astype(np.float32)) # type: ignore
        print(f"    FAISS style index: {faiss_style.ntotal} vectors", file=sys.stderr)
    else:
        print("    No style symbols — skipping style index.", file=sys.stderr)

    return faiss_code, bm25_code, faiss_style, code_syms, style_syms


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_index(
    out_dir: str | Path,
    faiss_code:  faiss.Index,
    bm25_code:   BM25Okapi,
    faiss_style: Optional[faiss.Index],
    code_syms:   list[dict],
    style_syms:  list[dict],
) -> None:
    """
    Write all index artefacts to out_dir.

    File layout
    -----------
        faiss_code.index    — FAISS code index
        faiss_style.index   — FAISS style index (only written when non-empty)
        bm25_code.pkl       — BM25 code index (pickle)
        index_meta.json     — {
                                  "code":  [ sym, ... ],   # ordered by FAISS row
                                  "style": [ sym, ... ]    # ordered by FAISS row
                              }

    index_meta.json is the single source of truth that maps FAISS row → symbol.
    Layer 4 will copy it verbatim; Layer 5 looks up results here.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_code, str(out / "faiss_code.index"))
    print(f"    Saved -> {out / 'faiss_code.index'}", file=sys.stderr)

    with open(out / "bm25_code.pkl", "wb") as f:
        pickle.dump(bm25_code, f)
    print(f"    Saved -> {out / 'bm25_code.pkl'}", file=sys.stderr)

    if faiss_style is not None:
        faiss.write_index(faiss_style, str(out / "faiss_style.index"))
        print(f"    Saved -> {out / 'faiss_style.index'}", file=sys.stderr)

    meta = {"code": code_syms, "style": style_syms}
    with open(out / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"    Saved -> {out / 'index_meta.json'}", file=sys.stderr)


def load_index(
    index_dir: str | Path,
) -> tuple[faiss.Index, BM25Okapi, Optional[faiss.Index], list[dict], list[dict]]:
    """
    Load a previously saved index from disk.  Inverse of save_index().
    Raises FileNotFoundError if the required files are missing.
    """
    d = Path(index_dir)

    faiss_code = faiss.read_index(str(d / "faiss_code.index"))

    with open(d / "bm25_code.pkl", "rb") as f:
        bm25_code: BM25Okapi = pickle.load(f)

    faiss_style: Optional[faiss.Index] = None
    style_path = d / "faiss_style.index"
    if style_path.exists():
        faiss_style = faiss.read_index(str(style_path))

    with open(d / "index_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return faiss_code, bm25_code, faiss_style, meta["code"], meta.get("style", [])


# ─────────────────────────────────────────────────────────────────────────────
# Query / Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def query(
    question:    str,
    faiss_code:  faiss.Index,
    bm25_code:   BM25Okapi,
    code_syms:   list[dict],
    model_name:  str = DEFAULT_MODEL,
    top_k:       int = 5,
    faiss_style: Optional[faiss.Index] = None,
    style_syms:  list[dict] | None = None,
    model:       Optional[SentenceTransformer] = None,  # pass cached model to avoid reload
) -> list[dict]:
    """
    Retrieve the top_k most relevant symbols for a plain-English question.

    Strategy: Reciprocal Rank Fusion (RRF)
    ----------------------------------------
    Rather than normalising raw scores (which have different scales),
    RRF combines rankings:

        rrf_score(d) = Σ  1 / (k + rank_i(d))
                      i ∈ {faiss, bm25}

    where k=60 is the standard RRF constant.  This is robust and parameter-free.

    If style symbols are provided, style results are appended after code results
    (they're kept separate so Layer 5 can choose to cite them differently).

    Parameters
    ----------
    question   : raw user question string
    top_k      : number of code results to return
    """
    if model is None:
        model = SentenceTransformer(model_name)

    # ── FAISS retrieval ───────────────────────────────────────────────────────
    q_vec = model.encode([question], normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
    faiss_k = min(_FAISS_TOP_K, faiss_code.ntotal)
    _, faiss_ids = faiss_code.search(q_vec, faiss_k)  # type: ignore # shape (1, k) 
    faiss_ranking: list[int] = faiss_ids[0].tolist()   # list of symbol row indices

    # ── BM25 retrieval ────────────────────────────────────────────────────────
    tokens = _tokenize_query(question)
    bm25_scores: np.ndarray = bm25_code.get_scores(tokens)
    bm25_k = min(_BM25_TOP_K, len(code_syms))
    bm25_ranking: list[int] = cast(list[int], np.argsort(bm25_scores)[::-1][:bm25_k].tolist(),
)

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    RRF_K = 60
    scores: dict[int, float] = {}
    for rank, idx in enumerate(faiss_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(bm25_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Sort by fused score descending, take top_k
    sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in sorted_ids:
        sym = dict(code_syms[idx])
        sym["_score"] = round(scores[idx], 6)
        results.append(sym)

    # ── Append style results if available ────────────────────────────────────
    if faiss_style is not None and style_syms:
        style_k = min(3, faiss_style.ntotal)
        _, s_ids = faiss_style.search(q_vec, style_k) # type: ignore
        for idx in s_ids[0].tolist():
            sym = dict(style_syms[idx])
            sym["_score"] = None   # style scores not fused — kept separate
            sym["_tier"]  = "style"
            results.append(sym)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_results(results: list[dict]) -> None:
    """Pretty-print query results to stdout."""
    if not results:
        print("  (no results)")
        return
    for i, sym in enumerate(results, 1):
        tier   = sym.get("_tier", "code")
        score  = sym.get("_score")
        score_str = f"  score={score:.6f}" if score is not None else "  (style)"
        display = f"{sym.get('parent', '')}.{sym['name']}" if sym.get("parent") else sym["name"]
        print(f"\n  [{i}] {sym['kind']}  {display}{score_str}  [{tier}]")
        print(f"       {sym.get('file', '')}  L{sym.get('start_line', '?')}–{sym.get('end_line', '?')}")
        # Show first 3 lines of code as a preview
        code_preview = "\n".join(sym.get("code", "").splitlines()[:3])
        if code_preview:
            print("       " + code_preview.replace("\n", "\n       "))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Layer 3 — Build (and optionally query) the FAISS + BM25 search index",
        epilog="Run after: python call_graph.py symbols.json --save",
    )
    ap.add_argument(
        "symbols_json",
        help="Path to symbols.json produced by Layer 1",
    )
    ap.add_argument(
        "call_graph.json",
        help="Path to call_graph produced by Layer 2"
    )
    ap.add_argument(
        "--out", default="repo_index",
        help="Output directory for index files (default: ./repo_index)",
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"sentence-transformers model name (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--batch-size", type=int, default=64,
        help="Embedding batch size (default: 64; reduce if OOM)",
    )
    ap.add_argument(
        "--query", metavar="QUESTION",
        help="After building (or loading) the index, run this query and print results",
    )
    ap.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return for --query (default: 5)",
    )
    ap.add_argument(
        "--no-rebuild", action="store_true",
        help="Skip index rebuild; load existing index from --out and run --query only",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.no_rebuild:
        # ── Load existing index ───────────────────────────────────────────────
        print(f"\n  Loading index from {out_dir} ...", file=sys.stderr)
        faiss_code, bm25_code, faiss_style, code_syms, style_syms = load_index(out_dir)
        print(f"  Loaded: {len(code_syms)} code + {len(style_syms)} style symbols",
              file=sys.stderr)
    else:
        # ── Load symbols ──────────────────────────────────────────────────────
        sym_path = Path(args.symbols_json)
        if not sym_path.exists():
            print(f"Error: '{sym_path}' not found.", file=sys.stderr)
            sys.exit(1)

        with open(sym_path, "r", encoding="utf-8") as f:
            symbols: list[dict] = json.load(f)
        print(f"\n  Loaded {len(symbols)} symbols from {sym_path}", file=sys.stderr)

        # ── Build ─────────────────────────────────────────────────────────────
        faiss_code, bm25_code, faiss_style, code_syms, style_syms = build_index(
            symbols,
            model_name=args.model,
            batch_size=args.batch_size,
        )

        # ── Save ──────────────────────────────────────────────────────────────
        save_index(out_dir, faiss_code, bm25_code, faiss_style, code_syms, style_syms)
        print(f"\n  Index saved to {out_dir}/", file=sys.stderr)
        print(f"  Files written:", file=sys.stderr)
        for fname in ["faiss_code.index", "bm25_code.pkl", "index_meta.json"]:
            p = out_dir / fname
            size_kb = p.stat().st_size // 1024
            print(f"    {fname}  ({size_kb} KB)", file=sys.stderr)
        style_p = out_dir / "faiss_style.index"
        if style_p.exists():
            size_kb = style_p.stat().st_size // 1024
            print(f"    faiss_style.index  ({size_kb} KB)", file=sys.stderr)

    # ── Optional query ────────────────────────────────────────────────────────
    if args.query:
        print(f"\n  Query: \"{args.query}\"", file=sys.stderr)
        results = query(
            args.query,
            faiss_code, bm25_code, code_syms,
            model_name=args.model,
            top_k=args.top_k,
            faiss_style=faiss_style,
            style_syms=style_syms,
        )
        _print_results(results)
    elif args.no_rebuild:
        print("\n  Index loaded OK. Pass --query <question> to search.", file=sys.stderr)


if __name__ == "__main__":
    main()