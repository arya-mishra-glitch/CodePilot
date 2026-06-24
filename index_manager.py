"""
Layer 4 — Index Manager (Portable Index)
=========================================
Orchestrates the full build pipeline (Layers 1-3) in one command and provides
a clean load interface that Layers 5+ import.

What this file does
-------------------
  1. build  — runs ast_parser → call_graph → embedder end-to-end and writes
              the index bundle to an output directory.
  2. load   — single call that returns everything Layer 5 needs:
              (faiss_code, bm25_code, faiss_style, code_syms, style_syms, call_graph)
  3. info   — prints a human-readable summary of an existing index.
  4. verify — sanity-checks that all expected files are present and readable.

Output layout (default: ./repo_index/)
----------------------------------------
    faiss_code.index    ← FAISS dense index for code symbols
    faiss_style.index   ← FAISS dense index for CSS/style symbols (optional)
    bm25_code.pkl       ← BM25 keyword index for code symbols
    index_meta.json     ← { "code": [...], "style": [...] }  (row → symbol map)
    call_graph.json     ← { display_name: [callee, ...] }
    symbols.json        ← flat symbol list (pass-through from Layer 2)
    index_info.json     ← build metadata (timestamp, counts, model name)

Usage
-----
    # Full build from a repo directory:
    python index_manager.py build /path/to/repo

    # Custom output directory and embedding model:
    python index_manager.py build /path/to/repo --out repo_index --model all-mpnet-base-v2

    # Print summary of an existing index:
    python index_manager.py info ./repo_index

    # Verify index integrity:
    python index_manager.py verify ./repo_index

Dependencies
------------
    pip install sentence-transformers faiss-cpu rank_bm25
    (ast_parser.py, call_graph.py, embedder.py must be on the Python path)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
from rank_bm25 import BM25Okapi

# ── Import sibling layers ─────────────────────────────────────────────────────
# These are the only cross-layer imports.  If you restructure into a package,
# replace these with relative imports (from .embedder import ...).
try:
    import ast_parser   # noqa: F401  — imported for side-effect: grammar init
    import call_graph as cg
    import embedder as emb
except ImportError as e:
    print(
        f"[index_manager] ImportError: {e}\n"
        "Make sure ast_parser.py, call_graph.py, and embedder.py are in the "
        "same directory (or on sys.path).",
        file=sys.stderr,
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Required files — used by verify() and load()
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_FILES = [
    "faiss_code.index",
    "bm25_code.pkl",
    "index_meta.json",
    "call_graph.json",
    "symbols.json",
]

_OPTIONAL_FILES = [
    "faiss_style.index",
    "index_info.json",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API — what Layer 5 imports
# ─────────────────────────────────────────────────────────────────────────────

def load(
    index_dir: str | Path,
) -> tuple[
    faiss.Index,
    BM25Okapi,
    Optional[faiss.Index],
    list[dict],
    list[dict],
    dict[str, list[str]],
]:
    """
    Load a complete index bundle from disk.

    This is the single function Layer 5 calls.  It returns everything needed
    to answer a user query:

    Returns
    -------
    faiss_code  : FAISS index for code symbols
    bm25_code   : BM25 keyword index for code symbols
    faiss_style : FAISS index for style symbols, or None
    code_syms   : list of code symbol dicts, ordered by FAISS row
    style_syms  : list of style symbol dicts, ordered by FAISS row
    call_graph  : { display_name: [callee_display_name, ...] }

    Raises
    ------
    FileNotFoundError if any required file is missing.
    """
    d = Path(index_dir)
    missing = [f for f in _REQUIRED_FILES if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Index at '{d}' is incomplete. Missing: {missing}\n"
            f"Run:  python index_manager.py build <repo_path> --out {d}"
        )

    # Load FAISS + BM25 + symbol map (delegated to embedder)
    faiss_code, bm25_code, faiss_style, code_syms, style_syms = emb.load_index(d)

    # Load call graph
    with open(d / "call_graph.json", "r", encoding="utf-8") as f:
        call_graph_data: dict[str, list[str]] = json.load(f)

    return faiss_code, bm25_code, faiss_style, code_syms, style_syms, call_graph_data


def verify(index_dir: str | Path) -> bool:
    """
    Check that all required index files exist and can be opened.

    Returns True if the index is healthy, False (+ printed errors) otherwise.
    Does not raise — safe to call in CI / startup checks.
    """
    d = Path(index_dir)
    ok = True

    print(f"\n  Verifying index at '{d}' ...")

    # ── File presence ─────────────────────────────────────────────────────────
    for fname in _REQUIRED_FILES:
        p = d / fname
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"    ✓  {fname:30s}  ({size_kb} KB)")
        else:
            print(f"    ✗  {fname:30s}  MISSING")
            ok = False

    for fname in _OPTIONAL_FILES:
        p = d / fname
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"    ~  {fname:30s}  ({size_kb} KB)  [optional]")

    if not ok:
        print("\n  ✗  Index is incomplete.  Run: python index_manager.py build <repo>")
        return False

    # ── Integrity checks ──────────────────────────────────────────────────────
    print("\n  Running integrity checks ...")
    try:
        faiss_code, bm25_code, faiss_style, code_syms, style_syms, cg_data = load(d)
        print(f"    ✓  FAISS code index loaded    ({faiss_code.ntotal} vectors)")
        print(f"    ✓  BM25 code index loaded     ({len(code_syms)} documents)")
        if faiss_style:
            print(f"    ✓  FAISS style index loaded   ({faiss_style.ntotal} vectors)")
        print(f"    ✓  call_graph.json loaded     ({len(cg_data)} nodes)")
        print(f"    ✓  index_meta.json loaded     ({len(code_syms)} code + {len(style_syms)} style symbols)")

        # Row count must match between FAISS and the symbol list
        assert faiss_code.ntotal == len(code_syms), (
            f"FAISS has {faiss_code.ntotal} vectors but index_meta has {len(code_syms)} code symbols. "
            "Index may be corrupted — rebuild."
        )
        print("    ✓  FAISS row count matches index_meta row count")

    except Exception as exc:
        print(f"    ✗  Integrity check failed: {exc}")
        return False

    print("\n  ✓  Index is healthy.\n")
    return True


def info(index_dir: str | Path) -> None:
    """
    Print a human-readable summary of an existing index.
    Does not load the full FAISS index — reads metadata files only (fast).
    """
    d = Path(index_dir)

    print(f"\n  Index at: {d.resolve()}\n")

    # ── index_info.json (build metadata) ─────────────────────────────────────
    info_path = d / "index_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            build_info: dict = json.load(f)
        print(f"  Built at   : {build_info.get('built_at', 'unknown')}")
        print(f"  Model      : {build_info.get('model', 'unknown')}")
        print(f"  Build time : {build_info.get('build_seconds', '?')}s")
    else:
        print("  (index_info.json not found — older index)")

    # ── index_meta.json ───────────────────────────────────────────────────────
    meta_path = d / "index_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta: dict = json.load(f)
        code_syms: list  = meta.get("code", [])
        style_syms: list = meta.get("style", [])
        print(f"\n  Symbols    : {len(code_syms)} code  +  {len(style_syms)} style")

        # Break down by language
        lang_counts: dict[str, int] = {}
        for s in code_syms:
            lang = s.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        if lang_counts:
            print("  By language:")
            for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
                print(f"      {lang:12s}  {count}")

        # Break down by kind
        kind_counts: dict[str, int] = {}
        for s in code_syms:
            kind = s.get("kind", "unknown")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if kind_counts:
            print("  By kind:")
            for kind, count in sorted(kind_counts.items(), key=lambda x: -x[1]):
                print(f"      {kind:20s}  {count}")

    # ── call_graph.json ───────────────────────────────────────────────────────
    cg_path = d / "call_graph.json"
    if cg_path.exists():
        with open(cg_path, "r", encoding="utf-8") as f:
            cg_data: dict = json.load(f)
        total_edges = sum(len(v) for v in cg_data.values())
        non_empty   = sum(1 for v in cg_data.values() if v)
        print(f"\n  Call graph : {len(cg_data)} nodes  |  {total_edges} edges  "
              f"|  {non_empty} nodes with outgoing calls")

    # ── File sizes ────────────────────────────────────────────────────────────
    print("\n  Files:")
    for fname in _REQUIRED_FILES + _OPTIONAL_FILES:
        p = d / fname
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"      {fname:30s}  {size_kb:>6} KB")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Build pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build(
    repo_path: str | Path,
    out_dir:   str | Path = "repo_index",
    model:     str = emb.DEFAULT_MODEL,
    batch_size: int = 64,
    resolve_callees: bool = False,
    force: bool = False,
) -> Path:
    """
    Run the full Layers 1-3 pipeline and write the index bundle to out_dir.

    Steps
    -----
    1. ast_parser   → symbols.json
    2. call_graph   → call_graph.json  (+ tagged symbols.json)
    3. embedder     → faiss_code.index, bm25_code.pkl, index_meta.json
    4. write        → index_info.json  (build metadata)

    Parameters
    ----------
    repo_path       : root directory of the codebase to index
    out_dir         : where to write all index files
    model           : sentence-transformers model name
    batch_size      : embedding batch size
    resolve_callees : run the optional callee-resolution pass in Layer 2
    force           : rebuild even if the index already exists

    Returns
    -------
    Path to the output directory.
    """
    repo  = Path(repo_path).resolve()
    out   = Path(out_dir)
    start = time.time()

    if not repo.exists():
        print(f"[index_manager] Error: repo path '{repo}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # ── Skip if already built ─────────────────────────────────────────────────
    if not force:
        missing = [f for f in _REQUIRED_FILES if not (out / f).exists()]
        if not missing:
            print(f"\n  Index already exists at '{out}'. Use --force to rebuild.", file=sys.stderr)
            return out

    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Building index for: {repo}", file=sys.stderr)
    print(f"  Output directory  : {out}", file=sys.stderr)
    print(f"  Model             : {model}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: AST parsing  (Layer 1)
    # ─────────────────────────────────────────────────────────────────────────
    print("  [1/3] Parsing codebase (Layer 1 — ast_parser) ...", file=sys.stderr)
    t0 = time.time()

    # ast_parser's main() is a CLI; invoke it as a subprocess so we don't have
    # to duplicate its argument-handling logic here.
    sym_path = out / "symbols.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "ast_parser",
            str(repo),
            "--json",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # ast_parser may not support -m; try direct script invocation
        result = subprocess.run(
            [sys.executable, "ast_parser.py", str(repo), "--json"],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        print(f"  [1/3] ast_parser failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    symbols: list[dict] = json.loads(result.stdout)

    with open(sym_path, "w", encoding="utf-8") as f:
        json.dump(symbols, f, indent=2)
    print(f"  [1/3] Done — {len(symbols)} symbols extracted  ({time.time()-t0:.1f}s)\n",
          file=sys.stderr)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Call graph  (Layer 2)
    # ─────────────────────────────────────────────────────────────────────────
    print("  [2/3] Building call graph (Layer 2 — call_graph) ...", file=sys.stderr)
    t0 = time.time()

    # Normalise paths and tag symbols (search_tier etc.)
    symbols = cg.normalise_paths(symbols, repo)
    symbols = cg.tag_symbols(symbols)
    graph   = cg.build_call_graph(symbols)

    if resolve_callees:
        graph = cg.resolve_callees(graph)
        print("       Callee resolution pass complete.", file=sys.stderr)

    total_edges = sum(len(v) for v in graph.values())
    print(f"  [2/3] Done — {len(graph)} nodes, {total_edges} edges  ({time.time()-t0:.1f}s)\n",
          file=sys.stderr)

    # Write tagged symbols back (overwrite with search_tier annotations)
    with open(sym_path, "w", encoding="utf-8") as f:
        json.dump(symbols, f, indent=2)

    # Write call graph
    cg_path = out / "call_graph.json"
    with open(cg_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"       Saved -> {cg_path}", file=sys.stderr)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Embeddings + FAISS + BM25  (Layer 3)
    # ─────────────────────────────────────────────────────────────────────────
    print("  [3/3] Building search index (Layer 3 — embedder) ...", file=sys.stderr)
    t0 = time.time()

    faiss_code, bm25_code, faiss_style, code_syms, style_syms = emb.build_index(
        symbols,
        graph,
        model_name=model,
        batch_size=batch_size,
    )
    emb.save_index(out, faiss_code, bm25_code, faiss_style, code_syms, style_syms)
    print(f"  [3/3] Done  ({time.time()-t0:.1f}s)\n", file=sys.stderr)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Write build metadata
    # ─────────────────────────────────────────────────────────────────────────
    build_info = {
        "built_at":      datetime.now(timezone.utc).isoformat(),
        "repo":          str(repo),
        "model":         model,
        "build_seconds": round(time.time() - start, 1),
        "code_symbols":  len(code_syms),
        "style_symbols": len(style_syms),
        "call_graph_nodes": len(graph),
        "call_graph_edges": total_edges,
    }
    with open(out / "index_info.json", "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = round(time.time() - start, 1)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  ✓  Index built in {elapsed}s", file=sys.stderr)
    print(f"  ✓  {len(code_syms)} code symbols  +  {len(style_syms)} style symbols", file=sys.stderr)
    print(f"  ✓  {len(graph)} call graph nodes  /  {total_edges} edges", file=sys.stderr)
    print(f"  ✓  Output: {out.resolve()}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Layer 4 — Index Manager: build, verify, and inspect the code index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python index_manager.py build /path/to/repo
    python index_manager.py build /path/to/repo --out repo_index --model all-mpnet-base-v2
    python index_manager.py info  ./repo_index
    python index_manager.py verify ./repo_index
    """,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # ── build ─────────────────────────────────────────────────────────────────
    bp = sub.add_parser("build", help="Run full pipeline and build the index")
    bp.add_argument("repo", help="Root directory of the codebase to index")
    bp.add_argument("--out", default="repo_index",
                    help="Output directory for index files (default: ./repo_index)")
    bp.add_argument("--model", default=emb.DEFAULT_MODEL,
                    help=f"sentence-transformers model (default: {emb.DEFAULT_MODEL})")
    bp.add_argument("--batch-size", type=int, default=64,
                    help="Embedding batch size (default: 64; reduce if OOM on GPU)")
    bp.add_argument("--resolve", action="store_true",
                    help="Run callee-resolution pass in Layer 2 (slower, more complete graph)")
    bp.add_argument("--force", action="store_true",
                    help="Rebuild even if the index already exists")

    # ── info ──────────────────────────────────────────────────────────────────
    ip = sub.add_parser("info", help="Print a summary of an existing index")
    ip.add_argument("index_dir", help="Path to the index directory")

    # ── verify ────────────────────────────────────────────────────────────────
    vp = sub.add_parser("verify", help="Check index integrity")
    vp.add_argument("index_dir", help="Path to the index directory")

    args = ap.parse_args()

    if args.command == "build":
        build(
            repo_path=args.repo,
            out_dir=args.out,
            model=args.model,
            batch_size=args.batch_size,
            resolve_callees=args.resolve,
            force=args.force,
        )

    elif args.command == "info":
        info(args.index_dir)

    elif args.command == "verify":
        ok = verify(args.index_dir)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()