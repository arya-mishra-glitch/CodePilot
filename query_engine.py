"""
Layer 5 — RAG Query Engine (LLM Integration)
=============================================
Takes a plain-English question, retrieves the most relevant code symbols using
the Layer 4 index, expands the context via the call graph, and sends everything
to an LLM (Gemini or Groq) to produce a grounded answer.

How it works
------------
  1. Load index   — call index_manager.load() to get all search objects
  2. Retrieve     — embedder.query() returns top-K symbols via FAISS + BM25 + RRF
  3. Expand       — follow one hop in the call graph for each retrieved symbol
                    (pull in callees + callers) so the LLM sees full context
  4. Build prompt — format the retrieved code into a structured prompt
  5. Call LLM     — send to Gemini (google-generativeai) or Groq
  6. Return answer

Usage
-----
    # Interactive mode (keeps index in memory, ask multiple questions):
    python query_engine.py ./repo_index

    # Single query:
    python query_engine.py ./repo_index --query "where is authentication handled?"

    # Use Groq instead of Gemini:
    python query_engine.py ./repo_index --provider groq

    # Show retrieved code chunks alongside the answer:
    python query_engine.py ./repo_index --show-context

Environment variables
---------------------
    GEMINI_API_KEY   — required when --provider gemini (default)
    GROQ_API_KEY     — required when --provider groq

Dependencies
------------
    pip install google-generativeai groq
    (index_manager.py, embedder.py, call_graph.py, ast_parser.py also required)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

# Load .env from the project root (silent if absent — real env vars take priority)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # won't overwrite vars already set in the shell
except ImportError:
    pass  # python-dotenv optional; fall back to os.environ

# ── Layer imports ─────────────────────────────────────────────────────────────
try:
    import index_manager
    import embedder as emb
except ImportError as e:
    print(
        f"[query_engine] ImportError: {e}\n"
        "Make sure index_manager.py and embedder.py are on sys.path.",
        file=sys.stderr,
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# How many symbols to retrieve from the index before call-graph expansion
_RETRIEVAL_TOP_K = 5

# Max call-graph hops to follow from each retrieved symbol.
# 1 = direct callees/callers only.  2 = one more level out.
# Keep this low (1-2): deeper hops bring in too much noise.
_GRAPH_HOPS = 1

# Max number of symbols to include in the final LLM prompt after expansion.
# More = more context but higher token cost and latency.
_MAX_CONTEXT_SYMBOLS = 12

# Max lines of code per symbol in the prompt.
# Prevents single huge functions from dominating.
_MAX_CODE_LINES = 40

# LLM generation parameters
_MAX_OUTPUT_TOKENS = 1024
_TEMPERATURE       = 0.2   # low temperature = factual, less creative

# Layer 5.5 — threshold-gated query expansion
# If the top RRF score after first-pass retrieval is below this, retrieval
# has likely failed and we trigger symbol-aware query expansion.
# RRF max per ranker ≈ 1/(60+1) ≈ 0.0164.  A top score < 0.025 means the
# best candidate appeared in only one ranker and not at rank 1.
_EXPANSION_THRESHOLD = 0.025




# ─────────────────────────────────────────────────────────────────────────────
# Layer 5.5 — Threshold-gated query expansion
#
# Two-level pipeline, both repo-aware (no hardcoded domain terms):
#
#   Level 1 — symbol-name expansion
#     Tokenize the query.  For each token that fuzzy-matches a symbol name,
#     split that symbol name (camelCase + snake_case) and append the parts.
#     "phase" → matches getPregnancyStatus → appends "get pregnancy status"
#
#   Level 2 — call-graph neighbour expansion
#     For every symbol surfaced in the first-pass retrieval, pull its
#     callers and callees from the call graph, split their names, and
#     append those tokens too.
#     getPregnancyStatus callee → pregnancyRows → appends "pregnancy rows"
#
# Both levels only fire when top_score < _EXPANSION_THRESHOLD, so fast
# queries (score already good) pay zero extra cost.
# ─────────────────────────────────────────────────────────────────────────────

def _name_to_tokens(name: str) -> list[str]:
    """
    Split a camelCase / PascalCase / snake_case identifier into lowercase tokens.

    Examples
    --------
    "getPregnancyStatus" → ["get", "pregnancy", "status"]
    "pregnancy_profile"  → ["pregnancy", "profile"]
    "BM25Okapi"          → ["b", "m25", "okapi"]  (best-effort)
    """
    # camelCase / PascalCase split
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # snake_case split
    s = s.replace("_", " ")
    return [t.lower() for t in s.split() if len(t) > 1]


def _expand_l1_symbol_names(question: str, code_syms: list[dict]) -> str:
    """
    Level 1: for each query token that appears inside a symbol name,
    split that symbol name and append the parts to the query.

    This is repo-aware — the expansion vocabulary comes entirely from the
    symbol names extracted by Layer 1, so it works on any codebase.
    """
    query_tokens = {t for t in re.split(r"\W+", question.lower()) if len(t) > 3}

    appended: set[str] = set()
    for sym in code_syms:
        name = sym.get("name", "")
        if not name:
            continue
        name_lower = name.lower()
        if any(tok in name_lower for tok in query_tokens):
            for part in _name_to_tokens(name):
                appended.add(part)

    if not appended:
        return question

    extra = " ".join(sorted(appended))
    print(f"  [expand-L1] appending symbol tokens: {extra}", file=sys.stderr)
    return f"{question} {extra}"


def _expand_l2_call_graph(
    question:      str,
    first_pass:    list[dict],
    call_graph:    dict[str, list[str]],
    callers_map:   dict[str, list[str]],
) -> str:
    """
    Level 2: take the symbols returned by the first-pass retrieval,
    walk one hop in the call graph (both directions), split neighbour
    names into tokens, and append them.

    This anchors the re-query in the actual graph neighbourhood of whatever
    the retriever found first — even if that first pass was imperfect.
    """
    appended: set[str] = set()
    for sym in first_pass:
        name   = sym.get("name", "")
        parent = sym.get("parent", "")
        display = f"{parent}.{name}" if parent else name

        neighbours = call_graph.get(display, []) + callers_map.get(display, [])
        for nb_name in neighbours:
            # nb_name may be "Class.method" or bare "functionName"
            bare = nb_name.split(".")[-1]
            for part in _name_to_tokens(bare):
                appended.add(part)

    if not appended:
        return question

    extra = " ".join(sorted(appended))
    print(f"  [expand-L2] appending call-graph tokens: {extra}", file=sys.stderr)
    return f"{question} {extra}"


# ─────────────────────────────────────────────────────────────────────────────
# Call-graph expansion
# ─────────────────────────────────────────────────────────────────────────────

def _expand_with_call_graph(
    retrieved: list[dict],
    call_graph: dict[str, list[str]],
    all_syms_by_name: dict[str, dict],
    hops: int = _GRAPH_HOPS,
    max_total: int = _MAX_CONTEXT_SYMBOLS,
) -> list[dict]:
    """
    Expand the retrieved symbol set by following call-graph edges.

    For each retrieved symbol we pull in:
      - its direct callees (functions it calls)
      - its direct callers (functions that call it)

    This gives the LLM enough context to understand not just the function
    itself but how it fits into the surrounding code.

    Parameters
    ----------
    retrieved         : symbols returned by embedder.query()
    call_graph        : { display_name: [callee_display_name, ...] }
    all_syms_by_name  : { display_name: symbol_dict } lookup table
    hops              : how many graph hops to follow
    max_total         : cap on total symbols after expansion

    Returns
    -------
    Deduplicated list of symbols, retrieved ones first, neighbours after.
    """
    # Build a reverse map: callee → set of callers
    callers_map: dict[str, set[str]] = {}
    for caller, callees in call_graph.items():
        for callee in callees:
            callers_map.setdefault(callee, set()).add(caller)

    seen: set[str] = set()
    result: list[dict] = []

    def _add(sym: dict) -> None:
        name = sym.get("name", "")
        parent = sym.get("parent", "")
        display = f"{parent}.{name}" if parent else name
        if display not in seen:
            seen.add(display)
            result.append(sym)

    # Seed with retrieved symbols (highest priority)
    for sym in retrieved:
        _add(sym)

    # BFS expansion
    frontier = list(retrieved)
    for _ in range(hops):
        next_frontier: list[dict] = []
        if len(result) >= max_total:
            break
        for sym in frontier:
            name = sym.get("name", "")
            parent = sym.get("parent", "")
            display = f"{parent}.{name}" if parent else name

            # Callees
            for callee_name in call_graph.get(display, []):
                if callee_name in all_syms_by_name and callee_name not in seen:
                    nbr = all_syms_by_name[callee_name]
                    _add(nbr)
                    next_frontier.append(nbr)
                    if len(result) >= max_total:
                        break

            # Callers
            for caller_name in callers_map.get(display, set()):
                if caller_name in all_syms_by_name and caller_name not in seen:
                    nbr = all_syms_by_name[caller_name]
                    _add(nbr)
                    next_frontier.append(nbr)
                    if len(result) >= max_total:
                        break

        frontier = next_frontier

    return result[:max_total]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def _format_symbol(sym: dict, max_lines: int = _MAX_CODE_LINES) -> str:
    """
    Format a single symbol as a readable code block for the prompt.

    We include:
      - a header line with kind, name, file, and line numbers
      - the first `max_lines` lines of source code
    """
    kind   = sym.get("kind", "symbol")
    name   = sym.get("name", "unknown")
    parent = sym.get("parent", "")
    file   = sym.get("file", "")
    start  = sym.get("start_line", "?")
    end    = sym.get("end_line",   "?")
    display = f"{parent}.{name}" if parent else name

    code = sym.get("code", "")
    code_lines = code.splitlines()
    if len(code_lines) > max_lines:
        code_lines = code_lines[:max_lines] + [f"... [{len(code_lines) - max_lines} more lines]"]
    code_block = "\n".join(code_lines)

    return (
        f"### {kind}: {display}\n"
        f"# File: {file}  (lines {start}–{end})\n"
        f"{code_block}"
    )


def _build_prompt(question: str, context_syms: list[dict]) -> str:
    """
    Build the full prompt sent to the LLM.

    Structure:
      - System instruction (persona + constraints)
      - Retrieved code context
      - User question
      - Answer format instruction
    """
    code_context = "\n\n".join(_format_symbol(s) for s in context_syms)

    return textwrap.dedent(f"""
        You are a senior software engineer helping a developer understand a codebase.
        You have been given the most relevant code snippets retrieved from the codebase.
        Answer the question using ONLY the provided code. Do not hallucinate functions
        or behaviour that are not present in the snippets below.

        If the answer cannot be determined from the provided code, say so clearly.
        Keep your answer concise: lead with the direct answer, then explain with
        references to specific function names and file paths.

        ── RETRIEVED CODE ──────────────────────────────────────────────────────
        {code_context}
        ────────────────────────────────────────────────────────────────────────

        QUESTION: {question}

        ANSWER:
    """).strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM providers
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """Send prompt to Gemini Flash and return the response text."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def _call_groq(prompt: str) -> str:
    """Send prompt to Groq (llama-3.1-8b-instant) and return the response text."""
    try:
        from groq import Groq
    except ImportError:
        print(
            "[query_engine] groq package not installed.\n"
            "Run:  pip install groq",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "[query_engine] GROQ_API_KEY environment variable not set.\n"
            "Get a free key at: https://console.groq.com/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_MAX_OUTPUT_TOKENS,
        temperature=_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _call_gemini_with_groq_fallback(prompt: str) -> str:
    """Try Gemini first; if it fails (busy, quota, error) fall back to Groq."""
    try:
        return _call_gemini(prompt)
    except Exception as e:
        print(f"  [query_engine] Gemini failed ({e}), falling back to Groq ...",
              file=sys.stderr)
        return _call_groq(prompt)


_PROVIDERS = {
    "gemini": _call_gemini,
    "groq":   _call_groq,
    "auto":   _call_gemini_with_groq_fallback,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main query function — the public API Layer 5 exposes
# ─────────────────────────────────────────────────────────────────────────────

class QueryEngine:
    """
    Stateful query engine.  Load once, query many times.

    Example
    -------
    >>> engine = QueryEngine("./repo_index", provider="gemini")
    >>> answer, context = engine.query("where is auth handled?")
    >>> print(answer)
    """

    def __init__(
        self,
        index_dir: str | Path,
        provider:  str = "gemini",
        model_name: str = emb.DEFAULT_MODEL,
        top_k:     int = _RETRIEVAL_TOP_K,
    ) -> None:
        """
        Parameters
        ----------
        index_dir  : path to the index directory built by index_manager.py
        provider   : "gemini" or "groq"
        model_name : sentence-transformers model (must match what was used at build time)
        top_k      : number of symbols to retrieve before call-graph expansion
        """
        if provider not in _PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")

        self.provider   = provider
        self.model_name = model_name
        self.top_k      = top_k
        self._llm_fn    = _PROVIDERS[provider]

        print(f"\n  Loading index from '{index_dir}' ...", file=sys.stderr)
        (
            self.faiss_code,
            self.bm25_code,
            self.faiss_style,
            self.code_syms,
            self.style_syms,
            self.call_graph,
        ) = index_manager.load(index_dir)

        # Cache the embedding model so it isn't reloaded on every query
        print(f"  Loading embedding model '{model_name}' ...", file=sys.stderr)
        self._embed_model = emb.SentenceTransformer(model_name)

        # Build a name → symbol lookup for call-graph expansion
        self._sym_by_name: dict[str, dict] = {}
        for sym in self.code_syms:
            name   = sym.get("name", "")
            parent = sym.get("parent", "")
            display = f"{parent}.{name}" if parent else name
            self._sym_by_name[display] = sym
            self._sym_by_name[name]    = sym   # also index by bare name

        # Reverse call-graph (callee → [callers]) — used by L2 expansion
        self._callers_map: dict[str, list[str]] = {}
        for caller, callees in self.call_graph.items():
            for callee in callees:
                self._callers_map.setdefault(callee, []).append(caller)

        print(
            f"  Ready — {len(self.code_syms)} code symbols, "
            f"{len(self.call_graph)} call graph nodes, "
            f"provider={provider}\n",
            file=sys.stderr,
        )

    def query(
        self,
        question: str,
        show_context: bool = False,
    ) -> tuple[str, list[dict]]:
        """
        Answer a plain-English question about the codebase.

        Parameters
        ----------
        question     : user's question
        show_context : if True, print retrieved code to stderr before answering

        Returns
        -------
        (answer_text, context_symbols)
          answer_text     : the LLM's answer string
          context_symbols : the symbols that were included in the prompt
        """
        # ── Step 1: First-pass retrieval ─────────────────────────────────────
        retrieved = emb.query(
            question,
            self.faiss_code,
            self.bm25_code,
            self.code_syms,
            model_name=self.model_name,
            top_k=self.top_k,
            faiss_style=self.faiss_style,
            style_syms=self.style_syms,
            model=self._embed_model,          # ← use cached model
        )

        # ── Step 1.5: Layer 5.5 — threshold-gated query expansion ────────────
        top_score = retrieved[0].get("_score", 0.0) if retrieved else 0.0
        if top_score < _EXPANSION_THRESHOLD:
            print(
                f"  [expand] top score {top_score:.4f} < {_EXPANSION_THRESHOLD} "
                f"— triggering query expansion",
                file=sys.stderr,
            )
            # L1: symbol-name expansion (repo vocab, no API cost)
            expanded = _expand_l1_symbol_names(question, self.code_syms)
            # L2: call-graph neighbour expansion on top of L1
            expanded = _expand_l2_call_graph(
                expanded, retrieved, self.call_graph, self._callers_map
            )
            # Re-retrieve only if we actually added something
            if expanded != question:
                retrieved = emb.query(
                    expanded,
                    self.faiss_code,
                    self.bm25_code,
                    self.code_syms,
                    model_name=self.model_name,
                    top_k=self.top_k,
                    faiss_style=self.faiss_style,
                    style_syms=self.style_syms,
                    model=self._embed_model,
                )
                new_top = retrieved[0].get("_score", 0.0) if retrieved else 0.0
                print(
                    f"  [expand] re-retrieval top score: {new_top:.4f}",
                    file=sys.stderr,
                )

        # ── Step 2: Expand via call graph ─────────────────────────────────────
        context_syms = _expand_with_call_graph(
            retrieved,
            self.call_graph,
            self._sym_by_name,
        )

        # ── Step 3: (Optional) print context ─────────────────────────────────
        if show_context:
            print("\n  ── Retrieved context ──────────────────────────────────",
                  file=sys.stderr)
            for i, sym in enumerate(context_syms, 1):
                tag = " [expanded]" if sym not in retrieved else ""
                name   = sym.get("name", "?")
                parent = sym.get("parent", "")
                display = f"{parent}.{name}" if parent else name
                score  = sym.get("_score")
                score_str = f"  score={score:.4f}" if score else ""
                print(f"  [{i}] {sym.get('kind','?'):15s} {display}{score_str}{tag}",
                      file=sys.stderr)
                print(f"       {sym.get('file','?')}  L{sym.get('start_line','?')}-{sym.get('end_line','?')}",
                      file=sys.stderr)
            print("  ─────────────────────────────────────────────────────\n",
                  file=sys.stderr)

        # ── Step 4: Build prompt ──────────────────────────────────────────────
        prompt = _build_prompt(question, context_syms)

        # ── Step 5: Call LLM ──────────────────────────────────────────────────
        answer = self._llm_fn(prompt)

        return answer, context_syms


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Layer 5 — RAG Query Engine: ask questions about your codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode:
  python query_engine.py ./repo_index

  # Single query:
  python query_engine.py ./repo_index --query "where is authentication handled?"

  # Use Groq, show retrieved context:
  python query_engine.py ./repo_index --provider groq --show-context

Environment variables:
  GEMINI_API_KEY   (required for --provider gemini)
  GROQ_API_KEY     (required for --provider groq)
        """,
    )
    ap.add_argument("index_dir", help="Path to the index directory (built by index_manager.py)")
    ap.add_argument("--query",    metavar="QUESTION",
                    help="Ask a single question and exit")
    ap.add_argument("--provider", choices=["gemini", "groq", "auto"], default="auto",
                    help="LLM provider: 'auto' tries Gemini then falls back to Groq (default: auto)")
    ap.add_argument("--model",    default=emb.DEFAULT_MODEL,
                    help=f"sentence-transformers model (default: {emb.DEFAULT_MODEL})")
    ap.add_argument("--top-k",   type=int, default=_RETRIEVAL_TOP_K,
                    help=f"Symbols to retrieve before expansion (default: {_RETRIEVAL_TOP_K})")
    ap.add_argument("--show-context", action="store_true",
                    help="Print retrieved code chunks to stderr alongside the answer")
    args = ap.parse_args()

    engine = QueryEngine(
        index_dir=args.index_dir,
        provider=args.provider,
        model_name=args.model,
        top_k=args.top_k,
    )

    if args.query:
        # ── Single query mode ─────────────────────────────────────────────────
        answer, _ = engine.query(args.query, show_context=args.show_context)
        print(f"\n{'─'*60}")
        print(f"Q: {args.query}")
        print(f"{'─'*60}")
        print(answer)
        print(f"{'─'*60}\n")
    else:
        # ── Interactive REPL ──────────────────────────────────────────────────
        print("  Code Intelligence — ask anything about your codebase.")
        print("  Type 'exit' or Ctrl-C to quit.\n")
        while True:
            try:
                question = input("  > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye.")
                break

            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("  Goodbye.")
                break

            answer, _ = engine.query(question, show_context=args.show_context)
            print(f"\n{answer}\n")


if __name__ == "__main__":
    main()