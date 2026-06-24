"""
Layer 2 - Call Graph Builder
Reads the flat JSON produced by Layer 1 (units_to_record / --json output)
and produces a call graph: for every symbol, which other symbols does it call?

Output
------
call_graph.json     -> { "display_name": ["callee1","callee2", ...], ... }
symbols.json        -> the original flat list (pass-through, useful for Layer 3)

Usage
-----
    python call_graph.py symbols.json           #prints call graph to stdout
    python call_graph.py symbols.json --save    #writes call_graph.json + symbols.json
    python call_graph.py symbols.json --save --out ./repo_index
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Callable

from tree_sitter import Language, Parser, Node

# Reuse the important objects already defined in ast_parser
# (import them so we dont reinstantiate grammar bindings twice)
from ast_parser import (
    PY_LANGUAGE, JS_LANGUAGE, TS_LANGUAGE, TSX_LANGUAGE,
    C_LANGUAGE, CPP_LANGUAGE, JAVA_LANGUAGE,
    _text,
)


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1: Callee stoplist
#
# Problem: extracting the rightmost name from member_expression calls like
# res.json(), pool.query(), arr.map() gives us "json", "query", "map" —
# which are meaningless for a call graph. They flood the edges with noise
# and make the 718/738 unresolved callee problem.
#
# Solution: filter these out before they ever enter the graph.
# Organised by category so it's easy to extend.
# ─────────────────────────────────────────────────────────────────────────────

_CALLEE_STOPLIST: frozenset[str] = frozenset({
    # Express / HTTP response methods
    "json", "send", "status", "sendStatus", "redirect", "render",
    "end", "set", "header", "cookie", "clearCookie", "download",
    # Express app/router methods
    "get", "post", "put", "patch", "delete", "use", "listen",
    "all", "route", "param",
    # Node / CommonJS
    "require", "resolve", "exports",
    # Promise / async chain
    "then", "catch", "finally", "resolve", "reject", "all",
    "allSettled", "race", "any",
    # JS array / string / object builtins
    "map", "filter", "reduce", "forEach", "find", "findIndex",
    "some", "every", "includes", "indexOf", "slice", "splice",
    "push", "pop", "shift", "unshift", "join", "split", "flat",
    "flatMap", "sort", "reverse", "fill", "entries", "keys", "values",
    "toString", "toISOString", "toLocaleDateString", "toLocaleString",
    "toLowerCase", "toUpperCase", "trim", "trimStart", "trimEnd",
    "startsWith", "endsWith", "replace", "replaceAll", "match",
    "matchAll", "padStart", "padEnd", "repeat", "charAt", "charCodeAt",
    # Object builtins
    "assign", "create", "freeze", "keys", "values", "entries",
    "fromEntries", "hasOwnProperty",
    # Console / logging
    "log", "error", "warn", "info", "debug", "table",
    # DB / query (too generic — actual DB functions should be named specifically)
    "query", "execute", "run",
    # React hooks (these are framework calls, not your app logic)
    "useState", "useEffect", "useContext", "useRef", "useMemo",
    "useCallback", "useReducer", "useLayoutEffect",
    # React Router
    "useNavigate", "useOutletContext", "useParams", "useLocation",
    "useSearchParams",
    # Common React patterns that appear as calls but aren't real graph edges
    "navigate", "preventDefault", "stopPropagation",
    # localStorage / sessionStorage
    "getItem", "setItem", "removeItem",
    # Type coercion
    "String", "Number", "Boolean", "parseInt", "parseFloat",
    # Browser globals
    "alert", "confirm", "prompt", "setTimeout", "setInterval",
    "clearTimeout", "clearInterval", "fetch",
    # Misc patterns that appear as noise in your specific codebase
    "fmt",
    "next", "nextTick",           # Express middleware param
    "isArray", "apply", "call",   # JS internals  
    "defineProperty",              # Object internals
    "ok", "strictEqual", "equal", "assert",  # test assertions
    "callback", "done",            # generic callback params
    "parse", "format",             # too generic
    "escapeHtml", "encodeUrl", "encodeURI",  # utility noise
})


def _is_noisy_callee(name: str) -> bool:
    """
    Return True if this callee name should be dropped from the graph.

    Covers two cases:
      1. In the stoplist above (generic builtins / framework noise)
      2. Setter pattern: names starting with "set" + capital letter
         e.g. setLoading, setError, setForm, setSubmitting — these are
         React useState setters, not real function calls in your codebase.
    """
    if name in _CALLEE_STOPLIST:
        return True
    # React setter heuristic: setXxx where X is uppercase
    if len(name) > 3 and name.startswith("set") and name[3].isupper():
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Language -> (tree-sitter Language object, call-node type, callee-extraction fn)
# ─────────────────────────────────────────────────────────────────────────────

def _callee_python(call_node: Node, source: bytes) -> Optional[str]:
    """
    Python call node looks like:
        call
        ├── function:  identifier          →  "foo"
        └── function:  attribute           →  "self.foo"  or  "obj.method"
    We grab the rightmost name to keep things simple.
    """
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "identifier":
        return _text(fn, source)
    if fn.type == "attribute":
        attr = fn.child_by_field_name("attribute")
        if attr:
            return _text(attr, source)
    return None


def _callee_js(call_node: Node, source: bytes) -> Optional[str]:
    """
    JS/TS call_expression:
        call_expression
        ├── function:   identifier          ->  "foo"
        └── function:   member_expression  ->  "obj.method"
    """
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "identifier":
        return _text(fn, source)
    if fn.type == "member_expression":
        prop = fn.child_by_field_name("property")
        if prop:
            return _text(prop, source)
    return None


def _callee_c_cpp(call_node: Node, source: bytes) -> Optional[str]:
    """
    C/C++ call_expression:
        call_expression
        └── function:  identifier  |  field_expression  |  qualified_identifier
    """
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "identifier":
        return _text(fn, source)
    if fn.type in ("field_expression", "qualified_identifier"):
        for child in reversed(fn.children):
            if child.type in ("field_identifier", "identifier"):
                return _text(child, source)
    return None


def _callee_java(call_node: Node, source: bytes) -> Optional[str]:
    """
    Java method_invocation:
        method_invocation
        ├── object:  identifier | method_invocation (optional)
        └── name:    identifier
    """
    name = call_node.child_by_field_name("name")
    if name:
        return _text(name, source)
    return None


# Maps language name -> (tree-sitter Language, call node type, callee extractor)
_LANG_CFG: dict[str, tuple[Language, str, Callable]] = {
    "python":       (PY_LANGUAGE,   "call",                 _callee_python),
    "javascript":   (JS_LANGUAGE,   "call_expression",      _callee_js),
    "typescript":   (TS_LANGUAGE,   "call_expression",      _callee_js),
    "c":            (C_LANGUAGE,    "call_expression",       _callee_c_cpp),
    "cpp":          (CPP_LANGUAGE,  "call_expression",       _callee_c_cpp),
    "java":         (JAVA_LANGUAGE, "method_invocation",     _callee_java),
}

# Languages we intentionally skip (no meaningful call semantics)
_SKIP_LANGUAGES = {"html", "css"}


# ─────────────────────────────────────────────────────────────────────────────
# Core: walk AST and collect all call-node callees
# ─────────────────────────────────────────────────────────────────────────────

def _collect_calls(root: Node, source: bytes,
                   call_type: str,
                   extractor: Callable) -> list[str]:
    """
    DFS through the tree; collect every callee name found under a call node.
    Deduplicates within this unit but preserves order of first occurrence.
    Noisy/builtin callees are filtered via _is_noisy_callee().        # CHANGE 1
    """
    seen = set()
    results = []

    def walk(node: Node):
        if node.type == call_type:
            name = extractor(node, source)
            # CHANGE 1: apply stoplist filter here, before adding to results
            if name and name not in seen and not _is_noisy_callee(name):
                seen.add(name)
                results.append(name)
            # Still recurse — calls can be nested: foo(bar())
        for child in node.children:
            walk(child)

    walk(root)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2: Path normalisation helper
#
# Problem: symbols.json stores absolute Windows paths like
# C:\Users\aryam\.vscode\DBMS\MaterCare\matercare\backend\server.js
# These break on any other machine and make the index non-portable.
#
# Solution: when we know the repo root, store paths relative to it.
# We detect the repo root as the longest common prefix across all file paths.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_repo_root(symbols: list[dict]) -> Optional[Path]:
    """Return the longest common ancestor directory of all file paths."""
    paths = [Path(s["file"]) for s in symbols if s.get("file")]
    if not paths:
        return None
    try:
        # Path.parents gives all ancestors; common_path is the shared prefix
        common = Path(*[str(p.parent) for p in paths[:1]])  # start with first
        for p in paths[1:]:
            # Walk up until we find a common ancestor
            while True:
                try:
                    p.relative_to(common)
                    break
                except ValueError:
                    common = common.parent
                    if common == common.parent:  # reached filesystem root
                        return None
        return common
    except Exception:
        return None


def normalise_paths(symbols: list[dict], repo_root: Optional[Path] = None) -> list[dict]:
    """
    CHANGE 2: Replace absolute file paths with paths relative to repo_root.
    If repo_root is None, auto-detect it from the symbol list.
    Returns a new list — does not mutate the input.
    """
    if repo_root is None:
        repo_root = _compute_repo_root(symbols)
    if repo_root is None:
        return symbols  # can't normalise, return unchanged

    normalised = []
    for sym in symbols:
        s = dict(sym)
        if s.get("file"):
            try:
                s["file"] = str(Path(s["file"]).relative_to(repo_root))
            except ValueError:
                pass  # file outside detected root — leave as-is
        normalised.append(s)
    return normalised


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 3: searchable flag on symbols
#
# Problem: CSS/HTML symbols are useful for Layer 3 queries like "where is
# the login button styled?" but they shouldn't be treated the same as
# code symbols — they have no call semantics and embed differently.
#
# Solution: tag every symbol with searchable=True/False and a search_tier
# so Layer 3 can decide how to handle them without re-implementing this logic.
#
#   search_tier="code"   — functions, methods, classes (full RAG context)
#   search_tier="style"  — CSS rules (lower weight or separate index)
#   search_tier="skip"   — keyframes, media rules, script/style blocks
# ─────────────────────────────────────────────────────────────────────────────

_STYLE_KINDS   = {"rule"}
_SKIP_KINDS    = {"keyframes_rule", "media_rule", "script_block", "style_block"}
_CODE_KINDS    = {"function", "arrow_function", "method", "class",
                  "constructor", "route", "struct"}


def tag_symbols(symbols: list[dict]) -> list[dict]:
    """
    CHANGE 3: Add 'searchable' (bool) and 'search_tier' (str) to every symbol.
    Returns a new list — does not mutate input.
    """
    tagged = []
    for sym in symbols:
        s = dict(sym)
        kind = s.get("kind", "")
        lang = s.get("language", "")

        if kind in _SKIP_KINDS or (lang in _SKIP_LANGUAGES and kind not in _STYLE_KINDS):
            s["searchable"] = False
            s["search_tier"] = "skip"
        elif kind in _STYLE_KINDS:
            s["searchable"] = True
            s["search_tier"] = "style"
        else:
            s["searchable"] = True
            s["search_tier"] = "code"

        tagged.append(s)
    return tagged


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_call_graph(symbols: list[dict]) -> dict[str, list[str]]:
    """
    Given the flat symbol list from units_to_records(), return:

        {
            "display_name": ["callee_a", "callee_b"],
            ...
        }

    Keys are display_names ("ClassName.method" or "function_name").
    Values are callee names after stoplist filtering.                  # CHANGE 1
    Classes and CSS/HTML blocks are skipped — only callable units indexed.
    """
    # Build a parser cache so we don't recreate Parser objects per symbol
    _parsers: dict[str, Parser] = {}

    graph: dict[str, list[str]] = {}
    skipped = 0

    for sym in symbols:
        lang = sym.get("language", "")
        kind = sym.get("kind", "")
        code = sym.get("code", "")

        # Skip non-callable kinds and unsupported languages
        if lang in _SKIP_LANGUAGES:
            continue
        if kind in ("class", "struct", "script_block", "style_block",
                    "rule", "media_rule", "keyframes_rule"):
            continue
        if not code.strip():
            continue
        if lang not in _LANG_CFG:
            skipped += 1
            continue

        ts_lang, call_type, extractor = _LANG_CFG[lang]

        if lang not in _parsers:
            _parsers[lang] = Parser(ts_lang)
        parser = _parsers[lang]

        source = code.encode("utf-8")
        tree = parser.parse(source)
        callees = _collect_calls(tree.root_node, source, call_type, extractor)

        # Key = display_name: "ClassName.method" or "function"
        display = sym["parent"] + "." + sym["name"] if sym.get("parent") else sym["name"]

        # Exclude self-reference (recursive calls)
        callees = [c for c in callees if c != sym["name"]]

        graph[display] = callees

    if skipped:
        print(f" [info] Skipped {skipped} symbol(s) with unsupported language.",
              file=sys.stderr)

    return graph


def resolve_callees(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Optional second pass: try to resolve raw callee names to display_names
    that actually exist in the graph.

    Falls back to the raw name if no match is found (external/stdlib call).
    """
    # Build lookup: short name -> list of display_names ending with ".name" or == name
    short_to_display: dict[str, list[str]] = {}
    for display in graph:
        short = display.split(".")[-1]
        short_to_display.setdefault(short, []).append(display)

    resolved: dict[str, list[str]] = {}
    for caller, callees in graph.items():
        resolved_callees = []
        for raw in callees:
            candidates = short_to_display.get(raw, [])
            if len(candidates) == 1:
                resolved_callees.append(candidates[0])          # unambiguous
            elif len(candidates) > 1:
                # Ambiguous — prefer same class prefix if possible
                caller_prefix = caller.split(".")[0] if "." in caller else ""
                match = next(
                    (d for d in candidates if d.startswith(caller_prefix + ".")),
                    candidates[0],  # fallback: first match
                )
                resolved_callees.append(match)
            else:
                resolved_callees.append(raw)    # external / stdlib
        resolved[caller] = resolved_callees

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Layer 2 - Build a call graph from Layer 1 JSON output"
    )
    ap.add_argument("symbols_json",
                    help="Path to the JSON file produced by: python ast_parser.py <repo> --json")
    ap.add_argument("--save", action="store_true",
                    help="Save call_graph.json and symbols.json to --out directory")
    ap.add_argument("--out", default="repo_index",
                    help="Output directory (default: ./repo_index)")
    ap.add_argument("--resolve", action="store_true",
                    help="Run the optional callee-resolution pass")
    ap.add_argument("--repo-root",
                    help="Repo root path for relative path normalisation (auto-detected if omitted)")
    args = ap.parse_args()

    # Load symbols
    with open(args.symbols_json, "r", encoding="utf-8") as f:
        symbols: list[dict] = json.load(f)
    print(f"    Loaded {len(symbols)} symbols from {args.symbols_json}", file=sys.stderr)

    # CHANGE 2: normalise paths
    repo_root = Path(args.repo_root) if args.repo_root else None
    symbols = normalise_paths(symbols, repo_root)
    print("    Paths normalised to relative.", file=sys.stderr)

    # CHANGE 3: tag symbols with searchable + search_tier
    symbols = tag_symbols(symbols)
    print("    Symbols tagged with search_tier.", file=sys.stderr)

    # Build graph
    graph = build_call_graph(symbols)
    print(f"    Built call graph: {len(graph)} callable nodes", file=sys.stderr)

    if args.resolve:
        graph = resolve_callees(graph)
        print("    Callee resolution pass complete.", file=sys.stderr)

    # Stats
    total_edges = sum(len(v) for v in graph.values())
    non_empty = sum(1 for v in graph.values() if v)
    print(f"    Edges: {total_edges}   |   Nodes with outgoing calls: {non_empty}", file=sys.stderr)

    if args.save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        cg_path  = out_dir / "call_graph.json"
        sym_path = out_dir / "symbols.json"

        with open(cg_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
        with open(sym_path, "w", encoding="utf-8") as f:
            json.dump(symbols, f, indent=2)

        print(f"    Saved -> {cg_path}", file=sys.stderr)
        print(f"    Saved -> {sym_path}", file=sys.stderr)
    else:
        print(json.dumps(graph, indent=2))


if __name__ == "__main__":
    main()