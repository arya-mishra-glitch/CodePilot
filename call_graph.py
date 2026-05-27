"""
Layer 2 - Call Graph 
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

#Reuse the important objects already defined in ast_parser
# (import them so we dont reinstantiate grammar bindings twice)

from ast_parser import (
    PY_LANGUAGE, JS_LANGUAGE, TS_LANGUAGE, TSX_LANGUAGE,
    C_LANGUAGE, CPP_LANGUAGE, JAVA_LANGUAGE,
    _text,
)

#------------------------------------------------------------------------------------
#Language -> (tree-sitter Language object, call - node type, callee-extraction fn)

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

def _callee_js(call_node: Node, source: bytes ) -> Optional[str]:
    """
    JS/TS call_expression:
        call_expression
        ├── function:   identifier          ->  "foo"
        └── function:   memeber_expression  ->  "obj.method"
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
    if fn.type == "member_expression":
        prop = fn.child_by_field_name("property")
        if prop:
            return _text(prop, source)
    return None


def _callee_java(call_node: Node, source: bytes) -> Optional[str]:
    """
    Java method_invocation:
        method_invocation
        ├── objects:    identifier | method_invocation (optional)
        └──name:        identifier
    """
    name = call_node.child_by_field_name("name")
    if name:
        return _text(name, source)
    return None



#Maps language name -> (tree-sitter Language, call node type, callee extractor)
_LANG_CFG: dict[str, tuple[Language, str, Callable]] = {
    "python":       (PY_LANGUAGE,   "call",                 _callee_python),
    "javascript":   (JS_LANGUAGE,   "call_expression",      _callee_js),
    "typescript":   (TS_LANGUAGE,   "call_expression",      _callee_js),
    "c":            (C_LANGUAGE,    "call_expression",      _callee_c_cpp),
    "cpp":          (CPP_LANGUAGE,  "call_expression",      _callee_c_cpp),
    "java":         (JAVA_LANGUAGE, "method_invocation",    _callee_java)
}

#Languages intentionally skip (no meaningful call semantics)
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
    """

    seen = set()
    results = []

    def walk(node: Node):
        if node.type == call_type:
            name = extractor(node, source)
            if name and name not in seen:
                seen.add(name)
                results.append(name)
            #Still recurse - calls can be nested: foo(bar())
        for child in node.children:
            walk(child)

    walk(root)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_call_graph(symbols: list[dict]) -> dict[str, list[str]]:
    """
    Given the flat symbol list from units_to_records(), return:

        {
            "display_name":["callee_a", "callee_b"],
            ...
        }

    Keys are display_names  ("ClassName.method" or "function_name").
    Values are raw calee names (not resolved to display_names yet).
    Classes and CSS/HTML blocks are skipped - only callable units are indexed.
    """

    #Build a parser cache so we don't recreate Parser objects per symbol
    _parsers: dict[str, Parser] = {}

    graph: dict[str, list[str]] = {}
    skipped=0

    for sym in symbols:
        lang = sym.get("language", "")
        kind = sym.get("kind", "")
        code = sym.get("code", "")

        #Skip non-callable kinds and unsupported languages

        if lang in _SKIP_LANGUAGES:
            continue
        if kind in ("class", "struct", "script_block", "style_block",
                    "rule", "media_rule", "keyframes_rule"):
            continue
        if not code.strip():
            continue
        if lang not in _LANG_CFG:
            skipped+= 1
            continue

        ts_lang, call_type, extractor = _LANG_CFG[lang]

        if lang not in _parsers:
            _parsers[lang] = Parser(ts_lang)
        parser = _parsers[lang]

        source = code.encode("utf-8")
        tree = parser.parse(source)
        callees = _collect_calls(tree.root_node, source, call_type, extractor)

        #Key = display_name : "ClassName.method" or "function"
        display = sym["parent"] + "." + sym["name"] if sym.get("parent") else sym ["name"]

        #exclude self - reference (eg: recursive call with the same name)
        callees = [c for c in callees if c != sym["name"]]

        graph[display] = callees


    if skipped:
        print(f" [info] Skipped {skipped} symbol(s) with unsupported language.", 
                file = sys.stderr)
            
    return graph
    
