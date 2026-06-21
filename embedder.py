"""
Layer 3 - Search Index Builder (Embeddings + FAISS + BM25)
==========================================================
Reads the tagged sybols.json produced by Layer 2 and builds two parallel
search indexes:

    1. FAISS - dense vector index (semantic similarity)
    2. BM25 - sparse keyword index (exact / token-overlap search)

Both indexes cover the same set of symbols: every entry whose
search_tier == "code". CSS symbols (search_tier == "style") go into a 
separate, lighter FAISS index so style questions can be answered too.

Output (written to --out directory ./repo_index)
-----------
    faiss_code.index    <- FAISS index for code symbols
    faiss_style.index   <- FAISS index for CSS/style symbols (may be absent)
    bm25_code.pkl       <- BM25 index for code.symbols
    index_meta.json     <- id -> symbol mapping consumed by Layers 4 and 5

Usage
-----
    # Build index from symbols.json produced by Layer 2:
    python embedder.py repo_index/symbols.json
 
    # Custom output directory:
    python embedder.py repo_index/symbols.json --out my_index
 
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

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

#------------------------------------------------------------------------------
#Constants
#------------------------------------------------------------------------------

# Default model: fast + good for code. Swap for all-mpnet-base-v2 for ~5%
# better retrieval at ~3x the encoding time

DEFAULT_MODEL = "all-MiniLM-L6-v2"

#how many results each sub - index returns before the merge step
_FAISS_TOP_K = 10
_BM25_TOP_K = 10

# Fusion weight: final_score = FAISS_WEIGHT * fiass_score + BM25_WEIGHT * bm25_score
# Both scores are normalized to [0,1] before combining
FAISS_WEIGHT = 0.6
BM25_WEIGHT = 0.4

#------------------------------------------------------------------------------
# Text representation
#
# A function whose code is just "const x = () => {}" embeds poorly on its own.
# We build a richer text by combining:
#   • display_name  — unique identifier that the model can tokenise meaningfully
#   • docstring     — natural-language description when present  (0 in your current
#                     corpus, but leave the hook here for when you add them)
#   • file path     — gives context: "auth/middleware.js" signals authentication
#   • first N lines of code — enough structure for the model, avoids padding waste
#
# CSS symbols get a leaner representation since their "code" is already concise.
#-------------------------------------------------------------------------------

_CODE_PREVIEW_LINES = 20 #embed this many lines of source; rest is noise

def _build_text(sym: dict) -> str:
    """
    Build the text that will be embedded for a single symbol

    Keep this readabel rather than tese intentionally:
    the model was trained on natural English + code, so a 
    sentence-like header helps it achor the meaning.
    """
    parts: list[str] = []

    #Header line: "function login in auth/middleware.js"
    kind = sym.get("kind", "symbol")
    name = sym.get("name", "unknown")
    parent= sym.get("parent", "")
    file = sym.get("file", "")
    display = f"{parent}.{name}" if parent else name

    parts.append(f"{kind} {display} in {file}")

    #Docstring (Layer 1 captures these for Python and JS)

    doc = sym.get("docstring", "")
    if doc:
        parts.append(doc.strip())

    # Source preview
    code = sym.get("code", "")
    if code:
        lines = code.splitlines()[:_CODE_PREVIEW_LINES]
        parts.append("\n".join(lines))

    return "\n".join(parts)