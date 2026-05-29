# CodePilot

CodePilot is a repository intelligence system that aims to understand codebases structurally rather than treating them as plain text.

The goal is to allow users to ask questions about a repository such as:

- Where is authentication handled?
- How does the training pipeline work?
- Which functions are related to database access?

and receive answers grounded in the actual code.

## Current Status

### Layer 1 — AST Parsing ✅
- Multi-language parsing using Tree-sitter
- Function extraction
- Class extraction
- Method extraction
- Metadata collection
- Flat symbol generation

### Layer 2 — Call Graph Construction 🚧
- AST traversal for call detection
- Multi-language call extraction
- Caller → callee relationship mapping
- Symbol relationship graph generation

### Planned Layers

#### Layer 3 — Retrieval Indexing
- Embeddings
- FAISS vector search
- BM25 keyword search
- Hybrid retrieval

#### Layer 4 — Portable Repository Index
- Persist embeddings
- Persist call graphs
- Save repository metadata

#### Layer 5 — RAG Interface
- Repository question answering
- Context expansion using call graphs
- LLM-powered responses

## Architecture

```text
Repository
    ↓
AST Parsing
    ↓
Symbol Extraction
    ↓
Call Graph
    ↓
Embeddings + BM25
    ↓
Retrieval
    ↓
RAG
