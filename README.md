# Project

Workspace for tooling, scripts, and non-wiki project files.

## tools/

Optional CLI utilities for operating on the wiki:

- **search.py** (build when needed) — full-text search over `wiki/pages/`
- Other scripts as the project grows

## Getting started

1. Drop source documents into `wiki/raw/`.
2. Open a Claude Code session in this directory (`CLAUDE.md` loads automatically).
3. Say `ingest [filename]` to process a source.
4. Ask questions; say `lint` periodically to keep the wiki healthy.
