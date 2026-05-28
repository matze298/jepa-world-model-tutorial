# Repository Guidance

## Chapter Review Checklist

When reviewing or editing tutorial chapters, check both the local chapter and its place in the full tutorial arc.

- Compare the chapter against `docs/index.md` and `mkdocs.yml`. Section titles, numbering, links, status labels, and roadmap descriptions should agree.
- Treat the tutorial as a build-along project. If a chapter says something is implemented, verify whether the repository actually contains that code. If the code is only specified by the tutorial, say that precisely.
- Keep chapter status labels current. Use `reviewed` only after the content has been checked for consistency, completeness, and notation clarity.
- Introduce mathematical notation before using it. If a symbol was already defined in an earlier chapter or section, do not repeat the full definition unnecessarily, but make sure local usage remains consistent with that definition.
- Check tensor shapes and symbol names across nearby sections. Do not reuse a symbol for a different object unless the text explicitly explains the change.
- Verify that equations, prose, code snippets, and planned file names describe the same interfaces. Pay special attention to config field names, notebook names, script names, CLI flags, and tensor dimensions.
- Prefer precise wording about implementation state. Use “implements” for code that exists or is being directly built in that section; use “specifies” or “the accompanying implementation should contain” when the docs describe a target not yet present in the repo.
- Keep command examples aligned with setup assumptions. This repo uses `uv` for setup and dependency management, but chapter commands may assume the `.venv` created by `./setup.py` has been activated.
- Review transitions between sections and chapters. Each section should state what it relies on from earlier material and what the next section will add.
- Avoid broad rewrites during consistency passes. Make targeted edits that preserve the tutorial’s voice and structure.
- After edits, run `git diff --check` and `uv run mkdocs build --strict` before claiming the docs are clean.
- If committing directly to `main`, note that the local hook blocks main commits. Only bypass it when the user explicitly requested direct commits to `main`.

## Useful Review Questions

Ask the user when the answer affects scope or wording materially:

- Should the chapter describe already-existing code, or the implementation target readers are expected to write next?
- Should a section be marked `drafted`, `reviewed`, or left as `to generate`?
- Should commands assume an activated `.venv`, or should a particular example intentionally show `uv run`?
- Should a notation definition be repeated locally for readability, even if it was introduced earlier?
