# AGENTS.md

> Rules for AI Agents (Hermes, Codex, Claude Code, Cursor) working in this repository.

---

## 0. Security & Secret Management (HARDLINE)

- **Zero Secrets in Git**: Never commit API keys, tokens, passwords, private keys, or actual credentials (`.env`, `credentials.json`, `token_*.json`).
- **Environment Variables Only**: All credentials must be loaded via process environment variables (`process.env` / `os.environ.get(...)`).
- **No Hardcoded Secrets**: Do not write key strings in code, comments, `.env.example`, or commit messages.
- **No Absolute Paths**: Do not use local absolute paths (e.g., `<home>/...`).
- **Gitleaks Protection**: Do not bypass gitleaks pre-commit checks with `--no-verify`.

---

## 1. Project Discipline

- **TDD / Testing**: Verify logic changes with unit tests before declaring completion.
- **Debugging**: Follow systematic debugging (Understand -> Minimal Repro -> Fix -> Verify).
- **Commits**: Small, atomic commits with concise messages.
