"""API キーの解決。

解決順: プロセス環境変数 → リポジトリ直下 `.env` → `~/.hermes/.env`(最後の受け皿)。
キーの値は決してログ・例外メッセージ・結果ファイルに出さない。
存在確認だけが必要な場合は `key_status()` を使う(値の代わりに True/False を返す)。
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_NAMES = ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "PERPLEXITY_API_KEY")
_HOME_ENV = Path.home() / ".hermes" / ".env"


def default_env_files(repo_root: Path | None = None) -> list[Path]:
    root = repo_root or Path.cwd()
    return [root / ".env", _HOME_ENV]


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_keys(env_files: list[Path] | None = None, names: tuple[str, ...] | None = None) -> dict[str, str]:
    """見つかったキーだけを {name: value} で返す。呼び出し側で表示しないこと。"""
    files = env_files if env_files is not None else default_env_files()
    file_env: dict[str, str] = {}
    for path in files:
        file_env.update(_read_env_file(path))
    found: dict[str, str] = {}
    for name in (names or ENV_NAMES):
        value = os.environ.get(name) or file_env.get(name)
        if value:
            found[name] = value
    return found


def lookup_key(name: str, env_files: list[Path] | None = None) -> str | None:
    """任意の名前のキーを 1 つ解決する(env → .env → ~/.hermes/.env)。値は表示しないこと。"""
    files = env_files if env_files is not None else default_env_files()
    value = os.environ.get(name)
    if value:
        return value
    for path in files:
        found = _read_env_file(path).get(name)
        if found:
            return found
    return None


def key_status(env_files: list[Path] | None = None) -> dict[str, bool]:
    """キーの有無のみを返す(値は返さない)。セットアップ確認・レポート用。"""
    keys = load_keys(env_files)
    return {name: name in keys for name in ENV_NAMES}


def require_key(name: str, env_files: list[Path] | None = None, repo_root: Path | None = None) -> str:
    if env_files is None and repo_root is not None:
        env_files = default_env_files(repo_root)
    value = lookup_key(name, env_files)
    if value:
        return value
    raise SystemExit(
        f"{name} が見つかりません。リポジトリ直下の .env(git-ignored)に設定するか、"
        f"環境変数で渡してください。書式は .env.example を参照。"
    )
