"""設定ローダ(config/default.toml + 任意の config/local.toml を deep-merge)。

標準ライブラリの tomllib のみを使う(依存を増やさない)。
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    pass


class Config:
    """ネストした dict を dotted path で読む軽量ラッパ。"""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise ConfigError(f"設定キーがありません: {path}")
        return value

    def __getitem__(self, path: str) -> Any:
        return self.require(path)


def _read_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _validate(data: dict[str, Any]) -> None:
    required = [
        "run.concurrency",
        "run.timeout_seconds",
        "evaluation.risk_label",
        "evaluation.risk_severity",
        "evaluation.recall_target",
        "models.llm.primary",
        "models.mercury.model",
        "models.pplx_decider.model",
        "models.jev.model",
    ]
    cfg = Config(data)
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise ConfigError(f"必須設定がありません: {', '.join(missing)}")
    if not 0.0 < float(cfg.get("evaluation.recall_target")) <= 1.0:
        raise ConfigError("evaluation.recall_target は (0, 1] の範囲で指定してください")
    if int(cfg.get("run.concurrency")) < 1:
        raise ConfigError("run.concurrency は 1 以上にしてください")


def load_config(
    path: str | Path | None = None,
    local_path: str | Path | None = None,
    repo_root: Path | None = None,
) -> Config:
    root = repo_root or REPO_ROOT
    cfg_path = Path(path) if path else root / "config" / "default.toml"
    if not cfg_path.exists():
        raise ConfigError(f"設定ファイルが見つかりません: {cfg_path}")
    data = _read_toml(cfg_path)
    local = Path(local_path) if local_path else root / "config" / "local.toml"
    if local.exists():
        data = deep_merge(data, _read_toml(local))
    _validate(data)
    return Config(data)


def load_task_config(path: str | Path, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or REPO_ROOT
    task_path = Path(path)
    if not task_path.is_absolute():
        task_path = root / task_path
    if not task_path.exists():
        raise ConfigError(f"タスク定義が見つかりません: {task_path}")
    return _read_toml(task_path)
