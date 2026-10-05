"""埋め込み + ロジスティック回帰(ローカル実行、外部送信なし)。

埋め込みは fastembed(ONNX)で多言語モデルをローカル実行する。モデルIDは config で差し替え可能。
ロジスティック回帰は calib データで学習する(小標本のため制約としてレポートに明記)。
確率は分類器の出力であり、モデルの確率として較正の検証対象にできる。
"""
from __future__ import annotations

from pathlib import Path

from adapters.base import BaseAdapter, Prediction
from common.dataset import Item
from tasks.base import TaskDefinition


class EmbeddingBackend:
    """fastembed によるローカル埋め込み(遅延 import)。"""

    def __init__(self, model_name: str, query_prefix: str = "", passage_prefix: str = ""):
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as exc:  # pragma: no cover - 環境依存
            raise SystemExit(
                "fastembed が必要です: `uv pip install -e \".[embed]\"` を実行してください"
            ) from exc
        self.model = TextEmbedding(model_name=model_name)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        prefix = self.query_prefix if query else self.passage_prefix
        payload = [f"{prefix}{t}" for t in texts]
        return [list(map(float, vec)) for vec in self.model.embed(payload)]


class EmbeddingLrAdapter(BaseAdapter):
    name = "embedding_lr"
    tier = "local"
    trained_on = None

    def __init__(self, task: TaskDefinition, config, *, repo_root: Path, backend: EmbeddingBackend | None = None):
        super().__init__(task, model=str(config.get("embedding.model")))
        self.repo_root = Path(repo_root)
        self.config = config
        self.backend = backend
        self.classifier = None
        self._labels: list[str] = list(task.labels)

    def ensure_backend(self) -> EmbeddingBackend:
        if self.backend is None:
            self.backend = EmbeddingBackend(
                str(self.config.get("embedding.model")),
                query_prefix=str(self.config.get("embedding.query_prefix", "")),
                passage_prefix=str(self.config.get("embedding.passage_prefix", "")),
            )
        return self.backend

    def train(self, items: list[Item], oof: bool = False) -> None:
        """calib で学習する。oof=True のときは out-of-fold 予測も用意する。

        calib 自身を評価する場合(閾値決定)に学習リークを避けるため、
        calib のスコアは out-of-fold 予測を使う。test では本モデルで予測する。
        """
        from collections import Counter

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        backend = self.ensure_backend()
        texts = [item.text for item in items]
        labels = [item.label for item in items]
        vectors = backend.embed(texts, query=False)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(vectors, labels)
        self.classifier = clf
        self._labels = list(clf.classes_)
        self.trained_on = {"n": len(items), "source": "calib", "oof": bool(oof)}
        self._oof = {}
        if not oof:
            return
        counts = Counter(labels)
        n_splits = min(5, min(counts.values()) if counts else 0)
        if n_splits < 2:
            return
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(self.config.get("run.seed", 0)))
        for train_idx, test_idx in skf.split(vectors, labels):
            fold = LogisticRegression(max_iter=2000)
            fold.fit([vectors[i] for i in train_idx], [labels[i] for i in train_idx])
            proba = fold.predict_proba([vectors[i] for i in test_idx])
            for row, item_idx in enumerate(test_idx):
                self._oof[items[item_idx].id] = {
                    str(c): float(p) for c, p in zip(fold.classes_, proba[row])
                }

    def predict(self, item: Item, run_index: int = 0) -> Prediction:
        if self.classifier is None:
            raise RuntimeError("train() を先に呼んでください(calib で学習)")
        oof_probs = getattr(self, "_oof", {}).get(item.id)
        if oof_probs is not None:
            probs = oof_probs
            source = "oof"
        else:
            backend = self.ensure_backend()
            vector = backend.embed([item.text], query=True)
            proba = self.classifier.predict_proba(vector)[0]
            probs = {str(label): float(p) for label, p in zip(self.classifier.classes_, proba)}
            source = "model"
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        risk = probs.get(self.task.risk_label)
        prediction = self._prediction(
            item,
            run_index=run_index,
            label=label,
            probabilities=probs,
            value_type="probability",
            risk_score=risk,
            risk_score_type="probability",
            raw_response={"classes": list(self.classifier.classes_), "source": source},
            latency_ms=0.0,
        )
        self._audit(prediction, text=item.text)
        return prediction
