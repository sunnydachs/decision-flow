"""タスク定義の共通インターフェース。

アダプタは「テキスト + TaskDefinition」を受け取り、予測を返す。
カテゴリ定義・タスク指示はコードではなく config/tasks/*.toml に置く(タスクと実装の分離)。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.config import load_task_config


@dataclass(frozen=True)
class Category:
    name: str
    description: str


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    language: str
    instruction: str
    categories: tuple[Category, ...]
    risk_label: str
    choice_instruction: str | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.categories)

    def descriptions(self) -> dict[str, str]:
        return {c.name: c.description for c in self.categories}

    def is_risk_label(self, label: str | None) -> bool:
        return label == self.risk_label

    def choice_criteria(self) -> dict[str, str]:
        """Decision Model の Choice 質問用の criteria(選択肢 -> 説明)。"""
        return self.descriptions()

    def choice_question(self, question_id: str = "category") -> dict[str, dict[str, object]]:
        """Decision Model(System One / Decisions API)用の Choice 質問を組み立てる。

        質問文はタスク定義(config/tasks/*.toml の choice_instruction)から差し替え可能。
        未指定ならカテゴリ説明を並べた既定文を使う。
        """
        if self.choice_instruction:
            instructions = self.choice_instruction
        else:
            instructions = (
                "Which single category best describes this customer inquiry? "
                "Categories: "
                + "; ".join(f"{name} = {desc}" for name, desc in self.descriptions().items())
            )
        return {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": self.choice_criteria(),
            }
        }


def load_task(path: str | Path) -> TaskDefinition:
    """config/tasks/*.toml を読み込んで TaskDefinition を返す。"""
    data = load_task_config(path)
    categories = tuple(
        Category(name=str(c["name"]), description=str(c["description"]))
        for c in data.get("categories", [])
    )
    if len(categories) < 2:
        raise ValueError(f"タスク {path} のカテゴリが 2 件未満です")
    task = TaskDefinition(
        id=str(data["id"]),
        language=str(data.get("language", "ja")),
        instruction=str(data["instruction"]),
        categories=categories,
        risk_label=str(data["risk_label"]),
        choice_instruction=str(data["choice_instruction"]) if data.get("choice_instruction") else None,
    )
    if task.risk_label not in task.labels:
        raise ValueError(f"risk_label={task.risk_label} がカテゴリに存在しません")
    return task
