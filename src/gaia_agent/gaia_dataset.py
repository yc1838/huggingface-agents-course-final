"""Load GAIA questions directly from the `gaia-benchmark/GAIA` HF dataset.

Provides a small client with the same surface as `GaiaApiClient`
(`get_questions()` + `download_file(task_id, dest_dir)`), so the rest of the
graph (perception node, runner, app.py) works unchanged.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from datasets import load_dataset


class GaiaDatasetClient:
    def __init__(
        self,
        config: str = "2023_all",
        split: str = "validation",
        level: str | None = "1",
        limit: int | None = None,
        token: str | None = None,
    ) -> None:
        ds = load_dataset("gaia-benchmark/GAIA", config, split=split, token=token)
        self._rows: list[dict[str, Any]] = []
        for row in ds:
            if level is not None and str(row.get("Level")) != str(level):
                continue
            self._rows.append(row)
            if limit is not None and len(self._rows) >= limit:
                break

    def get_questions(self) -> list[dict]:
        """Return questions in the shape the runner/graph expect."""
        return [
            {
                "task_id": row["task_id"],
                "question": row["Question"],
                "Level": row.get("Level"),
                "file_name": row.get("file_name", ""),
                "expected_answer": row.get("Final answer", ""),
            }
            for row in self._rows
        ]

    def download_file(self, task_id: str, dest_dir: str | Path) -> Path | None:
        """Copy the dataset's local file_path into dest_dir (matches API client)."""
        row = next((r for r in self._rows if r["task_id"] == task_id), None)
        if row is None:
            return None
        file_path = row.get("file_path")
        file_name = row.get("file_name")
        if not file_path or not file_name:
            return None
        src = Path(file_path)
        if not src.exists():
            return None
        dest_root = Path(dest_dir)
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / file_name
        if not dest.exists():
            shutil.copy(src, dest)
        return dest
