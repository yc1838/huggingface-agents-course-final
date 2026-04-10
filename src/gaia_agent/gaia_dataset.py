"""Load GAIA questions directly from the `gaia-benchmark/GAIA` HF dataset.

Provides a small client with the same surface as `GaiaApiClient`
(`get_questions()` + `download_file(task_id, dest_dir)`), so the rest of the
graph (perception node, runner, app.py) works unchanged.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import requests
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
        self.config = config
        self.split = split
        self.token = token
        
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
        
        file_name = row.get("file_name")
        if not file_name:
            return None

        dest_root = Path(dest_dir)
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / file_name

        # If already here, we are good
        if dest.exists() and dest.stat().st_size > 300:
             return dest

        # 1. Try local 'data/' directory fallback (some files might be there)
        level = row.get("Level", "1")
        local_data_path = Path("data") / f"gaia_level{level}" / "files" / task_id
        if local_data_path.exists() and local_data_path.stat().st_size > 300:
             shutil.copy(local_data_path, dest)
             return dest

        # 2. Try direct download from Hugging Face resolve API
        # GAIA file structure on HF: main/{year}/{split}/{task_id}.{ext}
        # Note: year is derived from config (e.g., '2023_all')
        year = "2023" # default
        if hasattr(self, "config") and "_" in self.config:
            year = self.config.split("_")[0]
        
        split = getattr(self, "split", "validation")
        ext = Path(file_name).suffix
        hf_url = f"https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/{year}/{split}/{task_id}{ext}"
        
        try:
            headers = {}
            if getattr(self, "token", None):
                headers["Authorization"] = f"Bearer {self.token}"
            
            response = requests.get(hf_url, headers=headers, timeout=20, stream=True)
            if response.status_code == 200:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
                return dest
        except Exception:
            pass
            
        return None
