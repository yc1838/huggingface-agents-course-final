from __future__ import annotations

from email.message import Message
from pathlib import Path

import requests


class GaiaApiClient:
    def __init__(self, api_url: str, timeout: int = 30) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def get_questions(self) -> list[dict]:
        response = requests.get(f"{self.api_url}/questions", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def download_file(self, task_id: str, dest_dir: str | Path) -> Path | None:
        response = requests.get(f"{self.api_url}/files/{task_id}", timeout=self.timeout)
        if response.status_code == 404:
            return None

        response.raise_for_status()
        directory = Path(dest_dir)
        directory.mkdir(parents=True, exist_ok=True)

        filename = self._filename_from_headers(response.headers, task_id)
        file_path = directory / filename
        file_path.write_bytes(response.content)
        return file_path

    def submit(
        self, username: str, agent_code: str, answers: list[dict]
    ) -> dict:
        response = requests.post(
            f"{self.api_url}/submit",
            json={
                "username": username,
                "agent_code": agent_code,
                "answers": answers,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _filename_from_headers(headers: dict[str, str], task_id: str) -> str:
        content_disposition = headers.get("content-disposition", "")
        message = Message()
        message["content-disposition"] = content_disposition
        filename = message.get_param("filename", header="content-disposition")
        return filename or f"{task_id}.bin"
