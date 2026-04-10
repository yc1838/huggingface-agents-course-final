from __future__ import annotations

from pathlib import Path

from gaia_agent.state import AgentState, Modality

_EXT_TO_MODALITY: dict[str, Modality] = {
    ".pdf": "pdf",
    ".xlsx": "excel",
    ".xls": "excel",
    ".csv": "csv",
    ".docx": "docx",
    ".doc": "docx",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
}


def _modality_from_question(question: str) -> Modality:
    lowered = question.lower()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "youtube"
    if "http://" in lowered or "https://" in lowered:
        return "web"
    return "text"


def make_perception_node(client, file_dir: Path):
    def perception(state: AgentState) -> dict:
        path = client.download_file(state["task_id"], file_dir)
        
        # Determine modality
        modality: Modality = "text"
        
        if path:
            modality = _EXT_TO_MODALITY.get(path.suffix.lower(), "text")
        else:
            # Fallback: try to see if the dataset row tells us the expected file type
            row = next((r for r in getattr(client, "_rows", []) if r["task_id"] == state["task_id"]), None)
            if row and row.get("file_name"):
                suffix = Path(row["file_name"]).suffix.lower()
                modality = _EXT_TO_MODALITY.get(suffix, "text")
            
            # If still text, check question for URLs
            if modality == "text":
                modality = _modality_from_question(state["question"])

        return {
            "file_path": str(path) if path else None,
            "modality": modality,
        }

    return perception
