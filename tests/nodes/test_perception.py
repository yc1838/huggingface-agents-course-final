from pathlib import Path
from unittest.mock import MagicMock

from gaia_agent.nodes.perception import make_perception_node
from gaia_agent.state import new_state


def test_perception_defaults_to_text_without_file():
    client = MagicMock()
    client.download_file.return_value = None
    node = make_perception_node(client, Path("/tmp/gaia"))

    out = node(new_state("task-1", "What is 2+2?"))

    assert out == {"file_path": None, "modality": "text"}


def test_perception_detects_pdf_attachment(tmp_path):
    fake_pdf = tmp_path / "notes.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")
    client = MagicMock()
    client.download_file.return_value = fake_pdf
    node = make_perception_node(client, tmp_path)

    out = node(new_state("task-2", "Summarize the attachment"))

    assert out == {"file_path": str(fake_pdf), "modality": "pdf"}


def test_perception_detects_youtube_question_without_attachment():
    client = MagicMock()
    client.download_file.return_value = None
    node = make_perception_node(client, Path("/tmp/gaia"))

    out = node(new_state("task-3", "Watch https://youtu.be/abc and summarize it"))

    assert out == {"file_path": None, "modality": "youtube"}
