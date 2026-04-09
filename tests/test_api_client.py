from pathlib import Path
from unittest.mock import Mock, patch

from gaia_agent.api_client import GaiaApiClient


def test_get_questions_returns_response_json():
    response = Mock()
    response.json.return_value = [{"task_id": "1", "Question": "Hi"}]
    response.raise_for_status.return_value = None

    with patch("gaia_agent.api_client.requests.get", return_value=response) as mock_get:
        client = GaiaApiClient("https://example.test")

        result = client.get_questions()

    assert result == [{"task_id": "1", "Question": "Hi"}]
    mock_get.assert_called_once_with("https://example.test/questions", timeout=30)


def test_download_file_uses_content_disposition_filename(tmp_path):
    response = Mock()
    response.status_code = 200
    response.headers = {"content-disposition": 'attachment; filename="notes.txt"'}
    response.content = b"hello"
    response.raise_for_status.return_value = None

    with patch("gaia_agent.api_client.requests.get", return_value=response) as mock_get:
        client = GaiaApiClient("https://example.test")

        file_path = client.download_file("task-1", tmp_path)

    assert file_path == tmp_path / "notes.txt"
    assert file_path.read_bytes() == b"hello"
    mock_get.assert_called_once_with("https://example.test/files/task-1", timeout=30)


def test_download_file_falls_back_to_task_id_filename(tmp_path):
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.content = b"\x00\x01"
    response.raise_for_status.return_value = None

    with patch("gaia_agent.api_client.requests.get", return_value=response):
        client = GaiaApiClient("https://example.test")

        file_path = client.download_file("task-2", Path(tmp_path))

    assert file_path == tmp_path / "task-2.bin"
    assert file_path.read_bytes() == b"\x00\x01"


def test_download_file_returns_none_on_404(tmp_path):
    response = Mock()
    response.status_code = 404

    with patch("gaia_agent.api_client.requests.get", return_value=response):
        client = GaiaApiClient("https://example.test")

        file_path = client.download_file("missing-task", tmp_path)

    assert file_path is None


def test_submit_posts_expected_payload():
    response = Mock()
    response.json.return_value = {"score": 1.0}
    response.raise_for_status.return_value = None
    answers = [{"task_id": "1", "submitted_answer": "4"}]

    with patch("gaia_agent.api_client.requests.post", return_value=response) as mock_post:
        client = GaiaApiClient("https://example.test")

        result = client.submit("alice", "gaia-agent", answers)

    assert result == {"score": 1.0}
    mock_post.assert_called_once_with(
        "https://example.test/submit",
        json={
            "username": "alice",
            "agent_code": "gaia-agent",
            "answers": answers,
        },
        timeout=30,
    )
