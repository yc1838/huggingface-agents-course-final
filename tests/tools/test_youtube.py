from unittest.mock import patch

from gaia_agent.tools.youtube import extract_video_id, youtube_transcript


def test_extract_video_id_from_watch_url():
    assert (
        extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )


def test_extract_video_id_from_short_url():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=10") == "dQw4w9WgXcQ"


def test_extract_video_id_from_embed_url():
    assert (
        extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )


def test_extract_video_id_invalid_raises():
    import pytest

    with pytest.raises(ValueError):
        extract_video_id("https://example.com/notyoutube")


def test_youtube_transcript_joins_segments():
    fake = [{"text": "hello"}, {"text": "world"}]
    with patch(
        "gaia_agent.tools.youtube.YouTubeTranscriptApi.fetch",
        return_value=fake,
    ):
        out = youtube_transcript("https://youtu.be/dQw4w9WgXcQ")
    assert "hello world" in out
