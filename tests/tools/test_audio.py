from unittest.mock import patch, MagicMock

from gaia_agent.tools.audio import transcribe_audio


def test_transcribe_audio_joins_segments():
    fake_model = MagicMock()
    seg = MagicMock()
    seg.text = " hello world "
    fake_model.transcribe.return_value = ([seg, seg], MagicMock(language="en"))
    with patch("gaia_agent.tools.audio._get_model", return_value=fake_model):
        out = transcribe_audio("/tmp/nope.mp3", model_size="base")
    assert "hello world" in out
    assert out.count("hello world") == 2
