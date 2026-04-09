from unittest.mock import patch, MagicMock

from gaia_agent.tools.web import fetch_url


def test_fetch_url_extracts_main_text():
    html = (
        "<html><body><article><h1>Hello</h1>"
        "<p>World body with enough text to extract properly.</p>"
        "</article></body></html>"
    )
    fake_resp = MagicMock(status_code=200, text=html)
    fake_resp.raise_for_status = MagicMock()
    with patch("gaia_agent.tools.web.httpx.get", return_value=fake_resp):
        out = fetch_url("https://example.test/a")
    assert "Hello" in out or "World body" in out


def test_fetch_url_truncates_long_output():
    html = "<html><body><p>" + ("x " * 20000) + "</p></body></html>"
    fake_resp = MagicMock(status_code=200, text=html)
    fake_resp.raise_for_status = MagicMock()
    with patch("gaia_agent.tools.web.httpx.get", return_value=fake_resp):
        out = fetch_url("https://example.test/a", max_chars=1000)
    assert len(out) <= 1000 + 50
