from unittest.mock import patch, MagicMock

from gaia_agent.tools.search import tavily_search


def test_tavily_search_returns_formatted_results():
    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [
            {"title": "T1", "url": "https://a.test", "content": "snippet 1"},
            {"title": "T2", "url": "https://b.test", "content": "snippet 2"},
        ]
    }
    with patch("gaia_agent.tools.search.TavilyClient", return_value=fake_client):
        out = tavily_search("who is X", api_key="tvly-xxx", max_results=2)
    assert "T1" in out
    assert "https://a.test" in out
    assert "snippet 1" in out
    assert "T2" in out


def test_tavily_search_handles_empty_results():
    fake_client = MagicMock()
    fake_client.search.return_value = {"results": []}
    with patch("gaia_agent.tools.search.TavilyClient", return_value=fake_client):
        out = tavily_search("q", api_key="tvly-xxx")
    assert out == "No results."


def test_tavily_search_missing_api_key():
    out = tavily_search("q", api_key="")
    assert "ERROR" in out
