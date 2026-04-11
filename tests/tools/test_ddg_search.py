from unittest.mock import patch, MagicMock
import pytest
from gaia_agent.tools.ddg_search import web_search

def test_web_search_success():
    """Test happy path for web_search."""
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.return_value = mock_ddgs
    mock_ddgs.text.return_value = [
        {"title": "Result 1", "href": "http://test1.com", "body": "Snippet 1"},
        {"title": "Result 2", "href": "http://test2.com", "body": "Snippet 2"}
    ]
    
    with patch("ddgs.DDGS", return_value=mock_ddgs):
        res = web_search("query")
        assert "Result 1" in res
        assert "http://test1.com" in res
        assert "Snippet 1" in res
        assert "METADATA" in res

def test_web_search_no_results():
    """Test when no results are found."""
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.return_value = mock_ddgs
    mock_ddgs.text.return_value = []
    
    with patch("ddgs.DDGS", return_value=mock_ddgs):
        res = web_search("empty query")
        assert res == "No results."

def test_web_search_exception():
    """Test handling of general exceptions."""
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.side_effect = Exception("Network error")
    
    with patch("ddgs.DDGS", return_value=mock_ddgs):
        res = web_search("query")
        assert "ERROR: DuckDuckGo search failed: Network error" in res
