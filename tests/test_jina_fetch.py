from unittest.mock import MagicMock, patch
from gaia_agent.tools.web import fetch_url

def test_jina_integration():
    print("Testing Jina Reader integration...")
    
    # Mock httpx.get to simulate Jina Success
    with patch("httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "# Jina Markdown Content\nThis is a test."
        mock_get.return_value = mock_resp
        
        result = fetch_url("https://benjerry.com")
        
        # Verify first call was to Jina
        args, kwargs = mock_get.call_args_list[0]
        assert args[0] == "https://r.jina.ai/https://benjerry.com"
        assert kwargs["headers"]["X-Return-Format"] == "markdown"
        assert "# Jina Markdown Content" in result
        print("✓ Jina primary attempt success")

def test_jina_fallback():
    print("Testing Jina Fallback to direct fetch...")
    
    with patch("httpx.get") as mock_get:
        # 1st call (Jina) fails with 403
        mock_jina_resp = MagicMock()
        mock_jina_resp.status_code = 403
        
        # 2nd call (Direct) succeeds
        mock_direct_resp = MagicMock()
        mock_direct_resp.status_code = 200
        mock_direct_resp.text = "<html>Original Content</html>"
        mock_direct_resp.headers = {"Content-Type": "text/html"}
        
        mock_get.side_effect = [mock_jina_resp, mock_direct_resp]
        
        with patch("trafilatura.extract", return_value="Extracted Content"):
            result = fetch_url("https://example.com")
            
            assert mock_get.call_count == 2
            assert result == "Extracted Content"
            print("✓ Successfully fell back to direct fetch after Jina failure")

if __name__ == "__main__":
    try:
        test_jina_integration()
        test_jina_fallback()
        print("\nALL JINA WEB TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
