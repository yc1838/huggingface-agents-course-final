import os
from unittest.mock import MagicMock, patch
from gaia_agent.config import Config
from gaia_agent.tools.vision import inspect_visual_content

def test_config_loading():
    print("Testing Config Loading...")
    os.environ["GAIA_VISION_PROVIDER"] = "google"
    os.environ["GAIA_VISION_MODEL"] = "gemini-3-flash-preview"
    
    cfg = Config.from_env()
    assert cfg.vision_provider == "google"
    assert cfg.vision_model == "gemini-3-flash-preview"
    print("✓ Config Loaded Correctly")

def test_tool_initialization():
    print("Testing Tool Initialization...")
    cfg = Config.from_env()
    
    # Mock ChatGoogleGenerativeAI to avoid actual network calls
    with patch("gaia_agent.tools.vision.ChatGoogleGenerativeAI") as mock_genai:
        # Mock file existence
        with patch("gaia_agent.tools.vision.Path.exists", return_value=True):
            with patch("gaia_agent.tools.vision.Path.read_bytes", return_value=b"fake_data"):
                inspect_visual_content("fake_image.png", "What is in this image?", cfg)
                
                # Verify it was called with the correct vision model
                mock_genai.assert_called_once()
                args, kwargs = mock_genai.call_args
                assert kwargs["model"] == "gemini-3-flash-preview"
                print(f"✓ Tool Initialized with model: {kwargs['model']}")

if __name__ == "__main__":
    try:
        test_config_loading()
        test_tool_initialization()
        print("\nALL CONFIG TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
