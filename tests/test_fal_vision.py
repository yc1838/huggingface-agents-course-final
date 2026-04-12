import os
import sys
from unittest.mock import MagicMock, patch

# Mock the SDK since it's not installed yet
sys.modules["fal_client"] = MagicMock()
import fal_client

from gaia_agent.config import Config
from gaia_agent.tools.vision import inspect_visual_content

def test_fal_config_loading():
    print("Testing FAL Config Loading...")
    os.environ["GAIA_VISION_PROVIDER"] = "fal"
    os.environ["GAIA_VISION_MODEL"] = "fal-ai/llava/v1.5/7b"
    os.environ["GAIA_FAL_VISION_API_KEY"] = "fake_fal_key"
    
    cfg = Config.from_env()
    assert cfg.vision_provider == "fal"
    assert cfg.vision_model == "fal-ai/llava/v1.5/7b"
    assert cfg.fal_vision_api_key == "fake_fal_key"
    print("✓ Config Loaded Correctly")

def test_fal_tool_logic():
    print("Testing FAL Tool Logic...")
    cfg = Config.from_env()
    
    # Mock return value from fal-client
    mock_result = {"text": "A detailed image description."}
    fal_client.subscribe.return_value = mock_result
    
    # Mock file existence
    with patch("gaia_agent.tools.vision.Path.exists", return_value=True):
        with patch("gaia_agent.tools.vision.Path.read_bytes", return_value=b"fake_data"):
            result = inspect_visual_content("fake_image.png", "What is in this image?", cfg)
            
            # Verify it was called with the correct model and prompt
            fal_client.subscribe.assert_called_once()
            args, kwargs = fal_client.subscribe.call_args
            assert args[0] == "fal-ai/llava/v1.5/7b"
            assert kwargs["arguments"]["prompt"] == "What is in this image?"
            assert kwargs["arguments"]["image_url"].startswith("data:image/png;base64,")
            
            assert result == "A detailed image description."
            print(f"✓ Tool correctly integrated with fal.ai SDK (Mocked)")

if __name__ == "__main__":
    try:
        test_fal_config_loading()
        test_fal_tool_logic()
        print("\nALL FAL INTEGRATION TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
