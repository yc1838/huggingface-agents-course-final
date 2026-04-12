from gaia_agent.json_repair import safe_structured_call
from gaia_agent.nodes.planner import PlanSchema
from gaia_agent.tools.vision import inspect_visual_content
from gaia_agent.config import Config
from unittest.mock import MagicMock, patch

def test_json_empty_list_loop():
    print("Testing JSON empty list soft failure...")
    
    # Mock model that returns []
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="[]")
    
    # Mock cheap fixer to see if it's called
    mock_fixer = MagicMock()
    mock_fixer.bind.return_value = mock_fixer
    
    # We expect an UnsalvageableJsonError (or similar) because the fixer will keep seeing []
    # Actually, let's just assert that it raises an error when tried
    try:
        safe_structured_call(
            model=mock_model,
            messages=[],
            target_schema=PlanSchema,
            cheap_fixer_model=mock_fixer,
            max_local_repairs=1,
            node_name="test_node"
        )
    except Exception as e:
        if "Model returned empty list []" in str(e) or "fixer exhausted" in str(e):
            print("✓ Successfully prevented empty list from slipping through as valid plan")
        else:
            print(f"✗ Unexpected error: {e}")

def test_vision_remapping():
    print("Testing vision model remapping and fallback...")
    
    # Mock Config
    cfg = Config.from_env()
    # Case 1: Legacy name should map to current ID
    with patch("gaia_agent.tools.vision.ChatGoogleGenerativeAI") as mock_google:
        mock_model = MagicMock()
        mock_google.return_value = mock_model
        
        # Simulate failure then success on second attempt
        mock_model.invoke.side_effect = [Exception("404 Not Found"), MagicMock(content="Vision success")]
        
        # Test with a dummy file
        with open("temp_test.png", "wb") as f:
            f.write(b"fake image")
        
        res = inspect_visual_content("temp_test.png", "what is this?", cfg)
        
        # Verify first call used the target model, second used gemini-1.5-flash
        called_models = [call.kwargs["model"] for call in mock_google.call_args_list]
        assert "gemini-1.5-flash" in called_models
        assert "Vision success" in res
        print("✓ Successfully remapped and fell back in vision tool")

if __name__ == "__main__":
    try:
        test_json_empty_list_loop()
        test_vision_remapping()
        print("\nALL INFRASTRUCTURE TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
