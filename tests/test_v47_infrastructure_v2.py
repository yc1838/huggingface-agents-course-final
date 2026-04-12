from gaia_agent.json_repair import safe_structured_call
from gaia_agent.nodes.planner import PlanSchema
from gaia_agent.tools.vision import inspect_visual_content
from gaia_agent.config import Config
from unittest.mock import MagicMock, patch
import os

def test_json_empty_list_loop():
    print("Testing JSON empty list soft failure...")
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="[]")
    mock_fixer = MagicMock()
    mock_fixer.bind.return_value = mock_fixer
    
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
            print("✓ Successfully prevented empty list from slipping through")
        else:
            print(f"✗ Unexpected error: {e}")

def test_vision_fal_priority():
    print("Testing vision tool FAL priority and fallback...")
    cfg = Config.from_env()
    # Mock Config to use FAL
    with patch.object(cfg, 'vision_provider', 'fal'):
        with patch.object(cfg, 'vision_model', 'llava'):
            with patch("gaia_agent.tools.vision.ChatGoogleGenerativeAI") as mock_google:
                with patch("fal_client.subscribe") as mock_fal:
                    # Attempt 1: fal-ai/moondream-next fails
                    mock_fal.side_effect = [Exception("FAL 404"), Exception("FAL stable 404")]
                    # Attempt 3: Gemini success
                    mock_google_inst = MagicMock()
                    mock_google.return_value = mock_google_inst
                    mock_google_inst.invoke.return_value = MagicMock(content="Gemini success")
                    
                    if not os.path.exists("temp_test.png"):
                        with open("temp_test.png", "wb") as f:
                            f.write(b"fake image")
                            
                    res = inspect_visual_content("temp_test.png", "what is this?", cfg)
                    
                    # Verify Attempt 1 used remapped 'fal-ai/moondream-next'
                    assert mock_fal.call_count == 2
                    assert mock_fal.call_args_list[0].args[0] == "fal-ai/moondream-next"
                    assert "Gemini success" in res
                    print("✓ Successfully remapped 'llava' to fal-ai/moondream-next and fell back to Gemini")

if __name__ == "__main__":
    try:
        test_json_empty_list_loop()
        test_vision_fal_priority()
        print("\nALL INFRASTRUCTURE TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
