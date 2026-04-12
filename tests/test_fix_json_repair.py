from pydantic import BaseModel
from typing import List
from unittest.mock import MagicMock
from gaia_agent.json_repair import safe_structured_call

class PlanStep(BaseModel):
    action: str

class PlanSchema(BaseModel):
    plan: List[PlanStep]

def test_list_wrapping():
    print("Testing List-to-Dict wrapping in json_repair...")
    
    # Mock model that returns a JSON list instead of the expected object
    mock_model = MagicMock()
    mock_response = MagicMock()
    # The record the user complained about: LLM returns []
    mock_response.content = "[]"
    mock_model.invoke.return_value = mock_response

    # This should now succeed instead of raising a validation error or entering a fix loop
    try:
        result = safe_structured_call(
            model=mock_model,
            messages=[],
            target_schema=PlanSchema,
            node_name="test_node"
        )
        assert isinstance(result, PlanSchema)
        assert result.plan == []
        print("✓ Successfully wrapped [] into {'plan': []}")
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

    # Test with a populated list
    mock_response.content = '[{"action": "search"}]'
    try:
        result = safe_structured_call(
            model=mock_model,
            messages=[],
            target_schema=PlanSchema,
            node_name="test_node"
        )
        assert len(result.plan) == 1
        assert result.plan[0].action == "search"
        print("✓ Successfully wrapped populated list into {'plan': [...]}")
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    test_list_wrapping()
    print("\nALL JSON REPAIR TESTS PASSED")
