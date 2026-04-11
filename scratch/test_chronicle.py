import logging
import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from gaia_agent.nodes.reflector import make_reflector_node
from gaia_agent.state import AgentState

# Setup logging
logging.basicConfig(level=logging.INFO)

class MockModel:
    def invoke(self, messages):
        print("\n--- MOCK REFLECTOR INVOKED ---")
        return type('obj', (object,), {
            'content': 'Reasoning: Tool found doc.\nUPDATED WORKING MEMORY: Found data.\nCHRONICLE UPDATE: Found USDA 1959 document.\nMATCH FOUND: 75%'
        })

def test_chronicle():
    model = MockModel()
    reflector_node = make_reflector_node(model)
    
    state: AgentState = {
        "task_id": "test-chronicle",
        "question": "TEST",
        "file_path": None,
        "modality": "text",
        "plan": [{"description": "Search"}],
        "step_idx": 1,
        "observations": [{"step_idx": 0, "tool": "test", "args": {}, "result": "raw data"}],
        "working_memory": "",
        "task_chronicle": " - Step 0: Initialized.",
        "draft_answer": None,
        "critique": None,
        "current_domain": None,
        "current_strategy": None,
        "retries": 0,
        "final_answer": None,
    }
    
    result = reflector_node(state)
    
    print("\n--- TEST RESULT ---")
    print(f"New Memory: {result.get('working_memory')}")
    print(f"New Chronicle: {result.get('task_chronicle')}")
    print(f"Draft Answer: {result.get('draft_answer')}")
    
    assert "Found USDA 1959 document." in result['task_chronicle']
    assert "Step 1" in result['task_chronicle']
    assert result['draft_answer'] == "75%"

if __name__ == "__main__":
    test_chronicle()
