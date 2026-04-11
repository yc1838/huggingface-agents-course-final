import logging
import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from gaia_agent.nodes.planner import make_planner_node
from gaia_agent.state import AgentState
from langchain_openai import ChatOpenAI # Using a placeholder model for local testing if possible, or just mock it

# Setup logging
logging.basicConfig(level=logging.INFO)

class MockModel:
    def invoke(self, messages):
        print("\n--- MOCK MODEL RECEIVED ---")
        for m in messages:
            print(f"[{type(m).__name__}]: {m.content[:500]}...")
        
        # Determine if it's a recovery
        is_recovery = any("Prior critique" in m.content for m in messages if hasattr(m, "content"))
        
        if is_recovery:
            return type('obj', (object,), {'content': '```json\n{"plan": [{"thought": "Critique says format issue. Just fix it.", "description": "DRAFT: 75%", "tier": "S1"}]}\n```'})
        else:
            return type('obj', (object,), {'content': '```json\n{"plan": [{"thought": "Search for data", "description": "Search USDA", "tier": "S1"}]}\n```'})

def test_recovery():
    model = MockModel()
    planner_node = make_planner_node(model)
    
    # Simulate a REWORK state
    state: AgentState = {
        "task_id": "test-123",
        "question": "What is the percentage?",
        "file_path": None,
        "modality": "text",
        "plan": [],
        "step_idx": 0,
        "observations": [{"step_idx": 0, "tool": "test", "args": {}, "result": "Found 75%**"}],
        "working_memory": "The data says 75%**",
        "draft_answer": "75%**",
        "critique": "Remove the trailing stars.",
        "current_domain": None,
        "current_strategy": None,
        "retries": 1,
        "final_answer": None,
    }
    
    result = planner_node(state)
    
    print("\n--- TEST RESULT ---")
    print(f"Plan: {result['plan']}")
    print(f"Working Memory Preserved: {result['working_memory'] != ''}")
    print(f"Observations Preserved: {len(result['observations']) > 0}")
    
    assert result['working_memory'] == "The data says 75%**"
    assert len(result['observations']) == 1
    assert result['plan'][0]['description'] == "DRAFT: 75%"

if __name__ == "__main__":
    test_recovery()
