from __future__ import annotations

from langgraph.graph import END, StateGraph

from gaia_agent.nodes.executor import make_executor_node
from gaia_agent.nodes.formatter import make_formatter_node
from gaia_agent.nodes.state_manager import make_state_manager_node
from gaia_agent.nodes.planner import make_planner_node
from gaia_agent.nodes.reflector import make_reflector_node
from gaia_agent.nodes.router import route_next
from gaia_agent.nodes.verifier import make_verifier_node, verifier_decision
from gaia_agent.state import AgentState


def build_graph(
    perception_node,
    planner_model,
    orchestrator_model,
    strong_model,
    cheap_model,
    verifier_model,
    tools,
    caveman: bool = False,
    caveman_mode: str = "full",
):
    graph = StateGraph(AgentState)

    graph.add_node("perception", perception_node)
    graph.add_node("planner", make_planner_node(planner_model, cheap_model=cheap_model, caveman=caveman, caveman_mode=caveman_mode))
    
    # MISSION 2: Specialized Multi-Agent Nodes
    # The State Manager uses the CHEAP model for dispatching and orchestrating todos
    graph.add_node("state_manager", make_state_manager_node(cheap_model, caveman=caveman, caveman_mode=caveman_mode))
    
    # Use CHEAP model for structured execution (Math)
    graph.add_node("exec_math", make_executor_node(cheap_model, tools, caveman=caveman, caveman_mode=caveman_mode))
    
    # Use STRONG model for all other execution tasks (Research, Vision, etc.)
    graph.add_node("exec_research", make_executor_node(strong_model, tools, caveman=caveman, caveman_mode=caveman_mode))
    graph.add_node("exec_vision", make_executor_node(strong_model, tools, caveman=caveman, caveman_mode=caveman_mode))
    graph.add_node("exec_audio", make_executor_node(strong_model, tools, caveman=caveman, caveman_mode=caveman_mode))
    graph.add_node("exec_file", make_executor_node(strong_model, tools, caveman=caveman, caveman_mode=caveman_mode))
    graph.add_node("exec_general", make_executor_node(strong_model, tools, caveman=caveman, caveman_mode=caveman_mode))

    graph.add_node("reflector", make_reflector_node(verifier_model, caveman=caveman, caveman_mode=caveman_mode))
    graph.add_node("verifier", make_verifier_node(verifier_model, caveman=caveman, caveman_mode=caveman_mode))
    graph.add_node("formatter", make_formatter_node(verifier_model, caveman=caveman, caveman_mode=caveman_mode))

    graph.set_entry_point("perception")
    graph.add_edge("perception", "planner")
    graph.add_edge("planner", "state_manager")
    
    graph.add_conditional_edges(
        "state_manager",
        route_next,
        {
            "exec_math": "exec_math",
            "exec_research": "exec_research",
            "exec_vision": "exec_vision",
            "exec_audio": "exec_audio",
            "exec_file": "exec_file",
            "exec_general": "exec_general",
            "verifier": "verifier"
        },
    )
    
    # All executors point to reflector
    for node in ["exec_math", "exec_research", "exec_vision", "exec_audio", "exec_file", "exec_general"]:
        graph.add_edge(node, "reflector")
        
    graph.add_edge("reflector", "state_manager")
    
    graph.add_conditional_edges(
        "verifier",
        verifier_decision,
        {"planner": "planner", "formatter": "formatter"},
    )
    graph.add_edge("formatter", END)

    return graph.compile()
