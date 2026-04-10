from __future__ import annotations

from langgraph.graph import END, StateGraph

from gaia_agent.nodes.executor import make_executor_node
from gaia_agent.nodes.formatter import formatter
from gaia_agent.nodes.orchestrator import make_orchestrator_node
from gaia_agent.nodes.planner import make_planner_node
from gaia_agent.nodes.reflector import make_reflector_node
from gaia_agent.nodes.router import route_next
from gaia_agent.nodes.verifier import make_verifier_node, verifier_decision
from gaia_agent.state import AgentState


def build_graph(
    perception_node,
    planner_model,
    executor_model_s1,
    executor_model_s2,
    verifier_model,
    tools,
):
    graph = StateGraph(AgentState)

    graph.add_node("perception", perception_node)
    graph.add_node("planner", make_planner_node(planner_model))
    graph.add_node("orchestrator", make_orchestrator_node(executor_model_s1))
    graph.add_node("exec_s1", make_executor_node(executor_model_s1, tools))
    graph.add_node("exec_s2", make_executor_node(executor_model_s2, tools))
    graph.add_node("reflector", make_reflector_node(verifier_model))
    graph.add_node("verifier", make_verifier_node(verifier_model))
    graph.add_node("formatter", formatter)

    graph.set_entry_point("perception")
    graph.add_edge("perception", "planner")
    graph.add_edge("planner", "orchestrator")
    
    graph.add_conditional_edges(
        "orchestrator",
        route_next,
        {"exec_s1": "exec_s1", "exec_s2": "exec_s2", "verifier": "verifier"},
    )
    
    graph.add_edge("exec_s1", "reflector")
    graph.add_edge("exec_s2", "reflector")
    graph.add_edge("reflector", "orchestrator")
    
    graph.add_conditional_edges(
        "verifier",
        verifier_decision,
        {"planner": "planner", "formatter": "formatter"},
    )
    graph.add_edge("formatter", END)

    return graph.compile()
