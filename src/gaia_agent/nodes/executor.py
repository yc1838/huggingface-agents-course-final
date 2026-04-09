from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.prompts import EXECUTOR_SYSTEM
from gaia_agent.state import AgentState, Observation

DRAFT_PREFIX = "DRAFT:"


def _format_context(state: AgentState) -> str:
    lines = [f"Question: {state['question']}"]
    if state["file_path"]:
        lines.append(f"File: {state['file_path']} ({state['modality']})")
    lines.append("")
    lines.append("Plan:")
    for index, step in enumerate(state["plan"]):
        marker = ">>" if index == state["step_idx"] else "  "
        lines.append(f"{marker} {index}. [{step['tier']}] {step['description']}")
    if state["observations"]:
        lines.append("")
        lines.append("Prior observations:")
        for observation in state["observations"]:
            lines.append(
                f"- step {observation['step_idx']} [{observation['tool']}]: {observation['result']}"
            )
    return "\n".join(lines)


def make_executor_node(model, tools):
    tools_by_name = {tool.name: tool for tool in tools}
    bound_model = model.bind_tools(tools)

    def executor(state: AgentState) -> dict:
        response = bound_model.invoke(
            [
                SystemMessage(content=EXECUTOR_SYSTEM),
                HumanMessage(content=_format_context(state)),
            ]
        )

        observations: list[Observation] = list(state["observations"])
        draft_answer = state["draft_answer"]
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            for tool_call in tool_calls:
                name = tool_call["name"]
                args = tool_call.get("args", {})
                result = tools_by_name[name].invoke(args)
                observations.append(
                    {
                        "step_idx": state["step_idx"],
                        "tool": name,
                        "args": args,
                        "result": str(result),
                    }
                )
        else:
            content = response.content if isinstance(response.content, str) else str(response.content)
            stripped = content.strip()
            if stripped.upper().startswith(DRAFT_PREFIX):
                draft_answer = stripped[len(DRAFT_PREFIX):].strip()
            else:
                observations.append(
                    {
                        "step_idx": state["step_idx"],
                        "tool": "reasoning",
                        "args": {},
                        "result": stripped,
                    }
                )

        return {
            "observations": observations,
            "step_idx": state["step_idx"] + 1,
            "draft_answer": draft_answer,
        }

    return executor
