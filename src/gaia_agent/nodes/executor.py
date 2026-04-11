from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from gaia_agent.llm_utils import extract_text
from gaia_agent.prompts import (
    AUDIO_SPECIALIST,
    BASE_EXECUTOR,
    FILE_SPECIALIST,
    GENERAL_EXECUTOR,
    MATH_SPECIALIST,
    RESEARCH_SPECIALIST,
    VISION_SPECIALIST,
    apply_caveman,
)
from gaia_agent.state import AgentState, Observation

log = logging.getLogger(__name__)

DRAFT_PREFIX = "DRAFT:"
_MAX_OBS_CHARS = 8000

_SPECIALISTS = {
    "math": MATH_SPECIALIST,
    "research": RESEARCH_SPECIALIST,
    "vision": VISION_SPECIALIST,
    "audio": AUDIO_SPECIALIST,
    "file": FILE_SPECIALIST,
    "general": GENERAL_EXECUTOR,
}


def _format_context(state: AgentState) -> str:
    lines = [f"Question: {state['question']}"]
    if state["file_path"]:
        lines.append(f"File: {state['file_path']} ({state['modality']})")
    
    if state["current_strategy"]:
        lines.append(f"STRATEGY: {state['current_strategy']}")
    
    lines.append("")
    lines.append("Plan:")
    for index, step in enumerate(state["plan"]):
        marker = ">>" if index == state["step_idx"] else "  "
        lines.append(f"{marker} {index}. [{step['tier']}] {step['description']}")
        if step.get("thought"):
            lines.append(f"   (Logic/Rationale: {step['thought']})")
    if state["observations"]:
        lines.append("")
        lines.append("Prior observations:")
        for observation in state["observations"]:
            result = observation["result"]
            if len(result) > _MAX_OBS_CHARS:
                result = result[:_MAX_OBS_CHARS] + "...[truncated]"
            lines.append(
                f"- step {observation['step_idx']} [{observation['tool']}]: {result}"
            )
    if state["todo_list"]:
        lines.append("")
        lines.append("DYNAMIC TODO LIST:")
        for idx, todo in enumerate(state["todo_list"]):
            lines.append(f"{idx}. {todo}")

    return "\n".join(lines)


def make_executor_node(model, tools, caveman: bool = False, caveman_mode: str = "full"):
    tools_by_name = {tool.name: tool for tool in tools}
    bound_model = model.bind_tools(tools)

    def executor(state: AgentState) -> dict:
        step = state["step_idx"]
        current_step = state["plan"][step] if step < len(state["plan"]) else {"description": "?", "tier": "?"}
        log.info("[executor] step=%d  tier=%s  desc=%s  task=%s", step, current_step["tier"], current_step["description"], state["task_id"])
        
        # Assemble dynamic system prompt
        domain = state.get("current_domain") or "general"
        specialist_prompt = _SPECIALISTS.get(domain, GENERAL_EXECUTOR)
        system_content = f"{BASE_EXECUTOR}\n\n{specialist_prompt}"
        executor_prompt = apply_caveman(system_content, caveman, caveman_mode)
        
        response = bound_model.invoke(
            [
                SystemMessage(content=executor_prompt),
                HumanMessage(content=_format_context(state)),
            ]
        )
        raw = extract_text(response.content)
        log.debug("[executor] raw response:\n%s", raw)

        observations: list[Observation] = list(state["observations"])
        draft_answer = state["draft_answer"]
        tool_calls = getattr(response, "tool_calls", None) or []
        log.info("[executor] tool_calls=%s", [tc["name"] for tc in tool_calls])
        if tool_calls:
            for tool_call in tool_calls:
                name = tool_call["name"]
                args = tool_call.get("args", {})
                log.info("[executor] calling tool=%s args=%s", name, str(args))
                result = tools_by_name[name].invoke(args)
                result_str = str(result)
                log.info("[executor] tool result (%d chars): %s", len(result_str), result_str)
                
                # Special handling for Todo tools
                todo_list = list(state.get("todo_list", []))
                if result_str.startswith("SET_TODOS:"):
                    import ast
                    try:
                        new_todos = ast.literal_eval(result_str[len("SET_TODOS:"):].strip())
                        if isinstance(new_todos, list):
                            todo_list = new_todos
                            log.info("[executor] updated todo_list: %s", todo_list)
                    except Exception as e:
                        log.error("[executor] failed to parse todos: %s", e)
                elif result_str.startswith("DONE_TODO:"):
                    try:
                        idx = int(result_str[len("DONE_TODO:"):].strip())
                        if 0 <= idx < len(todo_list):
                            # Mark as done by adding a prefix or similar
                            if not todo_list[idx].startswith("[DONE]"):
                                todo_list[idx] = f"[DONE] {todo_list[idx]}"
                                log.info("[executor] marked todo %d as done", idx)
                    except Exception as e:
                        log.error("[executor] failed to mark todo done: %s", e)

                observations.append(
                    {
                        "step_idx": state["step_idx"],
                        "tool": name,
                        "args": args,
                        "result": result_str,
                    }
                )
        else:
            content = extract_text(response.content)
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

        next_step = state["step_idx"] + 1
        # If this was the last plan step and we still have no draft,
        # ask the model (without tools) to synthesize an answer from observations.
        if draft_answer is None and next_step >= len(state["plan"]):
            log.info("[executor] all steps done, no draft — forcing synthesis")
            synthesis_prompt = (
                f"Question: {state['question']}\n\n"
            )
            if state["critique"]:
                synthesis_prompt += f"PRIOR REJECTION CRITIQUE: {state['critique']}\n\n"
            
            synthesis_prompt += (
                "Based on the observations below, give ONLY the final answer. "
                "No explanation. Just the answer. "
                "CRITICAL: Look carefully at the units requested in the Question! "
                "For example, if the question asks for 'millions of dollars', and you compute 45000000, you MUST answer '45'. "
                "Format strictly to match what the question asks for.\n\n"
            )
            for obs in observations:
                r = obs["result"]
                if len(r) > _MAX_OBS_CHARS:
                    r = r[:_MAX_OBS_CHARS] + "...[truncated]"
                synthesis_prompt += f"- [{obs['tool']}]: {r}\n"
            synth_system = apply_caveman("You produce short, exact answers. Nothing else.", caveman, caveman_mode)
            synth_response = model.invoke(
                [
                    SystemMessage(content=synth_system),
                    HumanMessage(content=synthesis_prompt),
                ]
            )
            synth_text = extract_text(synth_response.content)
            draft_answer = synth_text.strip()
            log.info("[executor] synthesised draft_answer=%r", draft_answer)

        return {
            "observations": observations,
            "step_idx": next_step,
            "draft_answer": draft_answer,
            "todo_list": todo_list,
        }

    return executor
