"""Dev script: run GaiaAgent on a small slice of real GAIA questions.

Usage:
    python scripts/dev_run.py --limit 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaia_agent.config import Config
from gaia_agent.runner import run_agent_on_questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="Number of questions to run")
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Optional path to a local JSON list of questions (skips API fetch)",
    )
    args = parser.parse_args()

    # Import here so config/env issues surface before heavy model loads
    from app import GaiaAgent  # noqa: WPS433

    cfg = Config.from_env()
    agent = GaiaAgent(cfg)

    if args.questions_file:
        questions = json.loads(Path(args.questions_file).read_text())
    else:
        questions = agent.client.get_questions()

    questions = questions[: args.limit]
    print(f"Running agent on {len(questions)} question(s)...")

    answers = run_agent_on_questions(agent.graph, questions, cfg.checkpoint_dir)

    for q, a in zip(questions, answers):
        print("-" * 60)
        print(f"task_id: {q.get('task_id')}")
        print(f"Q: {q.get('question')}")
        print(f"A: {a.get('submitted_answer')}")


if __name__ == "__main__":
    main()
