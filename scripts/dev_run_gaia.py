"""Run GaiaAgent against a small slice of real GAIA questions.

Runs locally OR inside an HF Space. Uses the gated `gaia-benchmark/GAIA`
dataset, which is accessible without extra config when running inside an
authenticated HF Space (and with $HF_TOKEN locally).

Usage:
    python scripts/dev_run_gaia.py --limit 3 --level 1
    python scripts/dev_run_gaia.py --limit 1 --task-id c61d22de-5f6c-4958-a7f6-5e9707bd3466
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `app.py` importable whether invoked from repo root or inside the Space.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaia_agent.config import Config  # noqa: E402
from gaia_agent.gaia_dataset import GaiaDatasetClient  # noqa: E402
from gaia_agent.runner import run_agent_on_questions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--level", type=str, default="1")
    parser.add_argument("--config", type=str, default="2023_all")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run only this specific task_id (ignores --limit/--level)",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("GAIA_HUGGINGFACE_API_KEY")

    # Import GaiaAgent lazily — it pulls heavy LLM deps.
    from app import GaiaAgent  # noqa: WPS433

    cfg = Config.from_env()

    client = GaiaDatasetClient(
        config=args.config,
        split=args.split,
        level=None if args.task_id else args.level,
        limit=None if args.task_id else args.limit,
        token=token,
    )

    questions = client.get_questions()
    if args.task_id:
        questions = [q for q in questions if q["task_id"] == args.task_id]

    if not questions:
        print("No matching questions.")
        return

    agent = GaiaAgent(cfg=cfg, client=client)

    print(f"Running agent on {len(questions)} question(s)...")
    answers = run_agent_on_questions(agent.graph, questions, cfg.checkpoint_dir)
    answers_by_id = {a["task_id"]: a["submitted_answer"] for a in answers}

    correct = 0
    for q in questions:
        tid = q["task_id"]
        got = answers_by_id.get(tid, "")
        expected = q.get("expected_answer", "")
        ok = str(got).strip().lower() == str(expected).strip().lower()
        if ok:
            correct += 1
        print("-" * 70)
        print(f"task_id : {tid}")
        print(f"level   : {q.get('Level')}")
        print(f"Q       : {q['question'][:200]}")
        print(f"expected: {expected}")
        print(f"got     : {got}")
        print(f"match   : {ok}")

    print("=" * 70)
    print(f"Score: {correct}/{len(questions)}")


if __name__ == "__main__":
    main()
