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
import dataclasses
import logging
import os
import sys
from pathlib import Path

# Make `app.py` and `src` importable whether invoked from repo root or inside the Space.
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

try:
    from dotenv import load_dotenv  # noqa: E402
    
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    
except ImportError:
    pass

from gaia_agent.config import Config  # noqa: E402
from gaia_agent.gaia_dataset import GaiaDatasetClient  # noqa: E402
from gaia_agent.runner import run_agent_on_questions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to run. Set to -1 for all.")
    parser.add_argument("--level", type=str, default="1")
    parser.add_argument("--verbose", action="store_true", help="Enable LangChain debug logging")
    parser.add_argument("--config", type=str, default="2023_all")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--local", action="store_true", help="Run with local model defined in .env")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="Model to run when not using --local (defaults to gemini provider)")
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run specific task_id(s) (comma-separated, ignores --limit/--level)",
    )
    parser.add_argument("--rerun-failed", action="store_true", help="Automatically rerun tasks that failed in the last run (from .last_failures.txt)")
    parser.add_argument("--force", action="store_true", help="Delete existing checkpoints for selected tasks before running")
    parser.add_argument("--cavemen", action="store_true", help="Enable Caveman Mode (ultra-terse communication)")
    parser.add_argument("--caveman-mode", type=str, default="full", choices=["lite", "full", "ultra", "wenyan-lite", "wenyan-full", "wenyan-ultra"], help="Caveman Mode intensity level")
    parser.add_argument("--gemma4", action=argparse.BooleanOptionalAction, default=None, help="Use gemma-4-31b-it for all model tiers (completely free on Gemini API). CLI flag overrides environment variables.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # suppress noisy third-party loggers unless verbose
    if not args.verbose:
        for noisy in ("httpx", "httpcore", "langchain_core", "openai", "urllib3", "google_genai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    token = os.getenv("HF_TOKEN") or os.getenv("GAIA_HUGGINGFACE_API_KEY")

    # Import GaiaAgent lazily — it pulls heavy LLM deps.
    from app import GaiaAgent  # noqa: WPS433

    cfg = Config.from_env()

    # Default to local-env behavior. The --model flag will only be used if GAIA_STRONG_MODEL is not set.
    if args.model != "gemini-3-flash-preview" or not cfg.strong_model:
        cfg = dataclasses.replace(
            cfg,
            cheap_provider="google",
            cheap_model=args.model,
            strong_provider="google",
            strong_model=args.model
        )

    # Manual override for the user to be absolutely sure
    if args.local:
        cfg = Config.from_env()

    # High-Performance Free Testing Shortcut
    # Hierarchy: CLI Flag (Explicit) > Environment Variable > Default (False)
    if args.gemma4 is not None:
        use_gemma4 = args.gemma4
        source = "CLI: --gemma4" if args.gemma4 else "CLI: --no-gemma4"
    else:
        use_gemma4 = os.getenv("USE_GEMMA_4", "false").lower() == "true"
        source = "ENV: USE_GEMMA_4=TRUE" if use_gemma4 else None

    if use_gemma4:
        banner_source = source or "DEFAULT: OFF"
        print("\n" + "!"*80, flush=True)
        print(f"!!! OVERRIDING ALL TIERED MODELS TO gemma-4-31b-it ({source}) !!!".center(80), flush=True)
        print("!"*80 + "\n", flush=True)
        cfg = dataclasses.replace(
            cfg,
            cheap_provider="google",
            cheap_model="gemma-4-31b-it",
            strong_provider="google",
            strong_model="gemma-4-31b-it",
            extra_strong_provider="google",
            extra_strong_model="gemma-4-31b-it"
        )

    # Apply Caveman flags
    cfg = dataclasses.replace(
        cfg,
        caveman=args.cavemen if args.cavemen else cfg.caveman,
        caveman_mode=args.caveman_mode if args.caveman_mode != "full" else cfg.caveman_mode
    )
    limit = args.limit if args.limit > 0 else None
    
    # Handle task_ids / rerun-failed
    target_ids = []
    if args.rerun_failed:
        fail_file = Path(".last_failures.txt")
        if fail_file.exists():
            target_ids = fail_file.read_text().strip().split(",")
            target_ids = [tid.strip() for tid in target_ids if tid.strip()]
            print(f"Loaded {len(target_ids)} failed IDs from {fail_file}")
        else:
            print("No .last_failures.txt found. Skipping rerun.")

    if args.task_id:
        target_ids.extend([tid.strip() for tid in args.task_id.split(",") if tid.strip()])

    client = GaiaDatasetClient(
        config=args.config,
        split=args.split,
        level=None if target_ids else args.level,
        limit=None if target_ids else limit,
        token=token,
    )

    questions = client.get_questions()
    if target_ids:
        questions = [q for q in questions if q["task_id"] in target_ids]

    if not questions:
        print("No matching questions.")
        return

    # Delete checkpoints if --force is requested
    if args.force:
        checkpoint_root = Path(cfg.checkpoint_dir)
        for q in questions:
            checkpoint_path = checkpoint_root / f"{q['task_id']}.json"
            if checkpoint_path.exists():
                print(f"Forcing rerun: deleting checkpoint {checkpoint_path}")
                checkpoint_path.unlink()

    agent = GaiaAgent(cfg=cfg, client=client)

    print(f"Running agent on {len(questions)} question(s)...")
    answers = run_agent_on_questions(agent.graph, questions, cfg.checkpoint_dir)
    answers_by_id = {a["task_id"]: a["submitted_answer"] for a in answers}

    correct = 0
    failed_ids = []

    # Normalize: case-insensitive, strip whitespace, hyphens->spaces
    # (matches GAIA's quasi-exact-match with punctuation normalization)
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace("-", " ")

    for q in questions:
        tid = q["task_id"]
        got = answers_by_id.get(tid, "")
        expected = q.get("expected_answer", "")
        ok = _norm(got) == _norm(expected)
        if ok:
            correct += 1
        else:
            failed_ids.append(tid)

        print("-" * 70)
        print(f"task_id : {tid}")
        print(f"level   : {q.get('Level')}")
        print(f"Q       : {q['question'][:200]}")
        print(f"expected: {expected}")
        print(f"got     : {got}")
        print(f"match   : {ok}")

    print("=" * 70)
    print(f"Score: {correct}/{len(questions)}")
    
    if failed_ids:
        ids_str = ",".join(failed_ids)
        print(f"\nFailed Task IDs (for easy rerun):")
        print(ids_str)
        Path(".last_failures.txt").write_text(ids_str)
        print(f"\nWritten to .last_failures.txt")
    elif target_ids:
        # All targets passed!
        if Path(".last_failures.txt").exists():
             Path(".last_failures.txt").unlink()
             print("\nCleared .last_failures.txt as all requested tasks passed.")


if __name__ == "__main__":
    main()
