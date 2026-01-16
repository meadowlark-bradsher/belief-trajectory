#!/usr/bin/env python3
"""CLI entry point for belief trajectory generation."""

import argparse
import random
import sys
from pathlib import Path

# Add the belief-trajectory directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import load_cuq_dataset
from src.generators import SecretFirstGenerator, PathFirstGenerator
from src.generators.archetypes import ARCHETYPE_CONSTRAINTS
from src.overlays import (
    # Prediction overlays
    CalibratedOverlay,
    OverconfidentOverlay,
    AlwaysYesOverlay,
    StickyOverlay,
    CommitEarlyOverlay,
    RefusesReviseOverlay,
    # Termination overlays
    RationalTerminationOverlay,
    PrematureStopOverlay,
    UnawareTerminationOverlay,
    WrongGuessOverlay,
    # Chain
    OverlayChain,
)
from src.storage import save_trajectory, save_batch, save_batch_summary, load_trajectory
from src.config import GeneratorConfig, OutputConfig
from src.validators import validate_trajectory, validate_overlay_tags, GATE_VALIDATORS


# Trajectory types best suited for each generation mode
SECRET_FIRST_TYPES = {"T1", "T6"}
PATH_FIRST_TYPES = {"T2", "T3", "T4", "T5", "T7", "T8"}  # T4, T7 need path_first for IG control

# Default distribution of trajectory types (percentages)
DEFAULT_DISTRIBUTION = {
    "T1": 15,
    "T2": 15,
    "T3": 15,
    "T4": 10,
    "T5": 15,
    "T6": 10,
    "T7": 10,
    "T8": 10,
}


PREDICTION_OVERLAYS = {
    "calibrated": CalibratedOverlay,
    "overconfident": OverconfidentOverlay,
    "always_yes": AlwaysYesOverlay,
    "sticky": StickyOverlay,
    "commit_early": CommitEarlyOverlay,
    "refuses_revise": RefusesReviseOverlay,
}

TERMINATION_OVERLAYS = {
    "rational": RationalTerminationOverlay,
    "premature_stop": PrematureStopOverlay,
    "unaware": UnawareTerminationOverlay,
    "wrong_guess": WrongGuessOverlay,
}


def create_overlay_chain(
    prediction_names: list[str],
    termination_names: list[str],
    items: list[str],
    secret_index: int,
    skip_prediction: bool = False
) -> OverlayChain:
    """Create an overlay chain from names."""
    pred_overlays = []
    for i, name in enumerate(prediction_names):
        overlay_cls = PREDICTION_OVERLAYS.get(name)
        if overlay_cls:
            pred_overlays.append(overlay_cls(priority=100 - i))

    term_overlays = []
    for i, name in enumerate(termination_names):
        overlay_cls = TERMINATION_OVERLAYS.get(name)
        if overlay_cls:
            term_overlays.append(overlay_cls(priority=100 - i))

    return OverlayChain(
        prediction_overlays=pred_overlays,
        termination_overlays=term_overlays,
        items=items,
        secret_index=secret_index,
        skip_prediction=skip_prediction
    )


def cmd_single(args):
    """Generate a single trajectory."""
    # Load dataset
    dataset = load_cuq_dataset(
        questions_path=args.questions,
        items_path=args.items,
    )
    print(f"Loaded {dataset.num_questions} questions, {dataset.num_items} items")

    # Choose generator based on type
    config = GeneratorConfig(seed=args.seed)
    if args.type in SECRET_FIRST_TYPES:
        generator = SecretFirstGenerator(dataset, config, seed=args.seed)
    else:
        generator = PathFirstGenerator(dataset, config, seed=args.seed)

    # Get secret index if specified by name
    secret_index = None
    if args.secret:
        secret_index = dataset.get_item_index(args.secret)
        if secret_index is None:
            print(f"Error: Unknown secret '{args.secret}'")
            sys.exit(1)

    # Generate trajectory
    trajectory = generator.generate(
        trajectory_type=args.type,
        secret_index=secret_index,
        max_turns=args.max_turns,
    )

    # Apply overlays
    pred_overlays = args.overlay or []
    term_overlays = args.termination or []
    skip_prediction = getattr(args, 'no_prediction', False)
    if pred_overlays or term_overlays or skip_prediction:
        chain = create_overlay_chain(
            prediction_names=pred_overlays,
            termination_names=term_overlays,
            items=dataset.items,
            secret_index=trajectory.secret_index,
            skip_prediction=skip_prediction
        )
        chain.apply_to_trajectory(trajectory, items=dataset.items)

    # Save or print
    if args.output_dir:
        output_config = OutputConfig(include_bitmasks=args.include_bitmasks)
        filepath = save_trajectory(trajectory, args.output_dir, output_config)
        print(f"Saved to {filepath}")
    else:
        # Print summary
        print(f"\nTrajectory: {trajectory.trajectory_id}")
        print(f"Type: {trajectory.trajectory_type} ({ARCHETYPE_CONSTRAINTS[args.type].description})")
        print(f"Mode: {trajectory.generation_mode}")
        print(f"Secret: {trajectory.secret} (index {trajectory.secret_index})")
        print(f"MAST modes: {trajectory.target_mast_modes}")
        print(f"Turns: {len(trajectory.turns)}")
        print()

        for turn in trajectory.turns:
            # Action string
            action_str = ""
            if turn.model_action == "guess":
                correct = "correct" if turn.guess_correct else "WRONG"
                action_str = f" | GUESS: {turn.guess.secret} ({correct})"
            elif turn.model_action == "stop":
                action_str = f" | STOP: {turn.stop_reason}"

            # Prediction string
            pred_str = ""
            if turn.prediction:
                ans = "YES" if turn.prediction.predicted_answer else "NO"
                pred_str = f" | Pred: {ans} ({turn.prediction.confidence:.2f})"
            actual = "YES" if turn.answer else "NO"
            print(
                f"  T{turn.turn}: {turn.question[:50]}... "
                f"-> {actual} "
                f"(split={turn.split_ratio:.2f}, entropy {turn.entropy_before:.2f}->{turn.entropy_after:.2f})"
                f"{pred_str}{action_str}"
            )


def cmd_batch(args):
    """Generate a batch of trajectories with validation stats."""
    # Load dataset
    dataset = load_cuq_dataset(
        questions_path=args.questions,
        items_path=args.items,
    )
    print(f"Loaded {dataset.num_questions} questions, {dataset.num_items} items")

    # Determine types to generate
    if args.types:
        types_to_gen = args.types
    else:
        types_to_gen = list(DEFAULT_DISTRIBUTION.keys())

    # Compute counts per type
    if args.distribution == "uniform":
        count_per_type = args.count // len(types_to_gen)
        type_counts = {t: count_per_type for t in types_to_gen}
    else:  # mast_weighted
        total_weight = sum(DEFAULT_DISTRIBUTION[t] for t in types_to_gen)
        type_counts = {
            t: int(args.count * DEFAULT_DISTRIBUTION[t] / total_weight)
            for t in types_to_gen
        }

    # Create generators
    config = GeneratorConfig(seed=args.seed)
    rng = random.Random(args.seed)

    secret_first = SecretFirstGenerator(dataset, config, seed=args.seed)
    path_first = PathFirstGenerator(dataset, config, seed=args.seed)

    # Overlay options
    pred_overlays = args.overlay or []
    term_overlays = args.termination or []
    skip_prediction = getattr(args, 'no_prediction', False)
    use_overlays = bool(pred_overlays or term_overlays or skip_prediction)

    # Stats tracking
    stats = {ttype: {
        "requested": 0,
        "generated": 0,
        "gate_passed": 0,
        "gate_failed": 0,
        "gen_failures": 0,
        "resamples": 0,
        "entropy_curves": [],  # list of (initial, final) tuples
        "feasible_by_turn": {},  # turn -> list of |S| values
    } for ttype in types_to_gen}

    # Generate trajectories
    trajectories = []
    max_resamples = getattr(args, 'max_resamples', 5)

    for ttype, count in type_counts.items():
        generator = secret_first if ttype in SECRET_FIRST_TYPES else path_first
        stats[ttype]["requested"] = count
        print(f"Generating {count} {ttype} trajectories...")

        generated = 0
        attempts = 0
        while generated < count and attempts < count * (max_resamples + 1):
            attempts += 1
            try:
                trajectory = generator.generate(
                    trajectory_type=ttype,
                    max_turns=args.max_turns,
                )

                if use_overlays:
                    chain = create_overlay_chain(
                        prediction_names=pred_overlays,
                        termination_names=term_overlays,
                        items=dataset.items,
                        secret_index=trajectory.secret_index,
                        skip_prediction=skip_prediction
                    )
                    chain.apply_to_trajectory(trajectory, items=dataset.items)

                # Validate against gate
                result = validate_trajectory(trajectory)

                if result.passed or not getattr(args, 'strict', False):
                    trajectories.append(trajectory)
                    generated += 1
                    stats[ttype]["generated"] += 1

                    if result.passed:
                        stats[ttype]["gate_passed"] += 1
                    else:
                        stats[ttype]["gate_failed"] += 1

                    # Collect entropy stats
                    if trajectory.turns:
                        stats[ttype]["entropy_curves"].append(
                            (trajectory.initial_entropy, trajectory.final_entropy)
                        )
                        for turn in trajectory.turns:
                            turn_num = turn.turn
                            if turn_num not in stats[ttype]["feasible_by_turn"]:
                                stats[ttype]["feasible_by_turn"][turn_num] = []
                            stats[ttype]["feasible_by_turn"][turn_num].append(
                                turn.feasible_set_size_after
                            )

                    if generated % 50 == 0:
                        print(f"  Generated {generated}/{count}")
                else:
                    # Gate failed in strict mode, resample
                    stats[ttype]["resamples"] += 1

            except RuntimeError as e:
                stats[ttype]["gen_failures"] += 1
                if attempts % 10 == 0:
                    print(f"  Warning: Generation failure ({attempts} attempts): {e}")

        if generated < count:
            print(f"  Warning: Only generated {generated}/{count} for {ttype}")

    print(f"\nGenerated {len(trajectories)} trajectories total")

    # Print stats summary
    print("\n" + "=" * 60)
    print("GENERATION STATS")
    print("=" * 60)
    print(f"{'Type':<6} {'Req':>5} {'Gen':>5} {'Pass':>5} {'Fail':>5} {'Resamp':>6} {'Errors':>6}")
    print("-" * 60)
    total_pass = 0
    total_gen = 0
    for ttype in types_to_gen:
        s = stats[ttype]
        total_pass += s["gate_passed"]
        total_gen += s["generated"]
        print(f"{ttype:<6} {s['requested']:>5} {s['generated']:>5} {s['gate_passed']:>5} "
              f"{s['gate_failed']:>5} {s['resamples']:>6} {s['gen_failures']:>6}")
    print("-" * 60)
    pass_rate = (total_pass / total_gen * 100) if total_gen > 0 else 0
    print(f"{'TOTAL':<6} {sum(type_counts.values()):>5} {total_gen:>5} {total_pass:>5} "
          f"{total_gen - total_pass:>5}")
    print(f"\nOverall gate pass rate: {pass_rate:.1f}%")

    # Print entropy stats
    print("\n" + "=" * 60)
    print("ENTROPY CURVES (initial -> final)")
    print("=" * 60)
    for ttype in types_to_gen:
        curves = stats[ttype]["entropy_curves"]
        if curves:
            avg_init = sum(c[0] for c in curves) / len(curves)
            avg_final = sum(c[1] for c in curves) / len(curves)
            print(f"{ttype}: {avg_init:.2f} -> {avg_final:.2f} bits (avg over {len(curves)} trajectories)")

    # Print |S| distribution by turn
    print("\n" + "=" * 60)
    print("FEASIBLE SET SIZE BY TURN (median)")
    print("=" * 60)
    for ttype in types_to_gen:
        fbt = stats[ttype]["feasible_by_turn"]
        if fbt:
            turn_medians = []
            for turn_num in sorted(fbt.keys()):
                values = sorted(fbt[turn_num])
                median = values[len(values) // 2]
                turn_medians.append(f"T{turn_num}:{median}")
            print(f"{ttype}: {' '.join(turn_medians[:8])}{'...' if len(turn_medians) > 8 else ''}")

    # Save
    output_config = OutputConfig(include_bitmasks=args.include_bitmasks)
    output_dir = Path(args.output_dir)

    batch_path = save_batch(trajectories, output_dir, output_config)
    print(f"\nSaved batch to {batch_path}")

    summary_path = save_batch_summary(trajectories, output_dir)
    print(f"Saved summary to {summary_path}")

    # Save stats
    import json
    stats_path = output_dir / "generation_stats.json"
    # Convert stats to serializable format
    stats_serializable = {}
    for ttype, s in stats.items():
        stats_serializable[ttype] = {
            k: v for k, v in s.items()
            if k not in ("entropy_curves", "feasible_by_turn")
        }
        if s["entropy_curves"]:
            stats_serializable[ttype]["avg_initial_entropy"] = sum(c[0] for c in s["entropy_curves"]) / len(s["entropy_curves"])
            stats_serializable[ttype]["avg_final_entropy"] = sum(c[1] for c in s["entropy_curves"]) / len(s["entropy_curves"])
    with open(stats_path, "w") as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"Saved stats to {stats_path}")


def cmd_validate(args):
    """Validate existing trajectories against their gates."""
    import json

    input_path = Path(args.input)
    trajectories = []

    if input_path.is_file():
        # Single file
        trajectory = load_trajectory(input_path)
        trajectories.append((input_path, trajectory))
    elif input_path.is_dir():
        # Directory of files
        for filepath in sorted(input_path.glob("*.json")):
            if filepath.name.startswith("batch_") or filepath.name.endswith("_summary.json"):
                continue
            try:
                trajectory = load_trajectory(filepath)
                trajectories.append((filepath, trajectory))
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")

    if not trajectories:
        print("No trajectories found")
        return

    print(f"Validating {len(trajectories)} trajectories...\n")

    # Validate each
    results_by_type = {}
    for filepath, trajectory in trajectories:
        ttype = trajectory.trajectory_type
        if ttype not in results_by_type:
            results_by_type[ttype] = {"passed": 0, "failed": 0, "failures": []}

        result = validate_trajectory(trajectory)
        if result.passed:
            results_by_type[ttype]["passed"] += 1
            if args.verbose:
                print(f"  PASS: {filepath.name}")
        else:
            results_by_type[ttype]["failed"] += 1
            results_by_type[ttype]["failures"].append((filepath.name, result.reason))
            if args.verbose or not args.summary_only:
                print(f"  FAIL: {filepath.name} - {result.reason}")

    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"{'Type':<6} {'Pass':>6} {'Fail':>6} {'Rate':>8}")
    print("-" * 50)
    total_pass = 0
    total_fail = 0
    for ttype in sorted(results_by_type.keys()):
        r = results_by_type[ttype]
        total = r["passed"] + r["failed"]
        rate = r["passed"] / total * 100 if total > 0 else 0
        total_pass += r["passed"]
        total_fail += r["failed"]
        print(f"{ttype:<6} {r['passed']:>6} {r['failed']:>6} {rate:>7.1f}%")
    print("-" * 50)
    total = total_pass + total_fail
    rate = total_pass / total * 100 if total > 0 else 0
    print(f"{'TOTAL':<6} {total_pass:>6} {total_fail:>6} {rate:>7.1f}%")

    # Show failure details if requested
    if args.show_failures:
        print("\n" + "=" * 50)
        print("FAILURE DETAILS")
        print("=" * 50)
        for ttype in sorted(results_by_type.keys()):
            failures = results_by_type[ttype]["failures"]
            if failures:
                print(f"\n{ttype}:")
                for fname, reason in failures[:5]:
                    print(f"  {fname}: {reason}")
                if len(failures) > 5:
                    print(f"  ... and {len(failures) - 5} more")


def cmd_list_types(args):
    """List available trajectory types."""
    print("Available trajectory types:\n")
    for ttype, constraints in ARCHETYPE_CONSTRAINTS.items():
        mode = "secret_first" if ttype in SECRET_FIRST_TYPES else "path_first"
        print(f"  {ttype}: {constraints.description}")
        print(f"      Mode: {mode}, Turns: {constraints.min_turns}-{constraints.max_turns}")
        print()


def cmd_list_items(args):
    """List available items (secrets)."""
    dataset = load_cuq_dataset(
        questions_path=args.questions,
        items_path=args.items,
    )
    print(f"Available items ({dataset.num_items} total):\n")
    for i, item in enumerate(dataset.items):
        print(f"  {i:3d}: {item}")


def main():
    parser = argparse.ArgumentParser(
        description="Belief Trajectory Generator for 20 Questions stress testing"
    )
    parser.add_argument(
        "--questions",
        default="data/questions.jsonl",
        help="Path to questions.jsonl"
    )
    parser.add_argument(
        "--items",
        default="data/items.txt",
        help="Path to items.txt"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single trajectory
    single_parser = subparsers.add_parser("single", help="Generate a single trajectory")
    single_parser.add_argument(
        "--type", "-t",
        required=True,
        choices=list(ARCHETYPE_CONSTRAINTS.keys()),
        help="Trajectory type (T1-T8)"
    )
    single_parser.add_argument(
        "--secret", "-s",
        help="Secret item (randomly chosen if not specified)"
    )
    single_parser.add_argument(
        "--overlay", "-o",
        nargs="+",
        choices=["calibrated", "overconfident", "always_yes", "sticky", "commit_early", "refuses_revise"],
        help="Prediction overlays to apply"
    )
    single_parser.add_argument(
        "--no-prediction",
        action="store_true",
        help="Skip prediction overlay (world-only baseline)"
    )
    single_parser.add_argument(
        "--termination", "-T",
        nargs="+",
        choices=["rational", "premature_stop", "unaware", "wrong_guess"],
        help="Termination overlays to apply"
    )
    single_parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum turns (uses type default if not specified)"
    )
    single_parser.add_argument(
        "--output-dir",
        help="Directory to save output (prints summary if not specified)"
    )
    single_parser.add_argument(
        "--include-bitmasks",
        action="store_true",
        help="Include hex bitmasks in output"
    )
    single_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    single_parser.set_defaults(func=cmd_single)

    # Batch generation
    batch_parser = subparsers.add_parser("batch", help="Generate a batch of trajectories")
    batch_parser.add_argument(
        "--count", "-n",
        type=int,
        default=100,
        help="Total number of trajectories to generate"
    )
    batch_parser.add_argument(
        "--types",
        nargs="+",
        choices=list(ARCHETYPE_CONSTRAINTS.keys()),
        help="Specific types to generate (all if not specified)"
    )
    batch_parser.add_argument(
        "--distribution",
        choices=["uniform", "mast_weighted"],
        default="mast_weighted",
        help="How to distribute types"
    )
    batch_parser.add_argument(
        "--overlay", "-o",
        nargs="+",
        choices=["calibrated", "overconfident", "always_yes", "sticky", "commit_early", "refuses_revise"],
        help="Prediction overlays to apply"
    )
    batch_parser.add_argument(
        "--no-prediction",
        action="store_true",
        help="Skip prediction overlay (world-only baseline)"
    )
    batch_parser.add_argument(
        "--termination", "-T",
        nargs="+",
        choices=["rational", "premature_stop", "unaware", "wrong_guess"],
        help="Termination overlays to apply"
    )
    batch_parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum turns per trajectory"
    )
    batch_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save output"
    )
    batch_parser.add_argument(
        "--include-bitmasks",
        action="store_true",
        help="Include hex bitmasks in output"
    )
    batch_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    batch_parser.add_argument(
        "--strict",
        action="store_true",
        help="Resample if gate validation fails"
    )
    batch_parser.add_argument(
        "--max-resamples",
        type=int,
        default=5,
        help="Max resamples per trajectory in strict mode"
    )
    batch_parser.set_defaults(func=cmd_batch)

    # Validate trajectories
    validate_parser = subparsers.add_parser("validate", help="Validate existing trajectories")
    validate_parser.add_argument(
        "input",
        help="Path to trajectory file or directory"
    )
    validate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all results (not just failures)"
    )
    validate_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary, not individual failures"
    )
    validate_parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Show detailed failure reasons"
    )
    validate_parser.set_defaults(func=cmd_validate)

    # List types
    list_types_parser = subparsers.add_parser("list-types", help="List trajectory types")
    list_types_parser.set_defaults(func=cmd_list_types)

    # List items
    list_items_parser = subparsers.add_parser("list-items", help="List available items")
    list_items_parser.set_defaults(func=cmd_list_items)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
