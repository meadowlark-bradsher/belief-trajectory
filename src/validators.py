"""Gate validators for trajectory types.

Each validator checks whether a generated trajectory meets the constraints
for its declared type. Returns (passed, reason) tuple.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .models import Trajectory


@dataclass
class ValidationResult:
    """Result of a gate validation."""
    passed: bool
    reason: Optional[str] = None
    metrics: Optional[dict] = None


# Gate parameters (tunable)
BALANCED_EPSILON = 0.15  # |p_yes - 0.5| <= epsilon for "balanced"
RARE_BRANCH_THRESHOLD = 0.25  # p_branch <= threshold for "rare"
LOW_IG_THRESHOLD = 0.15  # delta_H <= threshold for "low information gain"
PLATEAU_MIN_TURNS = 3  # minimum consecutive low-IG turns for T3/T4


def validate_t1_gate(trajectory: Trajectory) -> ValidationResult:
    """T1 gate: all turns have balanced splits until |S|=1.

    Constraint: |p_yes - 0.5| <= epsilon for all turns with |S| > 1
    """
    violations = []
    for turn in trajectory.turns:
        if turn.feasible_set_size_after > 1:
            deviation = abs(turn.split_ratio - 0.5)
            if deviation > BALANCED_EPSILON:
                violations.append(f"T{turn.turn}: split={turn.split_ratio:.3f}")

    if violations:
        return ValidationResult(
            passed=False,
            reason=f"Unbalanced splits: {', '.join(violations[:3])}{'...' if len(violations) > 3 else ''}",
            metrics={"num_violations": len(violations)}
        )
    return ValidationResult(passed=True, metrics={"num_violations": 0})


def validate_t2_gate(trajectory: Trajectory) -> ValidationResult:
    """T2 gate: early rare branch AND later rare branch with gap.

    Constraint: exists early turn (1-3) with p_branch <= threshold AND
                exists later turn (after gap) with p_branch <= threshold
    """
    early_rare = None
    later_rare = None

    for turn in trajectory.turns:
        p_branch = turn.branch_probability
        if p_branch <= RARE_BRANCH_THRESHOLD:
            if turn.turn <= 3 and early_rare is None:
                early_rare = turn.turn
            elif turn.turn >= 4 and early_rare is not None:
                # Need at least one turn gap with |S| > 2
                later_rare = turn.turn
                break

    if early_rare is None:
        return ValidationResult(
            passed=False,
            reason="No early rare branch (turns 1-3)",
            metrics={"early_rare": None, "later_rare": None}
        )
    if later_rare is None:
        return ValidationResult(
            passed=False,
            reason=f"No later rare branch after early (turn {early_rare})",
            metrics={"early_rare": early_rare, "later_rare": None}
        )

    return ValidationResult(
        passed=True,
        metrics={"early_rare": early_rare, "later_rare": later_rare}
    )


def validate_t3_gate(trajectory: Trajectory) -> ValidationResult:
    """T3 gate: K consecutive low-IG turns, then balanced resolution.

    Constraint: >= PLATEAU_MIN_TURNS with delta_H <= threshold,
                then >= 1 turn with balanced split
    """
    # Find plateau (consecutive low-IG turns)
    plateau_start = None
    plateau_length = 0
    max_plateau_length = 0

    for i, turn in enumerate(trajectory.turns):
        delta_h = turn.entropy_before - turn.entropy_after
        if delta_h <= LOW_IG_THRESHOLD:
            if plateau_start is None:
                plateau_start = i
            plateau_length = i - plateau_start + 1
            max_plateau_length = max(max_plateau_length, plateau_length)
        else:
            plateau_start = None
            plateau_length = 0

    if max_plateau_length < PLATEAU_MIN_TURNS:
        return ValidationResult(
            passed=False,
            reason=f"Plateau too short: {max_plateau_length} < {PLATEAU_MIN_TURNS}",
            metrics={"max_plateau_length": max_plateau_length}
        )

    # Check for resolution phase (balanced split after plateau)
    resolution_found = False
    for turn in trajectory.turns:
        if abs(turn.split_ratio - 0.5) <= BALANCED_EPSILON and turn.entropy_before - turn.entropy_after > LOW_IG_THRESHOLD:
            resolution_found = True
            break

    if not resolution_found:
        return ValidationResult(
            passed=False,
            reason="No balanced resolution phase after plateau",
            metrics={"max_plateau_length": max_plateau_length}
        )

    return ValidationResult(
        passed=True,
        metrics={"max_plateau_length": max_plateau_length}
    )


def validate_t4_gate(trajectory: Trajectory) -> ValidationResult:
    """T4 gate: K consecutive low-IG / no-op turns.

    Constraint: >= PLATEAU_MIN_TURNS consecutive turns with delta_H <= threshold
    """
    consecutive_low_ig = 0
    max_consecutive = 0

    for turn in trajectory.turns:
        delta_h = turn.entropy_before - turn.entropy_after
        if delta_h <= LOW_IG_THRESHOLD:
            consecutive_low_ig += 1
            max_consecutive = max(max_consecutive, consecutive_low_ig)
        else:
            consecutive_low_ig = 0

    if max_consecutive < PLATEAU_MIN_TURNS:
        return ValidationResult(
            passed=False,
            reason=f"Low-IG streak too short: {max_consecutive} < {PLATEAU_MIN_TURNS}",
            metrics={"max_consecutive_low_ig": max_consecutive}
        )

    return ValidationResult(
        passed=True,
        metrics={"max_consecutive_low_ig": max_consecutive}
    )


def validate_t5_gate(trajectory: Trajectory) -> ValidationResult:
    """T5 gate: NOT all balanced, require skewed mid-game segment.

    Constraint: forbid "all turns balanced" (that's T1)
                require at least one mid-segment (|S| in [8,32]) with skewed split
    """
    # Count balanced vs skewed turns
    balanced_count = 0
    skewed_mid_count = 0

    for turn in trajectory.turns:
        is_balanced = abs(turn.split_ratio - 0.5) <= BALANCED_EPSILON
        is_mid_game = 8 <= turn.feasible_set_size_before <= 32

        if is_balanced:
            balanced_count += 1
        if is_mid_game and not is_balanced:
            skewed_mid_count += 1

    # Fail if all balanced (that's T1)
    if balanced_count == len(trajectory.turns):
        return ValidationResult(
            passed=False,
            reason="All turns balanced - this is T1, not T5",
            metrics={"balanced_count": balanced_count, "skewed_mid_count": 0}
        )

    # Require at least one skewed mid-game turn
    if skewed_mid_count == 0:
        return ValidationResult(
            passed=False,
            reason="No skewed splits in mid-game (|S| in [8,32])",
            metrics={"balanced_count": balanced_count, "skewed_mid_count": 0}
        )

    return ValidationResult(
        passed=True,
        metrics={"balanced_count": balanced_count, "skewed_mid_count": skewed_mid_count}
    )


def validate_t6_gate(trajectory: Trajectory) -> ValidationResult:
    """T6 gate: no world constraints (optional: predictable run).

    Always passes for now. Could add optional constraint later.
    """
    return ValidationResult(passed=True, metrics={})


def validate_t7_gate(trajectory: Trajectory) -> ValidationResult:
    """T7 gate: late rare-branch shock.

    Constraint: last 1-2 turns have p_branch <= threshold
    """
    if len(trajectory.turns) < 2:
        return ValidationResult(
            passed=False,
            reason="Too few turns",
            metrics={}
        )

    # Check last 2 turns for rare branch
    late_shock_found = False
    shock_turn = None

    for turn in trajectory.turns[-2:]:
        if turn.branch_probability <= RARE_BRANCH_THRESHOLD:
            late_shock_found = True
            shock_turn = turn.turn
            break

    if not late_shock_found:
        last_probs = [t.branch_probability for t in trajectory.turns[-2:]]
        return ValidationResult(
            passed=False,
            reason=f"No late shock: last 2 branch probs = {last_probs}",
            metrics={"late_branch_probs": last_probs}
        )

    return ValidationResult(
        passed=True,
        metrics={"shock_turn": shock_turn}
    )


def validate_t8_gate(trajectory: Trajectory) -> ValidationResult:
    """T8 gate: overlay gate for FM-3.3.

    Constraint: verification_claim present + wrong guess + target_mast_modes=["FM-3.3"]
    """
    # Check for wrong guess with verification claim
    wrong_guess_with_claim = False

    for turn in trajectory.turns:
        if turn.guess is not None:
            if turn.guess_correct is False and turn.guess.verification_claim is not None:
                wrong_guess_with_claim = True
                break

    if not wrong_guess_with_claim:
        return ValidationResult(
            passed=False,
            reason="No wrong guess with verification_claim",
            metrics={"has_wrong_guess_with_claim": False}
        )

    # Check target_mast_modes
    if "FM-3.3" not in trajectory.target_mast_modes:
        return ValidationResult(
            passed=False,
            reason=f"target_mast_modes={trajectory.target_mast_modes}, expected FM-3.3",
            metrics={"has_wrong_guess_with_claim": True}
        )

    return ValidationResult(
        passed=True,
        metrics={"has_wrong_guess_with_claim": True}
    )


# Gate function registry
GATE_VALIDATORS: dict[str, Callable[[Trajectory], ValidationResult]] = {
    "T1": validate_t1_gate,
    "T2": validate_t2_gate,
    "T3": validate_t3_gate,
    "T4": validate_t4_gate,
    "T5": validate_t5_gate,
    "T6": validate_t6_gate,
    "T7": validate_t7_gate,
    "T8": validate_t8_gate,
}


def validate_trajectory(trajectory: Trajectory) -> ValidationResult:
    """Validate a trajectory against its declared type's gate."""
    validator = GATE_VALIDATORS.get(trajectory.trajectory_type)
    if validator is None:
        return ValidationResult(
            passed=False,
            reason=f"Unknown trajectory type: {trajectory.trajectory_type}"
        )
    return validator(trajectory)


def validate_overlay_tags(trajectory: Trajectory) -> ValidationResult:
    """Validate that overlay_tags are consistent with actual behavior."""
    issues = []

    # Check verify:claim_present
    has_claim = any(
        t.guess is not None and t.guess.verification_claim is not None
        for t in trajectory.turns
    )
    if has_claim and "verify:claim_present" not in trajectory.overlay_tags:
        issues.append("Missing verify:claim_present tag")
    if not has_claim and "verify:claim_present" in trajectory.overlay_tags:
        issues.append("Spurious verify:claim_present tag")

    # Check term:wrong_guess
    has_wrong_guess = any(
        t.guess is not None and t.guess_correct is False
        for t in trajectory.turns
    )
    if has_wrong_guess and "term:wrong_guess" not in trajectory.overlay_tags:
        # Could be budget guess, not necessarily wrong_guess overlay
        pass

    if issues:
        return ValidationResult(passed=False, reason="; ".join(issues))
    return ValidationResult(passed=True)
