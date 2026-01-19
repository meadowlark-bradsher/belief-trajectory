# MAST Failure Modes (adapted for 20Q trajectories)

This toolkit uses the **MAST (Multi-Agent System Failure Taxonomy)** failure-mode vocabulary introduced in:
- Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* (arXiv:2503.13657, 2025) :contentReference[oaicite:1]{index=1}
- Project page / blog: :contentReference[oaicite:2]{index=2}

MAST was designed for **multi-agent LLM systems**, but many failure modes translate cleanly to single-agent settings when interpreted as failures of:
- **specification adherence** (FM-1.x),
- **belief management and uncertainty handling** (FM-2.x),
- **verification and termination** (FM-3.x). :contentReference[oaicite:3]{index=3}

In this project, we use MAST primarily as a **design and labeling vocabulary** for 20Q belief trajectories.

---

## Overview

| Category | Focus | 20Q interpretation |
|----------|-------|-------------------|
| FM-1.x | Specification Issues | Invalid / rule-violating actions relative to the feasible set and task protocol |
| FM-2.x | Belief / Misalignment | Miscalibrated uncertainty, inconsistent belief reporting, or mismatch handling |
| FM-3.x | Task Verification | Premature stop, insufficient checks, or incorrect “verified” claims |

---

## Failure Modes used in this toolkit

### FM-1: Specification Failures

| Mode | Description | Typical trajectory |
|------|-------------|-------------------|
| FM-1.1 | Disobey task specification | T8 (if you treat “guess not in feasible set” as spec violation) |
| FM-1.3 | Step repetition / redundancy | T4 |
| FM-1.5 | Unaware of termination conditions | T3-U (overlay) |

### FM-2: Belief Failures

| Mode | Description | Typical trajectory |
|------|-------------|-------------------|
| FM-2.2 | Multi-modal ambiguity | T5 |
| FM-2.6 | Reasoning–action mismatch | T2, T6 (requires prediction overlay) |

### FM-3: Verification Failures

| Mode | Description | Typical trajectory |
|------|-------------|-------------------|
| FM-3.1 | Premature termination | T3-P, T7-P (requires termination overlay) |
| FM-3.2 | Incomplete verification | T7-V (requires a “skipped check” overlay) |
| FM-3.3 | Incorrect verification | T8 (requires verification claim + wrong outcome) |

---

## Mapping trajectories to modes (high level)

```

T1 (baseline)     → control (no failure mode)
T2 (collapse+shock)→ FM-2.6 when prediction overlay is present and mismatch is induced
T3 (plateau)      → FM-3.1 (premature stop) or FM-1.5 (unaware) depending on termination overlay
T4 (redundant)    → FM-1.3 (world-driven low-IG repetition)
T5 (multi-modal)  → FM-2.2 (world-driven ambiguity regime)
T6 (mismatch)     → FM-2.6 (prediction/answer mismatch under calibration protocol)
T7 (late shock)   → FM-3.1 and/or FM-3.2 depending on termination/verification overlays
T8 (wrong verify) → FM-3.3 (explicit verification_claim + wrong guess)

````

---

## Overlay requirements (what must be present in the JSON)

| Mode | Required fields | Typical tags |
|------|------------------|-------------|
| FM-1.5 | continue even when termination is expected (e.g., |S| ≤ threshold) | `term:unaware` |
| FM-2.6 | prediction present + mismatch event | `pred:calibrated_argmax` (or another named pred policy) |
| FM-3.1 | stop/guess attempt at high entropy with stop_accepted=false | `term:premature_stop` |
| FM-3.2 | guess without consistency check (or check explicitly skipped) | `verify:skipped` + `term:wrong_guess` (or `term:guess`) |
| FM-3.3 | verification_claim present + wrong guess | `verify:claim_present` + `term:wrong_guess` |

World-driven modes (can be assigned without overlays if you choose):
- FM-1.3 (T4 world enforces low-IG streak)
- FM-2.2 (T5 world enforces ambiguity regime)

---

## Deriving modes from trajectories

We derive `target_mast_modes` from `trajectory_type` and `overlay_tags`.
The goal is to keep the labeling **mechanical and auditable**.

```python
def derive_mast_modes(trajectory_type: str, overlay_tags: list[str]) -> list[str]:
    tags = set(overlay_tags or [])
    modes: list[str] = []

    # World-driven modes
    if trajectory_type == "T4":
        modes.append("FM-1.3")
    if trajectory_type == "T5":
        modes.append("FM-2.2")

    # Belief mismatch modes require explicit prediction instrumentation
    if trajectory_type in ("T2", "T6") and any(t.startswith("pred:") for t in tags):
        modes.append("FM-2.6")

    # Termination / verification modes
    if trajectory_type in ("T3", "T7") and "term:premature_stop" in tags:
        modes.append("FM-3.1")

    if trajectory_type == "T7" and "verify:skipped" in tags:
        modes.append("FM-3.2")

    if trajectory_type == "T8" and "verify:claim_present" in tags and "term:wrong_guess" in tags:
        modes.append("FM-3.3")

    return modes
````

---

## Citation (MAST)

If you use MAST in analysis or documentation, please cite:

```bibtex
@article{cemri2025multi,
  title={Why Do Multi-Agent LLM Systems Fail?},
  author={Cemri, Mert and Pan, Melissa Z and Yang, Shuyi and Agrawal, Lakshya A and Chopra, Bhavya and Tiwari, Rishabh and Keutzer, Kurt and Parameswaran, Aditya and Klein, Dan and Ramchandran, Kannan and others},
  journal={arXiv preprint arXiv:2503.13657},
  year={2025}
}
```