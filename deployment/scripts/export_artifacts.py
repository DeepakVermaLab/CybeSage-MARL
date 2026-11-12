#!/usr/bin/env python3
"""Export verified CybSAGE research artifacts into a static web data bundle."""

from __future__ import annotations

import ast
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
DESTINATION = REPO / "deployment" / "dist" / "data" / "research-console.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value in (None, ""):
        return None
    value = float(value)
    return int(value) if value.is_integer() else value


def typed_row(row: dict[str, str], text_fields: set[str]):
    return {key: (value if key in text_fields else number(value)) for key, value in row.items()}


def action_kind(action: str) -> str:
    return action.split(" ", 1)[0] if action else "Unknown"


def build_simulation():
    episode_rows = read_csv(
        OUTPUTS / "14_cybsage_cage4_policy" / "cybsage_cage4_episode_results.csv"
    )
    trace_rows = read_csv(
        OUTPUTS / "14_cybsage_cage4_policy" / "cybsage_cage4_step_trace.csv"
    )
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in trace_rows:
        actions = ast.literal_eval(row["actions"])
        grouped[(int(row["seed"]), int(row["episode"]))].append(
            {
                "step": int(row["step"]),
                "meanReward": float(row["mean_reward"]),
                "sumReward": float(row["sum_reward"]),
                "actions": actions,
                "actionKinds": {agent: action_kind(action) for agent, action in actions.items()},
            }
        )

    episodes = []
    for row in episode_rows:
        key = (int(row["seed"]), int(row["episode"]))
        action_counts = Counter(
            kind
            for step in grouped[key]
            for kind in step["actionKinds"].values()
        )
        episodes.append(
            {
                "seed": key[0],
                "episode": key[1],
                "policy": row["policy"],
                "profile": row["profile"],
                "split": row["split"],
                "totalReward": float(row["total_reward"]),
                "meanStepReward": float(row["mean_step_reward"]),
                "steps": int(row["steps"]),
                "wallTimeSeconds": float(row["wall_time_sec"]),
                "actionCounts": dict(action_counts),
                "trace": sorted(grouped[key], key=lambda item: item["step"]),
            }
        )
    return episodes


def build_resource_profile(baselines, simulation):
    policy_path = OUTPUTS / "07_posterior_policy" / "posterior_policy.pt"
    ssl_path = OUTPUTS / "06_ssl" / "ssl_encoder.pt"
    embeddings_path = OUTPUTS / "06_ssl" / "trajectory_embeddings.npy"
    cybsage_times = [episode["wallTimeSeconds"] for episode in simulation]
    return {
        "energyInstrumented": False,
        "energyStatement": (
            "Energy consumption was not instrumented in the retained runs. The console reports "
            "wall-clock runtime and artifact size only."
        ),
        "cybsageEpisodeRuntime": {
            "meanSeconds": statistics.mean(cybsage_times),
            "medianSeconds": statistics.median(cybsage_times),
            "minSeconds": min(cybsage_times),
            "maxSeconds": max(cybsage_times),
            "episodes": len(cybsage_times),
        },
        "baselineRuntime": [
            {
                "policy": row["policy"],
                "meanSeconds": float(row["wall_time_sec_mean"]),
                "episodes": int(row["episodes"]),
            }
            for row in baselines
        ],
        "artifacts": [
            {"name": "SSL encoder", "file": "ssl_encoder.pt", "bytes": ssl_path.stat().st_size},
            {
                "name": "Posterior policy",
                "file": "posterior_policy.pt",
                "bytes": policy_path.stat().st_size,
            },
            {
                "name": "Trajectory embeddings",
                "file": "trajectory_embeddings.npy",
                "bytes": embeddings_path.stat().st_size,
            },
        ],
    }


def main():
    cage_summary = read_json(
        OUTPUTS / "14_cybsage_cage4_policy" / "cybsage_cage4_summary.json"
    )
    sci_summary = read_json(
        OUTPUTS / "16_sci_readiness_pack" / "sci_readiness_summary.json"
    )
    ssl_summary = read_json(OUTPUTS / "06_ssl" / "ssl_summary.json")
    policy_summary = read_json(OUTPUTS / "07_posterior_policy" / "policy_summary.json")
    explanation_summary = read_json(
        OUTPUTS / "08_explanations" / "explanation_summary.json"
    )
    explanations = read_json(
        OUTPUTS / "08_explanations" / "action_explanations.json"
    )
    baselines = read_csv(
        OUTPUTS / "13_cage4_baseline_rollouts" / "cage4_baseline_summary.csv"
    )
    effects = [
        typed_row(row, {"baseline", "raw_superiority"})
        for row in read_csv(
            OUTPUTS / "16_sci_readiness_pack" / "cage4_effect_sizes.csv"
        )
    ]
    profiles = [
        typed_row(row, {"profile"})
        for row in read_csv(
            OUTPUTS / "16_sci_readiness_pack" / "cage4_profile_ablation_table.csv"
        )
    ]
    ablations = [
        typed_row(row, {"method", "method_role"})
        for row in read_csv(OUTPUTS / "10_ablation_sota" / "ablation_summary.csv")
    ]
    ssl_training = [
        typed_row(row, set())
        for row in read_csv(OUTPUTS / "06_ssl" / "ssl_training_log.csv")
    ]
    policy_training = [
        typed_row(row, set())
        for row in read_csv(
            OUTPUTS / "07_posterior_policy" / "policy_training_log.csv"
        )
    ]
    simulation = build_simulation()

    bundle = {
        "meta": {
            "project": "CybSAGE-MARL",
            "title": "Adaptive Cyber-Defense Research Console",
            "generatedFrom": "Retained implementation artifacts",
            "claimBoundary": sci_summary["claim_boundary"],
            "implementationBoundary": (
                "Recorded CAGE-4 playback uses the selected alert_response_decoy profile. "
                "The explanation explorer uses the separate synthetic-trajectory prototype outputs."
            ),
        },
        "summary": {
            "selectedProfile": cage_summary["selected_profile"],
            "evalEpisodes": cage_summary["episodes"],
            "evalSeeds": cage_summary["eval_seeds"],
            "rewardMean": cage_summary["reward_mean"],
            "rewardStd": cage_summary["reward_std"],
            "baselinesCompared": sci_summary["baselines_compared"],
            "allPositiveCI": sci_summary["all_positive_ci"],
            "pairedRuns": sci_summary["min_paired_runs"],
            "trajectoryRows": policy_summary["rows"],
            "embeddingDim": ssl_summary["embedding_dim"],
            "policyAccuracy": policy_summary["val_accuracy"],
            "policyMacroF1": policy_summary["val_macro_f1"],
            "explanationCoverage": explanation_summary["with_evidence"]
            / explanation_summary["explanations"],
        },
        "simulation": simulation,
        "baselines": [
            typed_row(row, {"policy"})
            for row in baselines
        ],
        "effects": effects,
        "profiles": profiles,
        "ablations": ablations,
        "explanations": explanations,
        "training": {"ssl": ssl_training, "policy": policy_training},
        "resources": build_resource_profile(baselines, simulation),
        "provenance": [
            "outputs/14_cybsage_cage4_policy/cybsage_cage4_step_trace.csv",
            "outputs/14_cybsage_cage4_policy/cybsage_cage4_episode_results.csv",
            "outputs/13_cage4_baseline_rollouts/cage4_baseline_summary.csv",
            "outputs/16_sci_readiness_pack/cage4_effect_sizes.csv",
            "outputs/16_sci_readiness_pack/cage4_profile_ablation_table.csv",
            "outputs/10_ablation_sota/ablation_summary.csv",
            "outputs/08_explanations/action_explanations.json",
            "outputs/06_ssl/ssl_training_log.csv",
            "outputs/07_posterior_policy/policy_training_log.csv",
        ],
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Wrote {DESTINATION} ({DESTINATION.stat().st_size:,} bytes)")
    print(f"Recorded simulation episodes: {len(simulation)}")
    print(f"Explanation records: {len(explanations)}")


if __name__ == "__main__":
    main()
