from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cybsage_marl.cage4.compat import install_ray_rllib_stub
from cybsage_marl.data.cyborg_adapter import add_cyborg_to_path
from cybsage_marl.utils.io import write_json


BASELINE_POLICIES = [
    "sleep",
    "random",
    "monitor",
    "restore_priority",
    "remove_priority",
    "decoy_priority",
]

CYBSAGE_PROFILES = {
    "decoy_guarded": {
        "_strategy": "decoy_guarded",
        "_priority": 3.0,
    },
    "decoy_context_low": {
        "_strategy": "decoy_contextual",
        "_priority": 5.0,
        "AnalyseThreshold": 0.10,
    },
    "decoy_context_mid": {
        "_strategy": "decoy_contextual",
        "_priority": 5.5,
        "AnalyseThreshold": 0.20,
    },
    "decoy_context_high": {
        "_strategy": "decoy_contextual",
        "_priority": 5.2,
        "AnalyseThreshold": 0.30,
    },
    "decoy_remove_pressure": {
        "_strategy": "decoy_remove_pressure",
        "_priority": 4.0,
        "RemoveThreshold": 0.24,
    },
    "adaptive_decoy_monitor": {
        "_strategy": "adaptive_decoy_monitor",
        "_priority": 6.2,
        "WarmupSteps": 12,
        "PressureThreshold": 0.18,
    },
    "adaptive_decoy_repair": {
        "_strategy": "adaptive_decoy_repair",
        "_priority": 6.0,
        "WarmupSteps": 10,
        "PressureThreshold": 0.22,
    },
    "rotating_decoy": {
        "_strategy": "rotating_decoy",
        "_priority": 5.8,
    },
    "alert_response_decoy": {
        "_strategy": "alert_response",
        "_priority": 7.5,
        "WarmupSteps": 8,
        "RemoveAfterStep": 12,
        "MaxDecoySteps": 18,
    },
    "decoy_alert_remove": {
        "_strategy": "alert_response",
        "_priority": 7.3,
        "WarmupSteps": 14,
        "RemoveAfterStep": 16,
        "MaxDecoySteps": 24,
    },
    "monitor_alert_remove": {
        "_strategy": "alert_response",
        "_priority": 7.0,
        "WarmupSteps": 0,
        "RemoveAfterStep": 8,
        "MaxDecoySteps": 0,
    },
    "staged_decoy_sweep": {
        "_strategy": "staged_decoy_sweep",
        "_priority": 6.8,
        "DecoySteps": 16,
        "AnalyseSteps": 28,
    },
    "balanced": {
        "_priority": 1.0,
        "Monitor": 1.2,
        "Analyse": 1.8,
        "Remove": 2.2,
        "Restore": 2.0,
        "DeployDecoy": 1.7,
        "BlockTrafficZone": 0.7,
        "AllowTrafficZone": -0.3,
        "Sleep": -1.5,
    },
    "containment": {
        "_priority": 1.0,
        "Monitor": 0.8,
        "Analyse": 1.6,
        "Remove": 2.8,
        "Restore": 1.4,
        "DeployDecoy": 1.2,
        "BlockTrafficZone": 1.1,
        "AllowTrafficZone": -0.5,
        "Sleep": -1.7,
    },
    "resilience": {
        "_priority": 1.0,
        "Monitor": 0.9,
        "Analyse": 1.4,
        "Remove": 1.8,
        "Restore": 2.9,
        "DeployDecoy": 1.5,
        "BlockTrafficZone": 0.4,
        "AllowTrafficZone": -0.2,
        "Sleep": -1.5,
    },
    "deception": {
        "_priority": 1.0,
        "Monitor": 1.0,
        "Analyse": 1.5,
        "Remove": 1.7,
        "Restore": 1.8,
        "DeployDecoy": 3.0,
        "BlockTrafficZone": 0.6,
        "AllowTrafficZone": -0.3,
        "Sleep": -1.6,
    },
}


@dataclass
class RolloutConfig:
    cyborg_repo: Path
    out_dir: Path
    seeds: list[int]
    episodes: int = 2
    max_steps: int = 100
    red_agent: str = "finite_state"


def _import_cage4(cyborg_repo: Path):
    add_cyborg_to_path(cyborg_repo)
    install_ray_rllib_stub()
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, RandomSelectRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueEnterpriseWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    return {
        "CybORG": CybORG,
        "EnterpriseGreenAgent": EnterpriseGreenAgent,
        "FiniteStateRedAgent": FiniteStateRedAgent,
        "RandomSelectRedAgent": RandomSelectRedAgent,
        "SleepAgent": SleepAgent,
        "BlueEnterpriseWrapper": BlueEnterpriseWrapper,
        "EnterpriseScenarioGenerator": EnterpriseScenarioGenerator,
    }


def _make_env(imports: dict[str, Any], seed: int, max_steps: int, red_agent: str):
    red_cls = imports["FiniteStateRedAgent"] if red_agent == "finite_state" else imports["RandomSelectRedAgent"]
    sg = imports["EnterpriseScenarioGenerator"](
        blue_agent_class=imports["SleepAgent"],
        green_agent_class=imports["EnterpriseGreenAgent"],
        red_agent_class=red_cls,
        steps=max_steps,
    )
    cyborg = imports["CybORG"](scenario_generator=sg, seed=seed)
    return imports["BlueEnterpriseWrapper"](cyborg, seed=seed)


def _valid_indices(info: dict[str, Any], agent: str, action_space: Any) -> list[int]:
    mask = info.get(agent, {}).get("action_mask")
    if mask is None:
        return list(range(int(action_space.n)))
    return [idx for idx, valid in enumerate(mask) if bool(valid)]


def _find_label(labels: list[str], valid: list[int], contains: str, fallback: int) -> int:
    needle = contains.lower()
    for idx in valid:
        if needle in labels[idx].lower():
            return idx
    return fallback


def _matching_indices(labels: list[str], valid: list[int], contains: str) -> list[int]:
    needle = contains.lower()
    return [idx for idx in valid if needle in labels[idx].lower() and "[invalid]" not in labels[idx].lower()]


def _label_exists(labels: list[str], valid: list[int], contains: str) -> bool:
    needle = contains.lower()
    return any(needle in labels[idx].lower() and "[invalid]" not in labels[idx].lower() for idx in valid)


def _obs_pressure(observation: Any) -> float:
    arr = np.asarray(observation, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    # CAGE-4 blue vectors are mostly binary indicators after the mission phase.
    return float(np.clip(arr[1:].mean(), 0.0, 1.0))


def _event_host_tokens(observation: Any) -> tuple[list[str], list[str]]:
    arr = np.asarray(observation, dtype=float).ravel()
    if arr.size < 60:
        return [], []
    subnet_count = max(1, int((arr.size - 1 - 32) // 59))
    process_tokens: list[str] = []
    network_tokens: list[str] = []
    for subnet_idx in range(subnet_count):
        base = 1 + subnet_idx * 59
        process = arr[base + 27 : base + 43]
        network = arr[base + 43 : base + 59]
        for idx, value in enumerate(process):
            if value > 0.5:
                process_tokens.append(_host_token(idx))
        for idx, value in enumerate(network):
            if value > 0.5:
                network_tokens.append(_host_token(idx))
    return list(dict.fromkeys(process_tokens)), list(dict.fromkeys(network_tokens))


def _host_token(obs_host_index: int) -> str:
    # CAGE-4 orders six server hosts followed by ten user hosts in each subnet block.
    if obs_host_index < 6:
        return f"server_host_{obs_host_index}"
    return f"user_host_{obs_host_index - 6}"


def _find_targeted_action(
    labels: list[str],
    valid: list[int],
    action_name: str,
    host_tokens: list[str],
    fallback: int,
) -> int:
    action = action_name.lower()
    for token in host_tokens:
        token = token.lower()
        for idx in valid:
            label = labels[idx].lower()
            if "[invalid]" not in label and action in label and token in label:
                return idx
    return fallback


def _choose_action(
    env: Any,
    agent: str,
    observation: Any,
    info: dict[str, Any],
    policy: str,
    rng: random.Random,
    step: int = 0,
    profile: dict[str, float] | None = None,
) -> int:
    action_space = env.action_space(agent)
    labels = env.action_labels(agent)
    valid = _valid_indices(info, agent, action_space)
    if not valid:
        return 0
    sleep_idx = _find_label(labels, valid, "Sleep", valid[0])
    monitor_idx = _find_label(labels, valid, "Monitor", sleep_idx)

    if policy == "sleep":
        return sleep_idx
    if policy == "random":
        return rng.choice(valid)
    if policy == "monitor":
        return monitor_idx
    if policy == "restore_priority":
        return _find_label(labels, valid, "Restore", monitor_idx)
    if policy == "remove_priority":
        return _find_label(labels, valid, "Remove", monitor_idx)
    if policy == "decoy_priority":
        return _find_label(labels, valid, "DeployDecoy", monitor_idx)

    weights = profile or CYBSAGE_PROFILES["balanced"]
    strategy = str(weights.get("_strategy", "weighted"))
    pressure = _obs_pressure(observation)

    if strategy == "decoy_guarded":
        return _find_label(labels, valid, "DeployDecoy", monitor_idx)

    if strategy == "decoy_contextual":
        decoy_idx = _find_label(labels, valid, "DeployDecoy", -1)
        if decoy_idx >= 0:
            return decoy_idx
        analyse_threshold = float(weights.get("AnalyseThreshold", 0.20))
        if pressure >= analyse_threshold and _label_exists(labels, valid, "Analyse"):
            return _find_label(labels, valid, "Analyse", monitor_idx)
        return monitor_idx

    if strategy == "decoy_remove_pressure":
        remove_threshold = float(weights.get("RemoveThreshold", 0.24))
        if pressure >= remove_threshold:
            remove_idx = _find_label(labels, valid, "Remove", -1)
            if remove_idx >= 0:
                return remove_idx
            restore_idx = _find_label(labels, valid, "Restore", -1)
            if restore_idx >= 0:
                return restore_idx
        return _find_label(labels, valid, "DeployDecoy", monitor_idx)

    if strategy == "rotating_decoy":
        decoys = _matching_indices(labels, valid, "DeployDecoy")
        if decoys:
            return decoys[(step + agent_index(agent)) % len(decoys)]
        return monitor_idx

    if strategy == "adaptive_decoy_monitor":
        warmup_steps = int(weights.get("WarmupSteps", 12))
        pressure_threshold = float(weights.get("PressureThreshold", 0.18))
        decoy_idx = _find_label(labels, valid, "DeployDecoy", -1)
        if decoy_idx >= 0 and (step < warmup_steps or pressure < pressure_threshold):
            return decoy_idx
        analyse_idx = _find_label(labels, valid, "Analyse", -1)
        if analyse_idx >= 0 and pressure >= pressure_threshold:
            return analyse_idx
        return monitor_idx

    if strategy == "adaptive_decoy_repair":
        warmup_steps = int(weights.get("WarmupSteps", 10))
        pressure_threshold = float(weights.get("PressureThreshold", 0.22))
        if step < warmup_steps:
            return _find_label(labels, valid, "DeployDecoy", monitor_idx)
        if pressure >= pressure_threshold:
            remove_idx = _find_label(labels, valid, "Remove", -1)
            if remove_idx >= 0:
                return remove_idx
            restore_idx = _find_label(labels, valid, "Restore", -1)
            if restore_idx >= 0:
                return restore_idx
        return monitor_idx

    if strategy == "alert_response":
        process_tokens, network_tokens = _event_host_tokens(observation)
        warmup_steps = int(weights.get("WarmupSteps", 8))
        remove_after_step = int(weights.get("RemoveAfterStep", 12))
        max_decoy_steps = int(weights.get("MaxDecoySteps", 18))
        if process_tokens and step >= remove_after_step:
            return _find_targeted_action(labels, valid, "Remove", process_tokens, monitor_idx)
        if process_tokens:
            return _find_targeted_action(labels, valid, "Analyse", process_tokens, monitor_idx)
        if network_tokens and step >= remove_after_step:
            return _find_targeted_action(labels, valid, "Remove", network_tokens, monitor_idx)
        if network_tokens:
            return _find_targeted_action(labels, valid, "Analyse", network_tokens, monitor_idx)
        if step < warmup_steps or (max_decoy_steps > 0 and step < max_decoy_steps and pressure < 0.08):
            return _find_label(labels, valid, "DeployDecoy", monitor_idx)
        return monitor_idx

    if strategy == "staged_decoy_sweep":
        process_tokens, network_tokens = _event_host_tokens(observation)
        decoy_steps = int(weights.get("DecoySteps", 16))
        analyse_steps = int(weights.get("AnalyseSteps", 28))
        if process_tokens and step >= analyse_steps:
            return _find_targeted_action(labels, valid, "Remove", process_tokens, monitor_idx)
        if process_tokens or network_tokens:
            return _find_targeted_action(labels, valid, "Analyse", process_tokens + network_tokens, monitor_idx)
        if step < decoy_steps:
            return _find_label(labels, valid, "DeployDecoy", monitor_idx)
        if step < analyse_steps:
            analyse_idx = _find_label(labels, valid, "Analyse", -1)
            if analyse_idx >= 0:
                return analyse_idx
        return monitor_idx

    best_idx = monitor_idx
    best_score = -math.inf
    for idx in valid:
        label = labels[idx]
        score = -0.02 * idx
        for token, weight in weights.items():
            if token.startswith("_"):
                continue
            if token.lower() in label.lower():
                score += float(weight)
        if "Analyse" in label or "Remove" in label:
            score += pressure
        if "Restore" in label:
            score += 0.7 * pressure
        if "DeployDecoy" in label:
            score += 0.4 * (1.0 - pressure)
        if score > best_score:
            best_score = score
            best_idx = idx
    return int(best_idx)


def agent_index(agent: str) -> int:
    try:
        return int(str(agent).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 0


def _run_episode(
    env: Any,
    seed: int,
    episode: int,
    policy: str,
    max_steps: int,
    profile: dict[str, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed * 1009 + episode)
    observations, info = env.reset(seed=seed + episode)
    total_reward = 0.0
    step_rows: list[dict[str, Any]] = []
    completed_steps = 0
    start = time.time()
    for step in range(max_steps):
        actions = {}
        labels = {}
        for agent in sorted(env.agents):
            action_idx = _choose_action(env, agent, observations[agent], info, policy, rng, step, profile)
            actions[agent] = action_idx
            labels[agent] = env.action_labels(agent)[action_idx]
        observations, rewards, terminated, truncated, info = env.step(actions)
        mean_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
        total_reward += mean_reward
        completed_steps = step + 1
        step_rows.append(
            {
                "seed": seed,
                "episode": episode,
                "step": step,
                "policy": policy,
                "mean_reward": mean_reward,
                "sum_reward": float(sum(rewards.values())) if rewards else 0.0,
                "actions": labels,
            }
        )
        done = {
            agent: bool(terminated.get(agent, False) or truncated.get(agent, False))
            for agent in getattr(env, "possible_agents", env.agents)
        }
        if done and all(done.values()):
            break
    return (
        {
            "seed": seed,
            "episode": episode,
            "policy": policy,
            "total_reward": total_reward,
            "mean_step_reward": total_reward / max(1, completed_steps),
            "steps": completed_steps,
            "wall_time_sec": time.time() - start,
        },
        step_rows,
    )


def _summarize_episode_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = (
        df.groupby("policy", as_index=False)
        .agg(
            episodes=("episode", "count"),
            reward_mean=("total_reward", "mean"),
            reward_std=("total_reward", "std"),
            reward_min=("total_reward", "min"),
            reward_max=("total_reward", "max"),
            mean_step_reward=("mean_step_reward", "mean"),
            steps_mean=("steps", "mean"),
            wall_time_sec_mean=("wall_time_sec", "mean"),
        )
        .fillna(0.0)
        .sort_values("reward_mean", ascending=False)
    )
    return summary


def run_baseline_rollouts(config: RolloutConfig) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        imports = _import_cage4(config.cyborg_repo)
    except Exception as exc:
        summary = {"status": "blocked", "error": str(exc), "policies": BASELINE_POLICIES}
        write_json(config.out_dir / "cage4_baseline_summary.json", summary)
        pd.DataFrame().to_csv(config.out_dir / "cage4_baseline_episode_results.csv", index=False)
        return summary

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for policy in BASELINE_POLICIES:
        for seed in config.seeds:
            env = _make_env(imports, seed, config.max_steps, config.red_agent)
            for episode in range(config.episodes):
                row, steps = _run_episode(env, seed, episode, policy, config.max_steps)
                episode_rows.append(row)
                step_rows.extend(steps)

    episode_df = pd.DataFrame(episode_rows)
    summary_df = _summarize_episode_rows(episode_rows)
    episode_df.to_csv(config.out_dir / "cage4_baseline_episode_results.csv", index=False)
    pd.DataFrame(step_rows).to_csv(config.out_dir / "cage4_baseline_step_trace.csv", index=False)
    summary_df.to_csv(config.out_dir / "cage4_baseline_summary.csv", index=False)
    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    summary = {
        "status": "completed",
        "episodes_per_policy": int(config.episodes * len(config.seeds)),
        "seeds": config.seeds,
        "max_steps": config.max_steps,
        "policies": BASELINE_POLICIES,
        "best_baseline": best.get("policy"),
        "best_baseline_reward_mean": float(best.get("reward_mean", 0.0)) if best else None,
    }
    write_json(config.out_dir / "cage4_baseline_summary.json", summary)
    return summary


def run_cybsage_cage4_rollouts(config: RolloutConfig) -> dict[str, Any]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        imports = _import_cage4(config.cyborg_repo)
    except Exception as exc:
        summary = {"status": "blocked", "error": str(exc), "profiles": list(CYBSAGE_PROFILES)}
        write_json(config.out_dir / "cybsage_cage4_summary.json", summary)
        pd.DataFrame().to_csv(config.out_dir / "cybsage_cage4_episode_results.csv", index=False)
        return summary

    train_seeds = config.seeds[: max(1, len(config.seeds) // 2)]
    eval_seeds = config.seeds[max(1, len(config.seeds) // 2) :] or train_seeds
    tuning_rows: list[dict[str, Any]] = []
    for profile_name, profile in CYBSAGE_PROFILES.items():
        for seed in train_seeds:
            env = _make_env(imports, seed, config.max_steps, config.red_agent)
            for episode in range(config.episodes):
                row, _ = _run_episode(env, seed, episode, "cybsage_" + profile_name, config.max_steps, profile)
                row["profile"] = profile_name
                row["split"] = "train"
                tuning_rows.append(row)
    tuning_df = pd.DataFrame(tuning_rows)
    profile_summary = (
        tuning_df.groupby("profile", as_index=False)["total_reward"].mean().sort_values("total_reward", ascending=False)
        if not tuning_df.empty
        else pd.DataFrame()
    )
    if not profile_summary.empty:
        profile_summary["selection_priority"] = profile_summary["profile"].map(
            lambda name: float(CYBSAGE_PROFILES[str(name)].get("_priority", 1.0))
        )
        best_reward = float(profile_summary["total_reward"].max())
        candidate_summary = profile_summary[profile_summary["total_reward"] >= best_reward - 5.0]
        selected_profile_name = str(
            candidate_summary.sort_values(
                ["selection_priority", "total_reward", "profile"],
                ascending=[False, False, True],
            ).iloc[0]["profile"]
        )
        profile_summary = profile_summary.sort_values(
            ["total_reward", "selection_priority"],
            ascending=[False, False],
        )
    else:
        selected_profile_name = "balanced"
    selected_profile = CYBSAGE_PROFILES[selected_profile_name]

    eval_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for seed in eval_seeds:
        env = _make_env(imports, seed, config.max_steps, config.red_agent)
        for episode in range(config.episodes):
            row, steps = _run_episode(env, seed, episode, "cybsage_full", config.max_steps, selected_profile)
            row["profile"] = selected_profile_name
            row["split"] = "eval"
            eval_rows.append(row)
            step_rows.extend(steps)

    pd.DataFrame(tuning_rows).to_csv(config.out_dir / "cybsage_profile_tuning.csv", index=False)
    profile_summary.to_csv(config.out_dir / "cybsage_profile_summary.csv", index=False)
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(config.out_dir / "cybsage_cage4_episode_results.csv", index=False)
    pd.DataFrame(step_rows).to_csv(config.out_dir / "cybsage_cage4_step_trace.csv", index=False)
    eval_summary = _summarize_episode_rows(eval_rows)
    eval_summary.to_csv(config.out_dir / "cybsage_cage4_summary.csv", index=False)
    best = eval_summary.iloc[0].to_dict() if not eval_summary.empty else {}
    summary = {
        "status": "completed",
        "selected_profile": selected_profile_name,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "episodes": int(len(eval_rows)),
        "max_steps": config.max_steps,
        "reward_mean": float(best.get("reward_mean", 0.0)) if best else None,
        "reward_std": float(best.get("reward_std", 0.0)) if best else None,
    }
    write_json(config.out_dir / "cybsage_cage4_summary.json", summary)
    return summary
