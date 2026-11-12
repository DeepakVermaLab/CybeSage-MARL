#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cybsage_marl.config import load_config
from cybsage_marl.cage4.comparison import run_cage4_sota_comparison
from cybsage_marl.cage4.dependencies import check_cage4_dependencies
from cybsage_marl.cage4.rollouts import RolloutConfig, run_baseline_rollouts, run_cybsage_cage4_rollouts
from cybsage_marl.cage4.sci_readiness import run_sci_readiness_pack
from cybsage_marl.cti.knowledge_graph import build_compact_graph
from cybsage_marl.data.audit import audit_all
from cybsage_marl.data.cyborg_adapter import cyborg_import_status
from cybsage_marl.data.eda import run_eda as run_eda_stage
from cybsage_marl.data.preprocess import run_preprocessing
from cybsage_marl.data.synthetic_env import generate_trajectories
from cybsage_marl.evaluation.ablation import run_ablation_suite
from cybsage_marl.evaluation.metrics import evaluate_pipeline
from cybsage_marl.evaluation.statistics import run_statistical_analysis
from cybsage_marl.explain.generator import generate_explanations
from cybsage_marl.marl.train import train_policy
from cybsage_marl.ssl.train import train_ssl_encoder
from cybsage_marl.utils.device import resolve_device
from cybsage_marl.utils.io import set_seed, write_json, write_jsonl
from cybsage_marl.utils.tracking import write_stage_manifest


def stage_paths(cfg):
    return {
        "audit": cfg.output_dir("01_audit"),
        "eda": cfg.output_dir("02_eda"),
        "trajectories": cfg.output_dir("03_trajectories"),
        "preprocess": cfg.output_dir("04_preprocessing"),
        "cti": cfg.output_dir("05_cti_graph"),
        "ssl": cfg.output_dir("06_ssl"),
        "marl": cfg.output_dir("07_posterior_policy"),
        "explain": cfg.output_dir("08_explanations"),
        "eval": cfg.output_dir("09_evaluation"),
        "ablation": cfg.output_dir("10_ablation_sota"),
        "statistical": cfg.output_dir("11_statistical_analysis"),
        "cage4_check": cfg.output_dir("12_cage4_dependency_check"),
        "cage4_baselines": cfg.output_dir("13_cage4_baseline_rollouts"),
        "cage4_cybsage": cfg.output_dir("14_cybsage_cage4_policy"),
        "cage4_sota": cfg.output_dir("15_sota_reward_comparison"),
        "cage4_sci_readiness": cfg.output_dir("16_sci_readiness_pack"),
    }


def run_audit(cfg):
    out = stage_paths(cfg)["audit"]
    summary = audit_all(cfg, out)
    summary["cyborg_import"] = cyborg_import_status(cfg.cyborg_dir / "cage-challenge-4")
    audit_path = write_json(out / "dataset_audit.json", summary)
    write_stage_manifest(
        out,
        "audit",
        [
            (audit_path, True, "Dataset audit with CybORG, MITRE, NVD status"),
            (out / "dataset_audit.csv", True, "Tabular audit summary"),
        ],
        metrics={"datasets_checked": 3, "cyborg_import_ok": bool(summary["cyborg_import"].get("import_ok"))},
    )
    print(f"Audit saved to {out}")


def run_eda(cfg):
    paths = stage_paths(cfg)
    trajectory_path = paths["trajectories"] / "trajectories.jsonl"
    summary = run_eda_stage(cfg, paths["eda"], trajectory_path if trajectory_path.exists() else None)
    print(f"EDA summary: {summary}")


def run_trajectories(cfg, episodes: int, max_steps: int):
    out = stage_paths(cfg)["trajectories"]
    rows = generate_trajectories(episodes=episodes, seed=cfg.seed, max_steps=max_steps)
    path = write_jsonl(out / "trajectories.jsonl", rows)
    summary = {"episodes": episodes, "rows": len(rows), "max_steps": max_steps, "source": "cpu_synthetic_smoke"}
    summary_path = write_json(out / "trajectory_summary.json", summary)
    write_stage_manifest(
        out,
        "trajectories",
        [
            (path, True, "Trajectory rows in common CybSAGE schema"),
            (summary_path, True, "Trajectory generation summary"),
        ],
        metrics=summary,
    )
    print(f"Trajectories saved to {path}")


def run_preprocess(cfg):
    paths = stage_paths(cfg)
    summary = run_preprocessing(paths["trajectories"] / "trajectories.jsonl", paths["preprocess"])
    print(f"Preprocessing summary: {summary}")


def run_cti(cfg, nvd_limit: int):
    out = stage_paths(cfg)["cti"]
    paths = build_compact_graph(cfg.mitre_file, cfg.nvd_dir, out, nvd_limit=nvd_limit)
    manifest_path = write_json(out / "cti_graph_manifest.json", {k: str(v) for k, v in paths.items()})
    write_stage_manifest(
        out,
        "cti",
        [(manifest_path, True, "CTI graph manifest")] + [(p, True, f"CTI artifact: {name}") for name, p in paths.items()],
        metrics={"nvd_limit": nvd_limit, "artifacts": len(paths)},
    )
    print(f"CTI graph saved to {out}")


def run_ssl(cfg, epochs: int):
    paths = stage_paths(cfg)
    device = resolve_device(cfg.device)
    summary = train_ssl_encoder(paths["trajectories"] / "trajectories.jsonl", paths["ssl"], epochs=epochs, device=device)
    write_stage_manifest(
        paths["ssl"],
        "ssl",
        [
            (paths["ssl"] / "ssl_encoder.pt", True, "Self-supervised trajectory encoder checkpoint"),
            (paths["ssl"] / "trajectory_embeddings.npy", True, "Trajectory embeddings used by posterior policy"),
            (paths["ssl"] / "trajectory_embedding_index.csv", True, "Embedding row index and metadata"),
            (paths["ssl"] / "ssl_training_log.csv", True, "Epoch-wise SSL train/validation losses"),
            (paths["ssl"] / "ssl_summary.json", True, "SSL training summary"),
        ],
        metrics=summary,
    )
    print(f"SSL summary: {summary}")


def run_marl(cfg, epochs: int):
    paths = stage_paths(cfg)
    device = resolve_device(cfg.device)
    summary = train_policy(
        paths["trajectories"] / "trajectories.jsonl",
        paths["ssl"] / "trajectory_embeddings.npy",
        paths["marl"],
        epochs=epochs,
        device=device,
    )
    write_stage_manifest(
        paths["marl"],
        "marl",
        [
            (paths["marl"] / "posterior_policy.pt", True, "Posterior-conditioned policy checkpoint"),
            (paths["marl"] / "policy_training_log.csv", True, "Epoch-wise policy training/testing metrics"),
            (paths["marl"] / "policy_dataset_index.csv", True, "Policy dataset row index"),
            (paths["marl"] / "policy_summary.json", True, "Policy training summary"),
        ],
        metrics=summary,
    )
    print(f"Policy summary: {summary}")


def run_explain(cfg, limit: int):
    paths = stage_paths(cfg)
    summary = generate_explanations(paths["trajectories"] / "trajectories.jsonl", paths["cti"], paths["explain"], limit=limit)
    write_stage_manifest(
        paths["explain"],
        "explain",
        [
            (paths["explain"] / "action_explanations.csv", True, "Tabular policy/evidence-grounded explanations"),
            (paths["explain"] / "action_explanations.json", True, "Structured explanation payloads with evidence"),
            (paths["explain"] / "explanation_summary.json", True, "Explanation coverage summary"),
        ],
        metrics=summary,
    )
    print(f"Explanation summary: {summary}")


def run_evaluate(cfg):
    paths = stage_paths(cfg)
    summary = evaluate_pipeline(paths["trajectories"] / "trajectories.jsonl", paths["marl"], paths["explain"], paths["eval"])
    write_stage_manifest(
        paths["eval"],
        "evaluate",
        [
            (paths["eval"] / "evaluation_summary.csv", True, "Evaluation metrics table"),
            (paths["eval"] / "evaluation_summary.json", True, "Evaluation metrics JSON"),
        ],
        metrics=summary,
    )
    print(f"Evaluation summary: {summary}")


def run_ablation(cfg, seeds: list[int], epochs: int):
    paths = stage_paths(cfg)
    device = resolve_device(cfg.device)
    summary = run_ablation_suite(
        paths["trajectories"] / "trajectories.jsonl",
        paths["ssl"] / "trajectory_embeddings.npy",
        paths["ablation"],
        seeds=seeds,
        epochs=epochs,
        device=device,
    )
    write_stage_manifest(
        paths["ablation"],
        "ablation",
        [
            (paths["ablation"] / "ablation_seed_results.csv", True, "Seed-wise baseline and ablation metrics"),
            (paths["ablation"] / "ablation_summary.csv", True, "Aggregated ablation ranking"),
            (paths["ablation"] / "ablation_summary.json", True, "Ablation run summary"),
        ],
        metrics=summary,
    )
    print(f"Ablation summary: {summary}")


def run_statistical(cfg):
    paths = stage_paths(cfg)
    summary = run_statistical_analysis(paths["eval"], paths["ablation"], paths["statistical"])
    write_stage_manifest(
        paths["statistical"],
        "statistical",
        [
            (paths["statistical"] / "paired_statistical_tests.csv", True, "Paired seed-wise tests against CybSAGE full"),
            (paths["statistical"] / "paper_ablation_table.csv", True, "Paper-ready ablation table"),
            (paths["statistical"] / "sota_comparability_matrix.csv", True, "SOTA comparison readiness matrix"),
            (paths["statistical"] / "statistical_sota_report.md", True, "Markdown report with claim boundary"),
            (paths["statistical"] / "statistical_analysis_summary.json", True, "Statistical analysis summary"),
        ],
        metrics=summary,
    )
    print(f"Statistical analysis summary: {summary}")


def run_cage4_check(cfg):
    paths = stage_paths(cfg)
    summary = check_cage4_dependencies(cfg.cyborg_dir / "cage-challenge-4", paths["cage4_check"])
    write_stage_manifest(
        paths["cage4_check"],
        "cage4_check",
        [
            (paths["cage4_check"] / "dependency_status.csv", True, "CAGE-4 package import status"),
            (paths["cage4_check"] / "dependency_summary.json", True, "CAGE-4 rollout readiness summary"),
        ],
        metrics=summary,
        notes="Install missing packages before stages 13-15 if ready_for_cage4_rollouts is false.",
    )
    print(f"CAGE-4 dependency summary: {summary}")


def _rollout_config(cfg, out_dir, seeds: list[int], episodes: int, max_steps: int, red_agent: str) -> RolloutConfig:
    return RolloutConfig(
        cyborg_repo=cfg.cyborg_dir / "cage-challenge-4",
        out_dir=out_dir,
        seeds=seeds,
        episodes=episodes,
        max_steps=max_steps,
        red_agent=red_agent,
    )


def run_cage4_baselines(cfg, seeds: list[int], episodes: int, max_steps: int, red_agent: str):
    paths = stage_paths(cfg)
    summary = run_baseline_rollouts(_rollout_config(cfg, paths["cage4_baselines"], seeds, episodes, max_steps, red_agent))
    write_stage_manifest(
        paths["cage4_baselines"],
        "cage4_baselines",
        [
            (paths["cage4_baselines"] / "cage4_baseline_episode_results.csv", True, "Episode-level CAGE-4 reward results for baselines"),
            (paths["cage4_baselines"] / "cage4_baseline_summary.csv", False, "Aggregated baseline reward table"),
            (paths["cage4_baselines"] / "cage4_baseline_step_trace.csv", False, "Step-level baseline reward and action trace"),
            (paths["cage4_baselines"] / "cage4_baseline_summary.json", True, "Baseline rollout completion summary"),
        ],
        metrics=summary,
    )
    print(f"CAGE-4 baseline summary: {summary}")


def run_cage4_cybsage(cfg, seeds: list[int], episodes: int, max_steps: int, red_agent: str):
    paths = stage_paths(cfg)
    summary = run_cybsage_cage4_rollouts(_rollout_config(cfg, paths["cage4_cybsage"], seeds, episodes, max_steps, red_agent))
    write_stage_manifest(
        paths["cage4_cybsage"],
        "cage4_cybsage",
        [
            (paths["cage4_cybsage"] / "cybsage_profile_tuning.csv", False, "Training-seed heuristic profile tuning results"),
            (paths["cage4_cybsage"] / "cybsage_profile_summary.csv", False, "Profile ranking by training reward"),
            (paths["cage4_cybsage"] / "cybsage_cage4_episode_results.csv", True, "Episode-level CAGE-4 reward results for CybSAGE"),
            (paths["cage4_cybsage"] / "cybsage_cage4_step_trace.csv", False, "Step-level CybSAGE reward and action trace"),
            (paths["cage4_cybsage"] / "cybsage_cage4_summary.json", True, "CybSAGE CAGE-4 rollout summary"),
        ],
        metrics=summary,
    )
    print(f"CybSAGE CAGE-4 summary: {summary}")


def run_cage4_sota(cfg):
    paths = stage_paths(cfg)
    summary = run_cage4_sota_comparison(paths["cage4_baselines"], paths["cage4_cybsage"], paths["cage4_sota"])
    write_stage_manifest(
        paths["cage4_sota"],
        "cage4_sota",
        [
            (paths["cage4_sota"] / "cage4_sota_comparison.csv", True, "Paired reward comparison against CAGE-4 baselines"),
            (paths["cage4_sota"] / "cage4_sota_summary.json", True, "SOTA superiority gate summary"),
            (paths["cage4_sota"] / "cage4_sota_report.md", True, "Paper-facing reward comparison report"),
        ],
        metrics=summary,
    )
    print(f"CAGE-4 SOTA comparison summary: {summary}")


def run_cage4_sci_readiness(cfg, recommended_min_paired_runs: int):
    paths = stage_paths(cfg)
    summary = run_sci_readiness_pack(
        paths["cage4_baselines"],
        paths["cage4_cybsage"],
        paths["cage4_sota"],
        paths["cage4_sci_readiness"],
        recommended_min_paired_runs=recommended_min_paired_runs,
    )
    write_stage_manifest(
        paths["cage4_sci_readiness"],
        "cage4_sci_readiness",
        [
            (paths["cage4_sci_readiness"] / "cage4_effect_sizes.csv", True, "Paired CAGE-4 effect sizes and confidence intervals"),
            (paths["cage4_sci_readiness"] / "cage4_profile_ablation_table.csv", True, "CybSAGE CAGE-4 profile ablation/ranking table"),
            (paths["cage4_sci_readiness"] / "sci_readiness_summary.json", True, "SCI claim-readiness gate summary"),
            (paths["cage4_sci_readiness"] / "sci_readiness_report.md", True, "Paper-facing SCI readiness report"),
        ],
        metrics=summary,
    )
    print(f"CAGE-4 SCI readiness summary: {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "audit",
            "eda",
            "trajectories",
            "preprocess",
            "cti",
            "ssl",
            "marl",
            "explain",
            "evaluate",
            "ablation",
            "statistical",
            "cage4_check",
            "cage4_baselines",
            "cage4_cybsage",
            "cage4_sota",
            "cage4_sci_readiness",
            "all",
        ],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--nvd-limit", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ssl-epochs", type=int, default=5)
    parser.add_argument("--marl-epochs", type=int, default=5)
    parser.add_argument("--explain-limit", type=int, default=100)
    parser.add_argument("--ablation-epochs", type=int, default=5)
    parser.add_argument("--ablation-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--cage4-episodes", type=int, default=2)
    parser.add_argument("--cage4-max-steps", type=int, default=100)
    parser.add_argument("--cage4-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--cage4-red-agent", choices=["finite_state", "random_select"], default="finite_state")
    parser.add_argument("--sci-min-paired-runs", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    if args.stage in {"audit", "all"}:
        run_audit(cfg)
    if args.stage in {"eda"}:
        run_eda(cfg)
    if args.stage in {"trajectories", "all"}:
        run_trajectories(cfg, args.episodes, args.max_steps)
    if args.stage in {"preprocess", "all"}:
        run_preprocess(cfg)
    if args.stage in {"eda", "all"}:
        run_eda(cfg)
    if args.stage in {"cti", "all"}:
        run_cti(cfg, args.nvd_limit)
    if args.stage in {"ssl", "all"}:
        run_ssl(cfg, args.ssl_epochs if args.stage == "all" else args.epochs)
    if args.stage in {"marl", "all"}:
        run_marl(cfg, args.marl_epochs if args.stage == "all" else args.epochs)
    if args.stage in {"explain", "all"}:
        run_explain(cfg, args.explain_limit)
    if args.stage in {"evaluate", "all"}:
        run_evaluate(cfg)
    if args.stage in {"ablation", "all"}:
        run_ablation(cfg, args.ablation_seeds, args.ablation_epochs if args.stage == "all" else args.epochs)
    if args.stage in {"statistical", "all"}:
        run_statistical(cfg)
    if args.stage in {"cage4_check", "all"}:
        run_cage4_check(cfg)
    if args.stage in {"cage4_baselines", "all"}:
        run_cage4_baselines(cfg, args.cage4_seeds, args.cage4_episodes, args.cage4_max_steps, args.cage4_red_agent)
    if args.stage in {"cage4_cybsage", "all"}:
        run_cage4_cybsage(cfg, args.cage4_seeds, args.cage4_episodes, args.cage4_max_steps, args.cage4_red_agent)
    if args.stage in {"cage4_sota", "all"}:
        run_cage4_sota(cfg)
    if args.stage in {"cage4_sci_readiness", "all"}:
        run_cage4_sci_readiness(cfg, args.sci_min_paired_runs)


if __name__ == "__main__":
    main()
