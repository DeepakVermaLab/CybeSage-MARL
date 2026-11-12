from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


ACTIONS = ["monitor", "analyze", "restore", "isolate", "deploy_decoy"]
REGIMES = ["recon", "lateral_movement", "privilege_escalation", "impact"]


@dataclass
class CyberState:
    step: int
    regime: str
    compromised: int
    service_down: int
    alerts: int
    decoys: int
    uncertainty: float

    def vector(self) -> list[float]:
        return [
            self.compromised / 10.0,
            self.service_down / 10.0,
            self.alerts / 10.0,
            self.decoys / 5.0,
            self.uncertainty,
            REGIMES.index(self.regime) / max(1, len(REGIMES) - 1),
        ]


class SimpleCyberDefenseEnv:
    """Small CPU simulator for pipeline smoke tests before CybORG rollouts."""

    def __init__(self, seed: int = 42, hosts: int = 10, max_steps: int = 32):
        self.rng = random.Random(seed)
        self.hosts = hosts
        self.max_steps = max_steps
        self.state = CyberState(0, "recon", 0, 0, 0, 0, 0.6)

    def reset(self, regime: str | None = None) -> CyberState:
        self.state = CyberState(
            step=0,
            regime=regime or self.rng.choice(REGIMES),
            compromised=self.rng.randint(0, 2),
            service_down=self.rng.randint(0, 1),
            alerts=self.rng.randint(1, 4),
            decoys=0,
            uncertainty=0.6,
        )
        return self.state

    def best_action(self, state: CyberState) -> str:
        if state.regime == "recon":
            return "monitor" if state.alerts < 4 else "analyze"
        if state.regime == "lateral_movement":
            return "isolate" if state.compromised > 2 else "analyze"
        if state.regime == "privilege_escalation":
            return "deploy_decoy" if state.decoys < 2 else "isolate"
        return "restore" if state.service_down > 0 else "isolate"

    def step(self, action: str) -> tuple[CyberState, float, bool, dict[str, Any]]:
        state = self.state
        optimal = self.best_action(state)
        reward = 1.0 if action == optimal else -0.2
        compromised = state.compromised
        service_down = state.service_down
        decoys = state.decoys

        attack_pressure = REGIMES.index(state.regime) + 1
        if action == "isolate":
            compromised = max(0, compromised - self.rng.randint(1, 2))
        elif action == "restore":
            service_down = max(0, service_down - 1)
        elif action == "deploy_decoy":
            decoys = min(5, decoys + 1)
        elif action == "monitor":
            reward += 0.1

        if action != optimal:
            compromised = min(self.hosts, compromised + self.rng.randint(0, attack_pressure))
        if state.regime == "impact":
            service_down = min(self.hosts, service_down + self.rng.randint(0, 2))

        next_regime = state.regime
        if self.rng.random() < 0.12:
            next_regime = self.rng.choice(REGIMES)

        next_state = CyberState(
            step=state.step + 1,
            regime=next_regime,
            compromised=compromised,
            service_down=service_down,
            alerts=max(0, int(np.clip(state.alerts + self.rng.randint(-1, 2), 0, 10))),
            decoys=decoys,
            uncertainty=max(0.05, state.uncertainty * (0.92 if action in {"analyze", "monitor"} else 0.98)),
        )
        done = next_state.step >= self.max_steps or next_state.compromised >= self.hosts
        reward -= 0.1 * next_state.compromised + 0.15 * next_state.service_down
        self.state = next_state
        return next_state, float(reward), done, {"optimal_action": optimal}


def generate_trajectories(episodes: int, seed: int = 42, max_steps: int = 32) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        env = SimpleCyberDefenseEnv(seed=seed + episode, max_steps=max_steps)
        state = env.reset(regime=rng.choice(REGIMES))
        done = False
        while not done:
            optimal = env.best_action(state)
            action = optimal if rng.random() < 0.78 else rng.choice(ACTIONS)
            next_state, reward, done, info = env.step(action)
            rows.append(
                {
                    "episode_id": episode,
                    "step": state.step,
                    "regime": state.regime,
                    "observation": state.vector(),
                    "action": action,
                    "optimal_action": info["optimal_action"],
                    "reward": reward,
                    "next_observation": next_state.vector(),
                    "done": done,
                }
            )
            state = next_state
    return rows
