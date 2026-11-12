from __future__ import annotations

import importlib.util
import sys
import types


def install_ray_rllib_stub() -> bool:
    """Provide the tiny RLlib surface CAGE-4 imports for non-RLlib CPU rollouts.

    The CAGE-4 package imports ``EnterpriseMAE`` from its wrappers ``__init__``.
    That class inherits from ``ray.rllib.env.multi_agent_env.MultiAgentEnv`` even
    when we only use ``BlueEnterpriseWrapper`` for heuristic CPU rollouts. This
    shim avoids forcing a heavy Ray/RLlib install for stages 13-15.
    """
    existing_multi_agent_env = sys.modules.get("ray.rllib.env.multi_agent_env")
    if existing_multi_agent_env is not None and hasattr(existing_multi_agent_env, "MultiAgentEnv"):
        return True

    try:
        ray_spec = importlib.util.find_spec("ray")
    except ValueError:
        ray_spec = None
    if ray_spec is not None:
        return False

    ray = sys.modules.setdefault("ray", types.ModuleType("ray"))
    rllib = sys.modules.setdefault("ray.rllib", types.ModuleType("ray.rllib"))
    env = sys.modules.setdefault("ray.rllib.env", types.ModuleType("ray.rllib.env"))
    multi_agent_env = sys.modules.setdefault(
        "ray.rllib.env.multi_agent_env",
        types.ModuleType("ray.rllib.env.multi_agent_env"),
    )

    class MultiAgentEnv:  # noqa: D401 - compatibility placeholder
        """Minimal placeholder base class for CAGE-4 wrapper imports."""

    multi_agent_env.MultiAgentEnv = MultiAgentEnv
    env.multi_agent_env = multi_agent_env
    env.MultiAgentEnv = MultiAgentEnv
    rllib.env = env
    ray.rllib = rllib
    return True
