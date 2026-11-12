from cybsage_marl.data.synthetic_env import generate_trajectories


def test_generate_trajectories():
    rows = generate_trajectories(episodes=2, seed=7, max_steps=4)
    assert rows
    assert {"episode_id", "observation", "action", "reward", "regime"} <= set(rows[0])
