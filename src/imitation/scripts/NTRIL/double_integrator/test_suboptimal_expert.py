"""Compare the suboptimal PID expert against MPC on the double integrator."""

from pathlib import Path

import numpy as np
import torch as th
import gymnasium as gym
import matplotlib.pyplot as plt

from imitation.scripts.NTRIL.robust_tube_mpc import RobustTubeMPC
from stable_baselines3 import PPO
from imitation.scripts.NTRIL.double_integrator.double_integrator import DoubleIntegratorSuboptimalPolicy


SCRIPT_DIR = Path(__file__).parent.resolve()


def main():
    env_policy = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0")
    env_mpc = gym.make("imitation.scripts.NTRIL.double_integrator:DoubleIntegrator-v0")

    # load final policy
    final_policy_path = SCRIPT_DIR / "ntril_outputs" / "final_policy" / "final_policy.zip"
    final_policy = PPO.load(final_policy_path, device="cuda")

    device = "cuda"
    if device == "mps":
        th.set_default_dtype(th.float32)
    else:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    mpc_policy = RobustTubeMPC(
        horizon=20,
        time_step=1.0,
        A=np.array([[0.0, 1.0], [0.0, 0.0]]),
        B=np.array([[0.0], [1.0]]),
        Q=np.diag([1.0, 1.0]),
        R=0.1 * np.eye(1),
        state_bounds=(np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
        control_bounds=(np.array([-2.0]), np.array([2.0])),
    )
    mpc_policy.setup()

    suboptimal_policy = DoubleIntegratorSuboptimalPolicy(
        observation_space=env_policy.observation_space,
        action_space=env_policy.action_space,
    )
    suboptimal_policy.set_K_values(mpc_policy.K[0,0], mpc_policy.K[0,1])

    for j in range(1):
        obs, info = env_policy.reset()
        obs_mpc, info = env_mpc.reset(state=obs)
        states_policy = [obs.copy()]
        states_mpc = [obs.copy()]
        actions_policy = []
        actions_mpc = []
        rewards_policy = []
        rewards_mpc = []

        for i in range(env_policy.unwrapped.max_episode_steps):
            action_policy = suboptimal_policy._choose_action(obs)
            # action_policy, _ = final_policy.predict(obs)
            _, action_mpc = mpc_policy.solve_mpc(obs_mpc)
            obs, reward, terminated, truncated, info = env_policy.step(action_policy)
            obs_mpc, reward_mpc, terminated_mpc, truncated_mpc, info_mpc = env_mpc.step(action_mpc)
            states_policy.append(obs)
            states_mpc.append(obs_mpc)
            actions_policy.append(action_policy)
            actions_mpc.append(action_mpc)
            rewards_policy.append(reward)
            rewards_mpc.append(reward_mpc)
            if terminated or truncated:
                break

        fig, ax = plt.subplots()
        ax.plot(np.array(states_policy)[:, 0], "k-", label="Suboptimal Expert Trajectory")
        ax.plot(np.array(states_mpc)[:, 0], "r-", label="MPC Trajectory")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.set_title("Trajectory Comparison of Suboptimal Expert and MPC")
        ax.legend()
        fig.savefig(SCRIPT_DIR / f"trajectory_comparison_{j}.png")
        plt.close()

        print("Total suboptimal expert cost: ", sum(rewards_policy))
        print("Total MPC cost: ", sum(rewards_mpc))


if __name__ == "__main__":
    main()
