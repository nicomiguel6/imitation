import numpy as np
import torch as th

from imitation.rewards.reward_nets import BasicRewardNet
from imitation.scripts.SSRR.reward_regression import SSRRRegressor


def test_reward_regression_loss_decreases_simple():
    # Create a tiny reward net and a synthetic batch of snippets where the
    # correct return is 0.0 (all-zero observations/actions).
    obs_dim = 4
    act_dim = 1

    obs_space_low = -np.ones(obs_dim, dtype=np.float32)
    obs_space_high = np.ones(obs_dim, dtype=np.float32)
    act_space_low = -np.ones(act_dim, dtype=np.float32)
    act_space_high = np.ones(act_dim, dtype=np.float32)

    import gymnasium as gym

    obs_space = gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=np.float32)
    act_space = gym.spaces.Box(low=act_space_low, high=act_space_high, dtype=np.float32)

    net = BasicRewardNet(obs_space, act_space, hid_sizes=(16, 16))
    reg = SSRRRegressor(net, lr=1e-2, weight_decay=0.0, device="cpu")

    # Build a "dataloader-like" iterable that yields fixed batches.
    obs = th.zeros((6, obs_dim), dtype=th.float32)      # L+1=6
    acts = th.zeros((5, act_dim), dtype=th.float32)     # L=5
    targets = th.zeros((4,), dtype=th.float32)

    batch = ([obs] * 4, [acts] * 4, targets)
    dl = [batch] * 50

    losses = reg.train(dl, n_steps=50)["step_losses"]
    assert losses[-1] < losses[0]

