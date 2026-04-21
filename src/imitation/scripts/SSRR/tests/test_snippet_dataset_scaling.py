import numpy as np

from imitation.data import types
from imitation.scripts.SSRR.reward_regression import SnippetDataset
from imitation.scripts.SSRR.types import NoiseBucket, SigmoidParams, SSRRRegressionConfig


def _make_dummy_traj(T: int, obs_dim: int = 4, act_dim: int = 1) -> types.TrajectoryWithRew:
    obs = np.zeros((T + 1, obs_dim), dtype=np.float32)
    acts = np.zeros((T, act_dim), dtype=np.float32)
    rews = np.zeros((T,), dtype=np.float32)
    infos = np.array([{} for _ in range(T)], dtype=object)
    return types.TrajectoryWithRew(obs=obs, acts=acts, rews=rews, infos=infos, terminal=True)


def test_snippet_target_length_scaling_matches_paper():
    # sigma(eta)=2.0 constant for easy checking
    sig = SigmoidParams(x0=0.0, y0=2.0, c=0.0, k=1.0)
    T = 100
    traj = _make_dummy_traj(T=T)
    buckets = [NoiseBucket(noise_level=0.3, trajectories=[traj])]
    cfg = SSRRRegressionConfig(min_steps=10, max_steps=10, target_scale=10.0, length_normalize=True)
    ds = SnippetDataset(buckets, sig, num_samples=1, cfg=cfg, rng=np.random.default_rng(0))

    obs_snip, act_snip, target = ds[0]
    L = len(act_snip)
    assert L == 10
    # Target should be sigma * (L/T) * scale
    assert np.isclose(target, 2.0 * (L / T) * 10.0)

