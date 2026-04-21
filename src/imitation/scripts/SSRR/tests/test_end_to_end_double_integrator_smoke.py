from pathlib import Path

from imitation.scripts.SSRR.runner_double_integrator import run_ssrr_double_integrator
from imitation.scripts.SSRR.types import SSRRRegressionConfig


def test_end_to_end_smoke(tmp_path: Path):
    out_dir = tmp_path / "ssrr"
    stats = run_ssrr_double_integrator(
        save_dir=out_dir,
        seed=0,
        phase1_algorithm="noisy_airl",
        noisy_airl_epsilon=0.2,
        demo_episodes=2,
        airl_gen_train_timesteps=64,
        airl_total_timesteps=256,
        noise_levels=(0.0, 0.5, 1.0),
        rollouts_per_noise=1,
        regression_num_samples=64,
        regression_steps=10,
        regression_batch_size=8,
        regression_cfg=SSRRRegressionConfig(min_steps=2, max_steps=5),
        rl_total_timesteps=128,
        device="cpu",
    )

    assert "sigmoid_r2" in stats
    assert (out_dir / "airl_reward_test.pt").exists()
    assert (out_dir / "sigmoid_params.npy").exists()
    assert (out_dir / "ssrr_reward.pt").exists()
    assert (out_dir / "final_policy.zip").exists()

