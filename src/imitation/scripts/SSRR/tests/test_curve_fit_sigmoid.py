import numpy as np
import matplotlib.pyplot as plt
from imitation.scripts.SSRR.curve_fit import fit_sigmoid_noise_performance
from imitation.scripts.SSRR.types import NoisePerformanceData


def test_sigmoid_fit_high_r2_on_synthetic():
    # Synthetic decreasing sigmoid-like curve with noise.
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 21)
    # True params: x0=0.5, y0=0.05, c=0.95, k=-8 (decreasing with noise)
    y_true = 0.95 / (1.0 + np.exp(-(-8.0) * (x - 0.5))) + 0.05
    y_noisy = y_true + rng.normal(scale=0.01, size=y_true.shape)

    data = NoisePerformanceData(
        noise_levels=x,
        returns_mean=y_noisy,
        returns_std=np.zeros_like(y_noisy),
        returns_all=None,
    )
    params, diag = fit_sigmoid_noise_performance(data, normalize_y=True, prefer_scipy=True)

    data = [x, y_noisy, y_true]

    assert np.isfinite(diag.r2)
    assert diag.r2 > 0.90
    # Should be monotone decreasing over [0,1] with negative k (common case).
    assert params.k < 0.0

    return params, diag, data

if __name__ == "__main__":
    params, diag, data = test_sigmoid_fit_high_r2_on_synthetic()

    # Plot
    plt.figure()
    plt.plot(data[0], data[1], label="Noisy")
    plt.plot(data[0], data[2], label="True")
    plt.plot(data[0], params(data[0]), label="Fitted")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("sigmoid_fit.png")
    plt.close()

