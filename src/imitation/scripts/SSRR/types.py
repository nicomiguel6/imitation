from __future__ import annotations

import dataclasses
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from imitation.data import types


@dataclasses.dataclass(frozen=True)
class SigmoidParams:
    """Four-parameter sigmoid for SSRR Phase 2 (paper Eq. 4).

    sigma(eta) = c / (1 + exp(-k * (eta - x0))) + y0
    """

    x0: float
    y0: float
    c: float
    k: float

    def __call__(self, eta: np.ndarray | float) -> np.ndarray:
        eta_arr = np.asarray(eta, dtype=np.float64)
        return self.c / (1.0 + np.exp(-self.k * (eta_arr - self.x0))) + self.y0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.c, self.k)


@dataclasses.dataclass(frozen=True)
class NoiseBucket:
    """Rollouts grouped by injected noise level."""

    noise_level: float
    trajectories: Sequence[types.TrajectoryWithRew]


@dataclasses.dataclass(frozen=True)
class NoisePerformanceData:
    """Phase 2 raw data used for sigmoid fitting."""

    noise_levels: np.ndarray  # shape (K,)
    returns_mean: np.ndarray  # shape (K,)
    returns_std: np.ndarray  # shape (K,)
    returns_all: Optional[np.ndarray] = None  # shape (K, N_i) ragged padded not guaranteed


@dataclasses.dataclass(frozen=True)
class CurveFitDiagnostics:
    """Diagnostics for Phase 2 fitting."""

    r2: float
    residuals: np.ndarray  # shape (K,)
    y_true: np.ndarray  # shape (K,)
    y_pred: np.ndarray  # shape (K,)


@dataclasses.dataclass(frozen=True)
class SnippetExample:
    """A single snippet regression example."""

    obs: np.ndarray  # shape (L+1, obs_dim)
    acts: np.ndarray  # shape (L, act_dim)
    target_return: float
    noise_level: float


@dataclasses.dataclass(frozen=True)
class SSRRRegressionConfig:
    """Controls snippet sampling + target scaling for Phase 3."""

    min_steps: int = 50
    max_steps: int = 500
    target_scale: float = 10.0
    # If True, target is sigma(eta) scaled by (L / T). If False, target is sigma(eta).
    length_normalize: bool = True


@dataclasses.dataclass(frozen=True)
class SSRRRunArtifacts:
    """Minimal artifact bundle produced by SSRR (for runners/tests)."""

    buckets: Sequence[NoiseBucket]
    sigmoid_params: SigmoidParams
    curve_diag: CurveFitDiagnostics

