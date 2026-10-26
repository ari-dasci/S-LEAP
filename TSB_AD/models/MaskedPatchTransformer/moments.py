"""Local-moment evidence: per-channel level and dispersion shifts against the training segment."""

import numpy as np

from .scaling import unit


def local_moments(x, window):
    """Sliding mean and std over `window` samples (cumulative sums, edge padding) → (M, S), both [n, C]."""
    x = np.asarray(x, dtype=np.float64)
    n, c = x.shape
    w = int(max(2, min(window, n)))
    left = w // 2
    right = w - 1 - left
    xp = np.pad(x, ((left, right), (0, 0)), mode='edge')

    zero = np.zeros((1, c), dtype=np.float64)
    c1 = np.concatenate([zero, np.cumsum(xp, axis=0)], axis=0)
    c2 = np.concatenate([zero, np.cumsum(xp * xp, axis=0)], axis=0)

    m = (c1[w:] - c1[:-w]) / w
    m2 = (c2[w:] - c2[:-w]) / w
    s = np.sqrt(np.maximum(m2 - m * m, 0.0))
    return m[:n], s[:n]


class MomentCalibrator:
    """Per-channel baseline of the local moments over the training segment (two-sided scoring)."""

    def __init__(self, floor_frac=0.05, gate=1.0):
        self.floor_frac = float(floor_frac)
        self.gate = float(gate)
        self.mu_m = self.sd_m = None
        self.mu_s = self.sd_s = None

    @staticmethod
    def _floored(sd, frac):
        """Floor each sigma at a fraction of the median of the positive ones."""
        positive = sd[sd > 0]
        anchor = np.median(positive) if positive.size else 1.0
        if not np.isfinite(anchor) or anchor <= 0:
            anchor = 1.0
        return np.maximum(sd, frac * anchor)

    def fit(self, m_train, s_train):
        self.mu_m = m_train.mean(axis=0)
        self.mu_s = s_train.mean(axis=0)
        self.sd_m = self._floored(m_train.std(axis=0), self.floor_frac)
        self.sd_s = self._floored(s_train.std(axis=0), self.floor_frac)
        return self

    @property
    def ready(self):
        return self.mu_m is not None

    _unit = staticmethod(unit)

    def threshold(self, n_channels):
        """Gate band scaled by the extreme-value bound sqrt(2 ln k) over 2C statistics."""
        k = max(2 * int(n_channels), 2)
        return self.gate * np.sqrt(2.0 * np.log(k))

    def excess(self, m, s):
        """Level and dispersion z-scores (max over channels), each rescaled with `unit` and summed → [n]."""
        z_m = np.abs(m - self.mu_m) / self.sd_m
        z_s = np.abs(s - self.mu_s) / self.sd_s
        return self._unit(z_m.max(axis=1)) + self._unit(z_s.max(axis=1))
