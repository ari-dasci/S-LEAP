"""Quantile rescaling unit(v) = v / q_0.99(v | v > 0), shared by every evidence term."""

import numpy as np

__all__ = ['quantile_scale', 'unit', 'winsor_unit']


def quantile_scale(v, q=0.99):
    """Divisor used by `unit`, or None when `v` has no finite positive sample."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return None
    active = v[np.isfinite(v) & (v > 0.0)]
    if active.size == 0:
        return None
    scale = float(np.quantile(active, q))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def unit(v, q=0.99):
    """Rescale a non-negative profile by the q-quantile of its active part (unchanged if all zero)."""
    v = np.asarray(v, dtype=np.float64)
    scale = quantile_scale(v, q)
    return v.copy() if scale is None else v / scale


def winsor_unit(v, q=0.99):
    """`unit` with the tail clipped at the same quantile."""
    v = np.asarray(v, dtype=np.float64)
    scale = quantile_scale(v, q)
    if scale is None:
        return v.copy()
    return np.clip(v, None, scale) / scale
