"""Window sizing from the dominant period of the training segment."""

import numpy as np


def autocorrelation(x, max_lag):
    """Autocorrelation of a 1-D signal via FFT, normalized so ACF(0) = 1."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 4 or not np.isfinite(x).all():
        return None
    var = np.dot(x, x)
    if var <= 0:
        return None
    # Zero-pad to a power of two ≥ 2n so the FFT correlation is linear.
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:max_lag + 1]
    return acf / var


def channel_period(x, base_lag=3, max_lag=400):
    """Highest ACF local maximum past `base_lag`, if it clears sqrt(2 ln m)/sqrt(n); else None."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    lag_cap = int(min(max_lag, n // 2 - 1))
    if lag_cap <= base_lag + 1:
        return None

    acf = autocorrelation(x, lag_cap)
    if acf is None:
        return None
    acf = acf[base_lag:]
    if acf.size < 3:
        return None

    is_peak = (acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:])
    peaks = np.nonzero(is_peak)[0] + 1
    if peaks.size == 0:
        return None

    best = int(peaks[np.argmax(acf[peaks])])
    threshold = np.sqrt(2.0 * np.log(max(acf.size, 2))) / np.sqrt(n)
    if acf[best] < threshold:
        return None
    return int(best + base_lag)


def estimate_period(data, base_lag=3, max_lag=400, max_points=20_000,
                    max_channels=64):
    """Median period over the (highest-variance) channels with a significant ACF peak, or None."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = arr[:max_points]

    n_channels = arr.shape[1]
    if n_channels > max_channels:
        cols = np.argsort(np.nanvar(arr, axis=0))[::-1][:max_channels]
    else:
        cols = range(n_channels)

    periods = [p for p in (channel_period(arr[:, c], base_lag, max_lag)
                           for c in cols) if p is not None]
    if not periods:
        return None
    return int(np.median(periods))


def choose_window(data, patch_size, periods=2, cap=384, fallback_patches=16):
    """win_size ≈ periods · dominant period, a multiple of patch_size between 4 patches and `cap`."""
    p = int(patch_size)
    floor = p * 4                      # ≥4 patches, else masking leaves no context
    cap = max(floor, (int(cap) // p) * p)

    period = estimate_period(data)
    if period is None:
        win = min(cap, max(floor, p * int(fallback_patches)))
        return int(win), None

    raw = int(periods) * int(period)
    win = int(round(raw / p)) * p
    win = max(floor, min(win, cap))
    return int(win), int(period)
