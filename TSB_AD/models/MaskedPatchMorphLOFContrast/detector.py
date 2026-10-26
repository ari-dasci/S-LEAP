"""MaskedPatchMorphLOFContrast — Stage 4: + local temporal contrast.

score = s0 + gamma * winsor_unit(s0 / moving_median(s0, W)), with s0 the Stage 3 score and
W = contrast_w_mult * win_size samples (or contrast_w if set). gamma=0 gives MaskedPatchMorphLOF.
"""
import numpy as np
import torch
from scipy.ndimage import median_filter

from TSB_AD.models.MaskedPatchTransformer.scaling import winsor_unit
from TSB_AD.models.MaskedPatchMorphLOF.detector import MaskedPatchMorphLOFAD

CONTRAST_W_MULT = 64    # W = CONTRAST_W_MULT * win_size
CONTRAST_W = None       # absolute W in samples; overrides the multiple when set
CONTRAST_GAMMA = 0.5    # weight of the contrast term
CONTRAST_Q = 0.99       # winsorizing quantile of the contrast ratio


class MaskedPatchMorphLOFContrastAD(MaskedPatchMorphLOFAD):
    """MaskedPatchMorphLOF plus a local-contrast term."""

    def __init__(self, *args, contrast_w=CONTRAST_W, contrast_w_mult=CONTRAST_W_MULT,
                 contrast_gamma=CONTRAST_GAMMA, contrast_q=CONTRAST_Q, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrast_w = None if contrast_w is None else int(contrast_w)
        self.contrast_w_mult = float(contrast_w_mult)
        self.contrast_gamma = float(contrast_gamma)
        self.contrast_q = float(contrast_q)
        self.s0_component_ = None          # last score before the contrast term
        self.contrast_component_ = None    # last contrast ratio, before winsorizing
        self.contrast_w_ = None            # effective W of the last call

    _winsor_unit = staticmethod(winsor_unit)

    def _contrast_window(self):
        """Effective W in samples: absolute if `contrast_w` is set, else relative to win_size."""
        if self.contrast_w is not None:
            return int(self.contrast_w)
        return int(max(3, self.contrast_w_mult * int(self.win_size)))

    def _contrast(self, s):
        """s / moving_median(s, W), with a median(s) * 1e-3 floor against near-zero context."""
        w = int(max(3, self._contrast_window()) | 1)
        self.contrast_w_ = w
        base = median_filter(s, size=w, mode="nearest")
        return s / (base + np.median(s) * 1e-3 + 1e-9)

    @torch.no_grad()
    def decision_function(self, data):
        s0 = np.asarray(super().decision_function(data), dtype=np.float64).ravel()
        self.s0_component_ = s0
        if self.contrast_gamma == 0.0:
            self.contrast_component_ = None
            return s0
        c = np.nan_to_num(self._contrast(s0))
        self.contrast_component_ = c
        return s0 + self.contrast_gamma * self._winsor_unit(c, self.contrast_q)
