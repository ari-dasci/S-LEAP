"""MaskedPatchMorphLOF — Stage 3: + local-density calibration of the latent novelty.

novelty = d_kNN / (alpha * local_density + (1 - alpha) * median(dloo)), where dloo are the
leave-one-out 1-NN distances inside the train bank and local_density is their mean over the
`morph_lof_k` neighbours of the window. The global median bounds the ratio where the bank is
very dense. alpha=0 gives MaskedPatchMorph.
"""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from TSB_AD.models.MaskedPatchTransformer import MaskedPatchAD
from TSB_AD.models.MaskedPatchTransformer.detector import _unit
from TSB_AD.models.MaskedPatchMorph.detector import MaskedPatchMorphAD

LOF_K = 4         # train neighbours averaged into the local density
LOF_ALPHA = 0.25  # weight of the local density against the global median


class MaskedPatchMorphLOFAD(MaskedPatchMorphAD):
    """MaskedPatchMorph with the k-NN distance calibrated by the local density of the train bank."""

    def __init__(self, *args, morph_lof_k=LOF_K, morph_lof_alpha=LOF_ALPHA, **kwargs):
        super().__init__(*args, **kwargs)
        self.morph_lof_k = int(morph_lof_k)
        self.morph_lof_alpha = float(morph_lof_alpha)
        self._morph_dloo = None    # leave-one-out 1-NN distance of each bank window
        self._morph_med = None     # median of those distances

    def fit(self, train):
        # Calls the Stage 1 fit directly, so the bank is built once with k+1 neighbours.
        MaskedPatchAD.fit(self, train)
        x = np.asarray(train, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        v = x.std(0)
        chan = np.argsort(v)[::-1][:min(self.morph_chan_cap, x.shape[1])].copy()
        self._morph_chan = torch.as_tensor(chan, dtype=torch.long, device=self.device)
        emb, _ = self._win_embeds(self._normalize(x))
        if len(emb) > self.morph_ntrain:
            idx = np.random.RandomState(0).choice(len(emb), self.morph_ntrain, replace=False)
            emb = emb[idx]

        k = min(self.morph_lof_k + 1, len(emb))
        self._morph_ref = NearestNeighbors(n_neighbors=k).fit(emb)
        if k >= 2:
            # Column 0 is the window itself; column 1 is its nearest other window.
            self._morph_dloo = self._morph_ref.kneighbors(emb)[0][:, 1]
            self._morph_med = float(np.median(self._morph_dloo))
        else:
            # Single-window bank: no density to estimate, so the raw distance is used.
            self._morph_dloo = None
            self._morph_med = None
        return self

    @torch.no_grad()
    def decision_function(self, data):
        base = np.asarray(MaskedPatchAD.decision_function(self, data),
                          dtype=np.float64).ravel()
        self.base_score_ = base
        n = len(base)
        x = np.asarray(data, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        emb, starts = self._win_embeds(self._normalize(x))
        dist, idx = self._morph_ref.kneighbors(emb)
        d = dist[:, 0]
        if self._morph_dloo is not None and self.morph_lof_alpha > 0:
            kk = min(self.morph_lof_k, idx.shape[1])
            dens = np.maximum(self._morph_dloo[idx[:, :kk]].mean(1), 1e-8)
            scale = (self.morph_lof_alpha * dens
                     + (1.0 - self.morph_lof_alpha) * self._morph_med)
            d = d / np.maximum(scale, 1e-8)
        morph = np.nan_to_num(self._to_time(starts, d, self.win_size, n))
        self.morph_component_ = morph
        return _unit(base) + self.morph_beta * _unit(morph)
