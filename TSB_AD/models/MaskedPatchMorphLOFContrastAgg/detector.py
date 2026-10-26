"""MaskedPatchMorphLOFContrastAgg — Stage 5 (LEAP): + per-channel novelty over a channel pool.

The pool is the union of the top `agg_chan_cap` channels under six label-free criteria. Each
pooled channel gets its own density-calibrated novelty; their mean is added to the Stage 3
score with weight delta, before the local contrast. delta=0 gives MaskedPatchMorphLOFContrast.
"""
import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from TSB_AD.models.MaskedPatchTransformer import WindowDataset
from TSB_AD.models.MaskedPatchTransformer.detector import _unit
from TSB_AD.models.MaskedPatchMorphLOF.detector import MaskedPatchMorphLOFAD
from TSB_AD.models.MaskedPatchMorphLOFContrast.detector import MaskedPatchMorphLOFContrastAD

AGG_DELTA = 0.5         # weight of the per-channel novelty term
AGG_CHAN_CAP = 4        # channels nominated by each criterion (the pool is their union)
AGG_NTRAIN = 1200       # cap on train windows kept as the k-NN reference, per channel
AGG_KS_SAMPLE = 20000   # subsample size for the KS statistic (fixed seed)
AGG_CHUNK = 4           # channels per encoder pass (throughput only)


@torch.no_grad()
def _encode(model, x):
    h = model.embed(model.patchify(x))
    h = h + model.pos_embed.view(1, 1, model.n_patches, -1)
    h = h + model.channel_embed.view(1, model.n_channels, 1, -1)
    return model.encoder(h)


class MaskedPatchMorphLOFContrastAggAD(MaskedPatchMorphLOFContrastAD):
    """MaskedPatchMorphLOFContrast plus the per-channel novelty of a criterion-based pool."""

    _applies_end_dilation = True   # `_fuse_contrast` ends with `_dilate_end`

    def __init__(self, *args, agg_delta=AGG_DELTA, agg_chan_cap=AGG_CHAN_CAP,
                 agg_ntrain=AGG_NTRAIN, agg_ks_sample=AGG_KS_SAMPLE, agg_chunk=AGG_CHUNK,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.agg_delta = float(agg_delta)
        self.agg_chan_cap = int(agg_chan_cap)
        self.agg_ntrain = int(agg_ntrain)
        self.agg_ks_sample = int(agg_ks_sample)
        self.agg_chunk = int(agg_chunk)
        self._agg_ntr = None           # train length, stored by fit()
        self.pool_ = None              # pooled channels of the last call
        self.agg_component_ = None     # last pooled novelty, before fusion

    def fit(self, train):
        super().fit(train)
        self._agg_ntr = len(train)
        return self

    @staticmethod
    def _criterios(x, tr, ks_sample):
        """Six label-free per-channel criteria used to build the pool."""
        a, b = x[:tr], x[tr:]
        sd = np.maximum(a.std(0), 1e-8)
        z = np.abs((x - a.mean(0)) / sd)
        out = {"var_train": a.std(0),
               "nov_p99": (x > np.quantile(a, 0.99, axis=0)).mean(0),
               "rango": (np.ptp(b, axis=0) + 1e-12) / (np.ptp(a, axis=0) + 1e-12),
               "nov_mean": z.mean(0),
               "shift_std": b.std(0) / sd}
        ks = np.empty(x.shape[1])
        rs = np.random.RandomState(0)
        ia = a if len(a) <= ks_sample else a[rs.choice(len(a), ks_sample, replace=False)]
        ib_ = b if len(b) <= ks_sample else b[rs.choice(len(b), ks_sample, replace=False)]
        for c in range(x.shape[1]):
            ks[c] = ks_2samp(ia[:, c], ib_[:, c]).statistic if len(ib_) else 0.0
        out["ks"] = ks
        return out

    @torch.no_grad()
    def _embed_pool(self, normalized, pool):
        """Per-channel window embeddings (not concatenated, not normalised) for `pool`."""
        out = {c: [] for c in pool}
        starts = []
        for i in range(0, len(pool), self.agg_chunk):
            chunk = pool[i:i + self.agg_chunk]
            idx = torch.as_tensor(np.asarray(chunk), device=self.device)
            loader = DataLoader(WindowDataset(normalized, self.win_size, self.step),
                                batch_size=self.batch_size, shuffle=False)
            st = []
            for b, s in loader:
                h = _encode(self.model, b.to(self.device)).float()[:, idx]
                h = h.reshape(h.shape[0], len(chunk), -1).cpu().numpy().astype(np.float32)
                for j, c in enumerate(chunk):
                    out[c].append(h[:, j])
                st.append(s.numpy())
            starts = np.concatenate(st)
        return {c: np.concatenate(v) for c, v in out.items()}, starts

    @staticmethod
    def _l2(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

    def _dist_lof_pool(self, E, ref_idx):
        """Density-calibrated k-NN novelty of MaskedPatchMorphLOF, for one channel."""
        k, alpha = self.morph_lof_k, self.morph_lof_alpha
        R = E[ref_idx]
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(R))).fit(R)
        dist, idx = nn.kneighbors(E)
        dloo = nn.kneighbors(R)[0][:, 1] if len(R) > 1 else np.array([1.0])
        med = float(np.median(dloo))
        dens = np.maximum(dloo[idx[:, :min(k, idx.shape[1])]].mean(1), 1e-8)
        return dist[:, 0] / np.maximum(alpha * dens + (1 - alpha) * med, 1e-8)

    @torch.no_grad()
    def decision_function(self, data):
        # Stage 3 score; the contrast is applied after the pooled term is added.
        s0 = np.asarray(MaskedPatchMorphLOFAD.decision_function(self, data),
                        dtype=np.float64).ravel()
        n = len(s0)
        x = np.asarray(data, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]

        if x.shape[1] == 1:
            # With one channel the pooled term duplicates morph_lof; skipped when delta=0.
            if self.agg_delta == 0.0:
                self.pool_ = [0]
                self.agg_component_ = None
                return self._fuse_contrast(s0)
            self._log(f'[WARN] C=1: the pooled term duplicates morph_lof; agg_delta='
                      f'{self.agg_delta} is equivalent to morph_beta={self.morph_beta + self.agg_delta}')

        tr = self._agg_ntr

        cr = self._criterios(x, tr, self.agg_ks_sample)
        cap = min(self.agg_chan_cap, x.shape[1])
        tuplas = {c: tuple(int(v) for v in np.argsort(cr[c])[::-1][:cap]) for c in cr}
        pool = sorted({c for t in tuplas.values() for c in t})
        self.pool_ = pool

        EP, starts = self._embed_pool(self._normalize(x), pool)

        ib = np.where((starts + self.win_size) <= tr)[0]
        if len(ib) < 5:
            ib = np.where(starts < max(1, tr - self.win_size))[0]
        if len(ib) > self.agg_ntrain:
            ib = np.random.RandomState(0).choice(ib, self.agg_ntrain, replace=False)

        M = np.stack([_unit(np.nan_to_num(self._to_time(
                starts, self._dist_lof_pool(self._l2(EP[c]), ib), self.win_size, n)))
              for c in pool])
        agg = M.mean(0)
        self.agg_component_ = agg

        s0_new = s0 + self.agg_delta * _unit(np.nan_to_num(agg))
        return self._fuse_contrast(s0_new)

    def _fuse_contrast(self, s0_new):
        """Local contrast on `s0_new`; with dilation_at='end' the dilation is applied here."""
        self.s0_component_ = s0_new
        if self.contrast_gamma == 0.0:
            self.contrast_component_ = None
            return self._dilate_end(s0_new)
        c = np.nan_to_num(self._contrast(s0_new))
        self.contrast_component_ = c
        return self._dilate_end(
            s0_new + self.contrast_gamma * self._winsor_unit(c, self.contrast_q))
