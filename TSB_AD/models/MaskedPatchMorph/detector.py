"""MaskedPatchMorph — Stage 2: + latent shape novelty.

score = unit(base) + beta * unit(novelty), where novelty is the 1-NN distance of each
window's latent embedding (top-variance channels, concatenated, L2-normalized) to a bank
of training windows, spread back to time steps with Gaussian overlap-add.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

from TSB_AD.models.MaskedPatchTransformer import MaskedPatchAD, WindowDataset
from TSB_AD.models.MaskedPatchTransformer.detector import _unit

BETA = 1.0          # weight of the novelty term
CHAN_CAP = 4        # highest-variance channels in the concatenated embedding
N_TRAIN = 1200      # cap on train windows kept in the k-NN bank


@torch.no_grad()
def _encode(model, x):
    """[B, L, C] -> [B, C, N, D]: the base encoder, without masking."""
    h = model.embed(model.patchify(x))
    h = h + model.pos_embed.view(1, 1, model.n_patches, -1)
    h = h + model.channel_embed.view(1, model.n_channels, 1, -1)
    return model.encoder(h)


class MaskedPatchMorphAD(MaskedPatchAD):
    """MaskedPatchAD plus latent shape novelty against the training windows."""

    def __init__(self, *args, morph_beta=BETA, morph_chan_cap=CHAN_CAP,
                 morph_ntrain=N_TRAIN, **kwargs):
        super().__init__(*args, **kwargs)
        self.morph_beta = float(morph_beta)
        self.morph_chan_cap = int(morph_chan_cap)
        self.morph_ntrain = int(morph_ntrain)
        self._morph_chan = None       # selected channels (top train variance)
        self._morph_ref = None        # NearestNeighbors fitted on the train windows
        self.base_score_ = None       # last Stage 1 score
        self.morph_component_ = None   # last novelty, before fusion

    def fit(self, train):
        super().fit(train)
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
        self._morph_ref = NearestNeighbors(n_neighbors=1).fit(emb)
        return self

    @torch.no_grad()
    def _win_embeds(self, normalized):
        """Sliding windows -> concatenated, L2-normalized embeddings → (emb, starts)."""
        loader = DataLoader(WindowDataset(normalized, self.win_size, self.step),
                            batch_size=self.batch_size, shuffle=False)
        embs, starts = [], []
        for batch, st in loader:
            h = _encode(self.model, batch.to(self.device)).float()[:, self._morph_chan]
            e = h.reshape(h.shape[0], -1)
            e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
            embs.append(e.cpu().numpy())
            starts.append(st.numpy())
        return np.concatenate(embs), np.concatenate(starts)

    @staticmethod
    def _to_time(starts, vals, L, n):
        """Spread each window's value over its time steps with Gaussian overlap-add."""
        g = np.exp(-0.5 * ((np.arange(L) - L / 2) / (L / 4)) ** 2)
        acc = np.zeros(n)
        ws = np.zeros(n)
        for s, val in zip(starts, vals):
            idx = np.arange(s, min(s + L, n))
            k = len(idx)
            acc[idx] += val * g[:k]
            ws[idx] += g[:k]
        return acc / np.clip(ws, 1e-8, None)

    @torch.no_grad()
    def decision_function(self, data):
        base = np.asarray(super().decision_function(data), dtype=np.float64).ravel()
        self.base_score_ = base
        n = len(base)
        x = np.asarray(data, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        emb, starts = self._win_embeds(self._normalize(x))
        dist, _ = self._morph_ref.kneighbors(emb)
        morph = np.nan_to_num(self._to_time(starts, dist[:, 0], self.win_size, n))
        self.morph_component_ = morph
        return _unit(base) + self.morph_beta * _unit(morph)
