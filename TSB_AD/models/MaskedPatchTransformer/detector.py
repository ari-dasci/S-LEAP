"""Sklearn-like wrapper: fit(train) / decision_function(full series)."""

import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .scaling import quantile_scale, unit as _unit
from .model import MaskedPatchTransformer
from .moments import MomentCalibrator, local_moments
from .window import choose_window

# `_unit` (see scaling.py) is also imported from here by the later stages.


def _num_workers():
    """Number of DataLoader worker processes."""
    return 0


class WindowDataset(Dataset):
    """Sliding windows over a normalized array; a series shorter than one window is edge-padded."""

    def __init__(self, data, win_size, stride):
        data = np.ascontiguousarray(data, dtype=np.float32)
        self.win_size = int(win_size)
        self.stride = max(1, int(stride))
        self.n_valid = len(data)
        if len(data) < self.win_size:
            pad = self.win_size - len(data)
            data = np.concatenate(
                [data, np.repeat(data[-1:], pad, axis=0)], axis=0)
        self.data = data
        n = len(self.data)
        self.starts = list(range(0, n - self.win_size + 1, self.stride))
        # Always include the tail window so the last points get full context.
        if self.starts[-1] != n - self.win_size:
            self.starts.append(n - self.win_size)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return torch.from_numpy(self.data[s:s + self.win_size]), s


class MaskedPatchAD:
    """Stage 1 of LEAP: calibrated masked-prediction error fused with local-moment evidence."""

    # True in detectors that dilate their final score (required for dilation_at='end').
    _applies_end_dilation = False

    def __init__(self,
                 input_c=1,
                 win_size=None,            # None → from the dominant period
                 patch_size=16,
                 win_periods=2,
                 win_cap=384,
                 d_model=128,
                 n_heads=8,
                 depth=4,
                 d_ff=None,
                 dropout=0.0,
                 mask_groups=4,            # G: masked fraction is 1/G
                 mask_block=1,             # masked span, in patches
                 lr=1e-3,
                 batch_size=32,
                 num_epochs=300,
                 patience=10,
                 validation_size=0.2,
                 window_stride_div=8,      # step = win_size // this
                 min_train_windows=512,    # tighten the stride until at least this many
                 max_train_windows=4000,
                 channel_latent_threshold=128,
                 n_latents=16,
                 lse_temp=0.5,
                 chan_agg='softmax',
                 exclude_quiet_channels=False,  # do not score channels constant in train
                 level_weight=1.0,         # weight of the local-moment evidence
                 level_gate_sigma=1.0,     # gate band of the moment calibrator
                 level_window=0,           # 0 -> auto (max(8, 4 x patch_size))
                 level_combine='add',      # 'add' | 'pmean' | 'max' | 'gate' | 'off'
                 score_smooth=0,           # 0 -> auto (8 x patch_size); 1 = no smoothing
                 sigma_floor_frac=0.05,    # sigma floor = frac x median of the active channels
                 calib_shrinkage=0.3,      # shrinkage of the error spread toward the median
                 dilation_kernel=0,        # 0 → auto (max(3, win_size // 10))
                 score_smooth_scales=None,  # e.g. (1, 8, 32, 128): max_s unit(smooth_s)
                 level_windows=None,       # e.g. (16, 64, 256): max_w unit(level_w)
                 dilation_at='base',       # 'base' | 'end' (applied by the final detector)
                 amp=True,
                 component_dir=None,   # dumps the unfused evidence terms
                 component_tag='series',
                 verbose=True):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.input_c = int(input_c)
        self._auto_win = win_size is None
        self.win_size = win_size
        self.patch_size = int(patch_size)
        self.win_periods = int(win_periods)
        self.win_cap = int(win_cap)
        self._d_model = int(d_model)
        self._n_heads = int(n_heads)
        self._depth = int(depth)
        self._d_ff = d_ff
        self._dropout = float(dropout)
        self.mask_groups = int(mask_groups)
        self.mask_block = int(mask_block)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.patience = int(patience)
        self.validation_size = float(validation_size)
        self.window_stride_div = max(1, int(window_stride_div))
        self.min_train_windows = max(1, int(min_train_windows))
        self.max_train_windows = int(max_train_windows)
        self.channel_latent_threshold = int(channel_latent_threshold)
        self.n_latents = int(n_latents)
        self.lse_temp = float(lse_temp)
        self.chan_agg = chan_agg
        self.exclude_quiet_channels = bool(exclude_quiet_channels)
        self.level_weight = max(0.0, float(level_weight))
        self.level_gate_sigma = float(level_gate_sigma)
        self._level_window = int(level_window)
        if level_combine not in ('pmean', 'max', 'gate', 'add', 'off'):
            raise ValueError(f"level_combine must be 'add', 'pmean', 'max', "
                             f"'gate' or 'off', not {level_combine!r}")
        self.level_combine = level_combine
        self._score_smooth = int(score_smooth)
        self.score_smooth = None
        self.sigma_floor_frac = float(sigma_floor_frac)
        self.calib_shrinkage = float(calib_shrinkage)
        self._auto_dilation = int(dilation_kernel) == 0
        self.dilation_kernel = max(1, int(dilation_kernel))
        # Multi-scale readout (None = single scale).
        self.score_smooth_scales = (None if not score_smooth_scales
                                    else tuple(int(s) for s in score_smooth_scales))
        self.level_windows = (None if not level_windows
                              else tuple(int(w) for w in level_windows))
        if dilation_at not in ('base', 'end'):
            raise ValueError(f"dilation_at must be 'base' or 'end', not {dilation_at!r}")
        if dilation_at == 'end' and not self._applies_end_dilation:
            # 'end' requires a detector that dilates its final score.
            raise ValueError(f"{type(self).__name__} does not dilate its final score: "
                             f"use dilation_at='base'")
        self.dilation_at = dilation_at
        self.component_dir = component_dir
        self.component_tag = str(component_tag)
        self.amp = bool(amp) and torch.cuda.is_available()
        self.verbose = bool(verbose)

        self.model = None
        self.step = None
        self.data_mean = None
        self.data_std = None
        self.err_mean = None          # [C] calibrated mean masked error
        self.err_std = None           # [C]
        self.quiet_channels = None    # [C] bool, channels without variation in train
        self._score_channels = None   # [C] bool or None, channels that are scored
        self.moment_cal = None        # MomentCalibrator or None
        self.masked_component_ = None  # masked-error score, before fusion
        self.level_component_ = None   # local-moment evidence, before fusion
        self.level_window = None
        self.unit_fallback_ = None     # True if `unit` found no scale (all-zero profile)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _fit_normalizer(self, train):
        """Per-channel standardization with a sigma floor relative to the active channels."""
        self.data_mean = train.mean(axis=0)
        sigma = train.std(axis=0)
        positive = sigma[sigma > 0]
        anchor = np.median(positive) if positive.size else 1.0
        if not np.isfinite(anchor) or anchor <= 0:
            anchor = 1.0
        floor = self.sigma_floor_frac * anchor
        quiet = sigma < floor
        # Kept for the calibration, which anchors on the channels that vary.
        self.quiet_channels = quiet
        self.data_std = np.maximum(sigma, floor)
        keep = ~quiet
        # If every channel is quiet, score them all.
        self._score_channels = (
            torch.as_tensor(keep, device=self.device)
            if (self.exclude_quiet_channels and keep.any() and not keep.all())
            else None)
        if self._score_channels is not None:
            self._log(f'[INFO] scoring {int(keep.sum())}/{len(sigma)} '
                      f'channels (channels constant in train are excluded)')
        if quiet.any():
            self._log(f'[INFO] {int(quiet.sum())}/{len(sigma)} channel(s) nearly '
                      f'constant in train → sigma floor {floor:.4g} '
                      f'(median of the active channels {anchor:.4g})')

    def _normalize(self, x):
        return (x - self.data_mean) / self.data_std

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        latents = (None if self.input_c <= self.channel_latent_threshold
                   else self.n_latents)
        checkpointing = self.input_c > 128
        self.model = MaskedPatchTransformer(
            n_channels=self.input_c, patch_size=self.patch_size,
            win_size=self.win_size, d_model=self._d_model,
            n_heads=self._n_heads, depth=self._depth, d_ff=self._d_ff,
            dropout=self._dropout, n_latents=latents,
            gradient_checkpointing=checkpointing,
        ).to(self.device)
        n_par = sum(p.numel() for p in self.model.parameters())
        self._log(
            f'[INFO] model: C={self.input_c} L={self.win_size} '
            f'p={self.patch_size} N={self.model.n_patches} d={self._d_model} '
            f'depth={self._depth} | channel-attn='
            f'{"latent K=%d" % self.n_latents if latents else "full"} '
            f'| {n_par/1e6:.2f}M params')

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        self.input_c = data.shape[1]

        p = self.patch_size
        if self._auto_win:
            # Window capped at half the training segment.
            cap = min(self.win_cap, max(p * 4, (len(data) // 2 // p) * p))
            win, period = choose_window(data, p, self.win_periods, cap)
            self.win_size = win
            self._log(f'[INFO] win_size={win} '
                      + (f'(period≈{period})' if period else '(no significant period)')
                      + (f' [capped by the train length: cap={cap}]'
                         if cap < self.win_cap else ''))
        elif self.win_size % p:
            self.win_size = max(p, (self.win_size // p) * p)

        # Split first, so the stride can be chosen against the real train length.
        split = int((1 - self.validation_size) * len(data))
        if len(data) - split < self.win_size:
            split = len(data) - self.win_size
        if split < self.win_size:
            # Too short for two disjoint segments: validate on the same data.
            train = valid = data
            self._log('[INFO] short train segment: validating on the same data')
        else:
            train, valid = data[:split], data[split:]

        # Stride: win // window_stride_div, tightened to reach min_train_windows.
        self.step = max(1, self.win_size // self.window_stride_div)
        usable = len(train) - self.win_size
        if usable > 0:
            need = max(1, usable // max(1, self.min_train_windows - 1))
            self.step = max(1, min(self.step, need))
        if self._auto_dilation:
            self.dilation_kernel = max(3, self.win_size // 10)
        # Event-scale smoothing width: 8 x patch_size by default.
        self.score_smooth = (self._score_smooth if self._score_smooth > 0
                             else 8 * self.patch_size)

        self._fit_normalizer(train)
        self._build()

        train_loader = self._loader(self._normalize(train), shuffle=True,
                                    cap=self.max_train_windows)
        valid_loader = self._loader(self._normalize(valid), shuffle=False,
                                    cap=max(256, self.max_train_windows // 8))
        self._log(f'[INFO] step={self.step} | windows train='
                  f'{len(train_loader.dataset)} valid={len(valid_loader.dataset)}')

        self._train(train_loader, valid_loader)
        self._calibrate(train_loader)
        self._calibrate_moments(train)
        return self

    def _loader(self, data, shuffle, cap=None):
        ds = WindowDataset(data, self.win_size, self.step)
        if cap is not None and len(ds) > cap:
            # Evenly spaced subsample of the windows.
            idx = np.linspace(0, len(ds) - 1, cap).astype(int)
            ds.starts = [ds.starts[i] for i in idx]
        nw = _num_workers()
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=nw, persistent_workers=False,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=False)

    def _masked_loss(self, x, generator=None):
        """L1 on the masked positions of one randomly chosen group."""
        C = x.shape[-1]
        N = self.model.n_patches
        g = int(torch.randint(self.mask_groups, (1,), generator=generator).item())
        rot = int(torch.randint(self.mask_groups, (1,), generator=generator).item())
        from .masking import mask_for_group
        mask = mask_for_group(C, N, self.mask_groups, g, self.mask_block, rot,
                              device=x.device)
        pred = self.model(x, mask)
        keep = mask.repeat_interleave(self.patch_size, dim=1).t().unsqueeze(0)
        diff = (x - pred).abs()
        # Mean over the masked entries only.
        return (diff * keep).sum() / keep.expand_as(diff).sum().clamp(min=1)

    def _train(self, train_loader, valid_loader):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        gen = torch.Generator().manual_seed(2024)

        best, best_state, bad, best_epoch = np.inf, None, 0, -1
        for epoch in range(self.num_epochs):
            t0 = time.time()
            self.model.train()
            losses = []
            for batch, _ in train_loader:
                x = batch.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=self.amp):
                    loss = self._masked_loss(x, gen)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())

            val = self._validate(valid_loader)
            sched.step(val)
            if val < best - 1e-6:
                best, bad, best_epoch = val, 0, epoch
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
            if self.verbose and (epoch % 5 == 0 or bad >= self.patience):
                self._log(f'  epoch {epoch+1}/{self.num_epochs} '
                          f'train={np.mean(losses):.5f} val={val:.5f} '
                          f'({time.time()-t0:.1f}s){" *" if bad == 0 else ""}')
            if bad >= self.patience:
                self._log(f'  early stopping at epoch {epoch+1}')
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self._log(f'  restored best weights: val={best:.5f} '
                      f'(epoch {best_epoch + 1})')
        else:
            self._log('  [WARN] no epoch improved: keeping the last weights')

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        gen = torch.Generator().manual_seed(7)
        total, n = 0.0, 0
        for batch, _ in loader:
            x = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=self.amp):
                total += self._masked_loss(x, gen).item() * len(x)
            n += len(x)
        return total / max(n, 1)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _calibrate(self, loader):
        """Per-channel mean/std of the masked error on the anomaly-free training data."""
        self.model.eval()
        s1 = torch.zeros(self.input_c, dtype=torch.float64, device=self.device)
        s2 = torch.zeros_like(s1)
        n = 0
        for batch, _ in loader:
            x = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=self.amp):
                err = self.model.masked_error(x, self.mask_groups,
                                              self.mask_block)
            e = err.float().reshape(-1, self.input_c).double()
            s1 += e.sum(0)
            s2 += (e * e).sum(0)
            n += e.shape[0]
        mean = s1 / max(n, 1)
        var = (s2 / max(n, 1) - mean * mean).clamp(min=0)
        std = var.sqrt()

        # Shrinkage toward the median of the active channels, plus a spread floor.
        lam = self.calib_shrinkage
        active = torch.as_tensor(~self.quiet_channels, device=std.device)
        anchor = std[active].median() if bool(active.any()) else std.median()
        std = (1 - lam) * std + lam * anchor
        floor = torch.clamp(0.1 * anchor, min=1e-6)

        self.err_mean = mean.float()
        self.err_std = std.clamp(min=floor).float()
        self._log(f'[INFO] calibrated on {n} points | mean error '
                  f'{mean.mean().item():.4f} ± {std.mean().item():.4f} | '
                  f'shrinkage λ={lam} toward median {anchor.item():.4f}')

    def _calibrate_moments(self, train):
        """Baseline of the local moment profile over the training segment."""
        if self.level_combine == 'off' or self.level_weight <= 0:
            self.moment_cal = None
            return
        self.level_window = (self._level_window if self._level_window > 0
                             else max(8, 4 * self.patch_size))
        norm = self._normalize(train)
        m, s = local_moments(norm, self.level_window)
        self.moment_cal = MomentCalibrator(
            floor_frac=self.sigma_floor_frac,
            gate=self.level_gate_sigma).fit(m, s)
        # Multi-scale: one calibrator per window, against the train moments at that scale.
        self._level_cals = None
        if self.level_windows:
            self._level_cals = []
            for w in self.level_windows:
                mw, sw = local_moments(norm, w)
                self._level_cals.append((w, MomentCalibrator(
                    floor_frac=self.sigma_floor_frac,
                    gate=self.level_gate_sigma).fit(mw, sw)))
        self._log(f'[INFO] local moments calibrated | window='
                  f'{self.level_window} | combine={self.level_combine} '
                  f'w={self.level_weight}')

    def _level_excess(self, normalized):
        """Level/dispersion excess over the whole normalized series → [n] ≥ 0, or None."""
        if self.moment_cal is None or not self.moment_cal.ready:
            return None
        if getattr(self, '_level_cals', None):
            # Multi-scale: each window rescaled with `unit`, maximum per point.
            per_w = []
            for w, cal in self._level_cals:
                m, s = local_moments(normalized, w)
                per_w.append(_unit(np.nan_to_num(cal.excess(m, s))))
            return np.max(per_w, axis=0)
        m, s = local_moments(normalized, self.level_window)
        return self.moment_cal.excess(m, s)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _aggregate(self, z):
        """[B, L, C] standardized errors → [B, L] ≥ 0, reduced over channels with `chan_agg`."""
        mask = self._score_channels
        if mask is not None:
            z = z[..., mask]
        if self.chan_agg == 'max':
            return z.amax(dim=-1).clamp(min=0)
        if self.chan_agg == 'mean':
            return z.clamp(min=0).mean(dim=-1)
        w = F.softmax(z / self.lse_temp, dim=-1)
        return (w * z.clamp(min=0)).sum(dim=-1)

    @torch.no_grad()
    def decision_function(self, data):
        if self.model is None:
            raise RuntimeError('call fit() before decision_function()')
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        self.model.eval()

        normalized = self._normalize(data)
        n = len(normalized)
        acc = torch.zeros(n, device=self.device)
        wsum = torch.zeros(n, device=self.device)

        L = self.win_size
        # Gaussian weights: positions near a window edge count less.
        gauss = torch.exp(-0.5 * ((torch.arange(L, device=self.device,
                                                dtype=torch.float32) - L / 2)
                                  / (L / 4)) ** 2)
        offsets = torch.arange(L, device=self.device)

        loader = DataLoader(WindowDataset(normalized, L, self.step),
                            batch_size=self.batch_size, shuffle=False,
                            num_workers=_num_workers(),
                            pin_memory=torch.cuda.is_available())
        for batch, starts in loader:
            x = batch.to(self.device, non_blocking=True)
            starts = starts.to(self.device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=self.amp):
                err = self.model.masked_error(x, self.mask_groups,
                                              self.mask_block)
            z = (err.float() - self.err_mean) / self.err_std
            scored = self._aggregate(z)                       # [B, L]

            idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [B, L]
            valid = idx < n
            w = gauss.unsqueeze(0).expand_as(scored)
            acc.scatter_add_(0, idx[valid], (scored * w)[valid])
            wsum.scatter_add_(0, idx[valid], w[valid])

        scores = (acc / wsum.clamp(min=1e-8)).cpu().numpy()[:n]
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Event-scale smoothing of the masked error.
        masked_raw = scores.copy()
        from scipy.ndimage import uniform_filter1d
        if self.score_smooth_scales:
            # Multi-scale: each width rescaled with `unit`, maximum per point.
            scores = np.max([_unit(uniform_filter1d(masked_raw, size=s, mode='nearest')
                                   if s > 1 else masked_raw)
                             for s in self.score_smooth_scales], axis=0)
        elif self.score_smooth > 1:
            scores = uniform_filter1d(scores, size=self.score_smooth,
                                      mode='nearest')

        level = self._level_excess(normalized)
        # Unfused components, stored before the (non-invertible) dilation.
        self.masked_component_ = scores.copy()
        self.masked_raw_ = masked_raw
        self.level_component_ = (np.zeros_like(scores) if level is None
                                 else np.nan_to_num(level[:n]).copy())
        if self.component_dir:
            os.makedirs(self.component_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(self.component_dir, f'{self.component_tag}.npz'),
                masked=self.masked_component_, masked_raw=self.masked_raw_,
                level=self.level_component_,
                dilation=np.int32(self.dilation_kernel),
                score_smooth=np.int32(self.score_smooth))

        if level is not None and self.level_weight > 0:
            level = np.nan_to_num(level[:n], nan=0.0, posinf=0.0, neginf=0.0)
            # An all-zero evidence enters without rescaling; report it.
            self.unit_fallback_ = quantile_scale(level) is None
            if self.unit_fallback_:
                self._log('[WARN] local-moment evidence is identically zero: '
                          '`unit` finds no scale and leaves it unscaled')
            if self.level_combine == 'gate':
                # Multiplicative: the level evidence rescales the base score.
                scores = scores * (1.0 + self.level_weight * level)
            elif self.level_combine == 'pmean':
                # Quadratic mean of the two rescaled evidences.
                scores = np.sqrt(_unit(scores) ** 2
                                 + (self.level_weight * _unit(level)) ** 2)
            elif self.level_combine == 'max':
                # Pointwise maximum of the two rescaled evidences.
                scores = np.maximum(_unit(scores),
                                    self.level_weight * _unit(level))
            else:
                # 'add' (default): sum of the two rescaled evidences.
                scores = (_unit(scores)
                          + self.level_weight * _unit(level))

        if self.dilation_at == 'base':
            scores = self._dilate(scores)
        return scores

    def _dilate(self, scores):
        """Max-filter of `dilation_kernel` samples (identity if <= 1)."""
        if self.dilation_kernel > 1:
            from scipy.ndimage import maximum_filter1d
            scores = maximum_filter1d(scores, size=self.dilation_kernel)
        return scores

    def _dilate_end(self, scores):
        """Dilation of the final score, only with dilation_at='end' (identity otherwise)."""
        return self._dilate(scores) if self.dilation_at == 'end' else scores
