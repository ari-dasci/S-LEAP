import numpy as np

Unsupervise_AD_Pool = []
Semisupervise_AD_Pool = ['MaskedPatchTransformer', 'MaskedPatchMorph', 'MaskedPatchMorphLOF',
                         'MaskedPatchMorphLOFContrast', 'MaskedPatchMorphLOFContrastAgg']


def _sync():
    """torch.cuda.synchronize() when a GPU is present, so timings include the GPU work."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _fit_and_score(clf, data_train, data_test):
    """fit() + decision_function(), timed separately → (scores, {'train': s, 'infer': s})."""
    import time
    _sync()
    t0 = time.time()
    clf.fit(data_train)
    _sync()
    t1 = time.time()
    score = clf.decision_function(data_test)
    _sync()
    t2 = time.time()
    return score.ravel(), {'train': t1 - t0, 'infer': t2 - t1}


def run_Unsupervise_AD(model_name, data, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message, None
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message, None


def run_Semisupervise_AD(model_name, data_train, data_test, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data_train, data_test, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message, None
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message, None


def run_MaskedPatchTransformer(data_train, data_test, win_size=None,
                               patch_size=16, d_model=128, n_heads=8, depth=4,
                               d_ff=None, dropout=0.0,
                               mask_groups=4, mask_block=1,
                               lr=1e-3, batch_size=32, epochs=300, patience=10,
                               win_periods=2, win_cap=384,
                               window_stride_div=8, min_train_windows=128, max_train_windows=4000,
                               channel_latent_threshold=32, n_latents=16,
                               lse_temp=0.5, chan_agg='softmax',
                               exclude_quiet_channels=False,
                               level_weight=2.0, level_gate_sigma=1.0,
                               level_window=0, level_combine='max', score_smooth=0,
                               sigma_floor_frac=0.05, calib_shrinkage=0.3,
                               dilation_kernel=0, amp=True,
                               component_dir=None, component_tag='series'):
    """Stage 1: masked-patch prediction error + local-moment evidence. See models/MaskedPatchTransformer/."""
    import torch
    import gc
    import os
    from .models.MaskedPatchTransformer import MaskedPatchAD
    # Optional evidence dumps, configured through the environment.
    component_dir = component_dir or os.environ.get('MPT_COMPONENT_DIR')
    component_tag = os.environ.get('MPT_COMPONENT_TAG', component_tag)
    clf = None
    try:
        clf = MaskedPatchAD(
            input_c=data_test.shape[1], win_size=win_size,
            patch_size=patch_size, d_model=d_model, n_heads=n_heads,
            depth=depth, d_ff=d_ff, dropout=dropout,
            mask_groups=mask_groups, mask_block=mask_block,
            lr=lr, batch_size=batch_size, num_epochs=epochs,
            patience=patience, win_periods=win_periods, win_cap=win_cap,
            window_stride_div=window_stride_div,
            min_train_windows=min_train_windows,
            max_train_windows=max_train_windows,
            channel_latent_threshold=channel_latent_threshold,
            n_latents=n_latents, lse_temp=lse_temp, chan_agg=chan_agg,
            exclude_quiet_channels=exclude_quiet_channels,
            level_weight=level_weight,
            level_gate_sigma=level_gate_sigma,
            level_window=level_window,
            level_combine=level_combine,
            score_smooth=score_smooth,
            sigma_floor_frac=sigma_floor_frac,
            calib_shrinkage=calib_shrinkage,
            dilation_kernel=dilation_kernel, amp=amp,
            component_dir=component_dir, component_tag=component_tag)
        return _fit_and_score(clf, data_train, data_test)
    finally:
        del clf
        gc.collect()
        torch.cuda.empty_cache()


def run_MaskedPatchMorph(data_train, data_test, win_size=None,
                         patch_size=16, d_model=128, n_heads=8, depth=4,
                         d_ff=None, dropout=0.0,
                         mask_groups=4, mask_block=1,
                         lr=5e-4, batch_size=32, epochs=300, patience=10,
                         win_periods=2, win_cap=384,
                         window_stride_div=8, min_train_windows=512,
                         max_train_windows=4000,
                         channel_latent_threshold=128, n_latents=16,
                         lse_temp=0.5, chan_agg='softmax',
                         exclude_quiet_channels=False,
                         level_weight=1.0, level_gate_sigma=1.0,
                         level_window=0, level_combine='add', score_smooth=0,
                         sigma_floor_frac=0.05, calib_shrinkage=0.0,
                         dilation_kernel=0, amp=True,
                         morph_beta=0.6, morph_chan_cap=4, morph_ntrain=1200):
    """Stage 2: + latent shape novelty. See models/MaskedPatchMorph/."""
    import torch
    import gc
    from .models.MaskedPatchMorph import MaskedPatchMorphAD
    clf = None
    try:
        clf = MaskedPatchMorphAD(
            input_c=data_test.shape[1], win_size=win_size,
            patch_size=patch_size, d_model=d_model, n_heads=n_heads,
            depth=depth, d_ff=d_ff, dropout=dropout,
            mask_groups=mask_groups, mask_block=mask_block,
            lr=lr, batch_size=batch_size, num_epochs=epochs,
            patience=patience, win_periods=win_periods, win_cap=win_cap,
            window_stride_div=window_stride_div,
            min_train_windows=min_train_windows,
            max_train_windows=max_train_windows,
            channel_latent_threshold=channel_latent_threshold,
            n_latents=n_latents, lse_temp=lse_temp, chan_agg=chan_agg,
            exclude_quiet_channels=exclude_quiet_channels,
            level_weight=level_weight, level_gate_sigma=level_gate_sigma,
            level_window=level_window, level_combine=level_combine,
            score_smooth=score_smooth, sigma_floor_frac=sigma_floor_frac,
            calib_shrinkage=calib_shrinkage, dilation_kernel=dilation_kernel,
            amp=amp,
            morph_beta=morph_beta, morph_chan_cap=morph_chan_cap,
            morph_ntrain=morph_ntrain)
        return _fit_and_score(clf, data_train, data_test)
    finally:
        del clf
        gc.collect()
        torch.cuda.empty_cache()


def run_MaskedPatchMorphLOF(data_train, data_test, win_size=None,
                            patch_size=16, d_model=128, n_heads=8, depth=4,
                            d_ff=None, dropout=0.0,
                            mask_groups=4, mask_block=1,
                            lr=5e-4, batch_size=32, epochs=300, patience=10,
                            win_periods=2, win_cap=384,
                            window_stride_div=8, min_train_windows=512,
                            max_train_windows=4000,
                            channel_latent_threshold=128, n_latents=16,
                            lse_temp=0.5, chan_agg='softmax',
                            exclude_quiet_channels=False,
                            level_weight=1.0, level_gate_sigma=1.0,
                            level_window=0, level_combine='add', score_smooth=0,
                            sigma_floor_frac=0.05, calib_shrinkage=0.0,
                            dilation_kernel=0, amp=True,
                            morph_beta=1.0, morph_chan_cap=4, morph_ntrain=1200,
                            morph_lof_k=4, morph_lof_alpha=0.25):
    """Stage 3: + local-density calibration of the latent novelty. See models/MaskedPatchMorphLOF/."""
    import torch
    import gc
    from .models.MaskedPatchMorphLOF import MaskedPatchMorphLOFAD
    clf = None
    try:
        clf = MaskedPatchMorphLOFAD(
            input_c=data_test.shape[1], win_size=win_size,
            patch_size=patch_size, d_model=d_model, n_heads=n_heads,
            depth=depth, d_ff=d_ff, dropout=dropout,
            mask_groups=mask_groups, mask_block=mask_block,
            lr=lr, batch_size=batch_size, num_epochs=epochs,
            patience=patience, win_periods=win_periods, win_cap=win_cap,
            window_stride_div=window_stride_div,
            min_train_windows=min_train_windows,
            max_train_windows=max_train_windows,
            channel_latent_threshold=channel_latent_threshold,
            n_latents=n_latents, lse_temp=lse_temp, chan_agg=chan_agg,
            exclude_quiet_channels=exclude_quiet_channels,
            level_weight=level_weight, level_gate_sigma=level_gate_sigma,
            level_window=level_window, level_combine=level_combine,
            score_smooth=score_smooth, sigma_floor_frac=sigma_floor_frac,
            calib_shrinkage=calib_shrinkage, dilation_kernel=dilation_kernel,
            amp=amp,
            morph_beta=morph_beta, morph_chan_cap=morph_chan_cap,
            morph_ntrain=morph_ntrain,
            morph_lof_k=morph_lof_k, morph_lof_alpha=morph_lof_alpha)
        return _fit_and_score(clf, data_train, data_test)
    finally:
        del clf
        gc.collect()
        torch.cuda.empty_cache()


def run_MaskedPatchMorphLOFContrast(data_train, data_test, win_size=None,
                                    patch_size=16, d_model=128, n_heads=8, depth=4,
                                    d_ff=None, dropout=0.0,
                                    mask_groups=4, mask_block=1,
                                    lr=5e-4, batch_size=32, epochs=400, patience=10,
                                    win_periods=2, win_cap=384,
                                    window_stride_div=8, min_train_windows=512,
                                    max_train_windows=4000,
                                    channel_latent_threshold=128, n_latents=16,
                                    lse_temp=0.5, chan_agg='softmax',
                                    exclude_quiet_channels=False,
                                    level_weight=1.0, level_gate_sigma=1.0,
                                    level_window=0, level_combine='add', score_smooth=0,
                                    sigma_floor_frac=0.05, calib_shrinkage=0.0,
                                    dilation_kernel=0, amp=True,
                                    morph_beta=1.0, morph_chan_cap=4, morph_ntrain=1200,
                                    morph_lof_k=4, morph_lof_alpha=0.25,
                                    contrast_w=None, contrast_w_mult=64,
                                    contrast_gamma=0.5, contrast_q=0.99):
    """Stage 4: + local temporal contrast. See models/MaskedPatchMorphLOFContrast/."""
    import torch
    import gc
    from .models.MaskedPatchMorphLOFContrast import MaskedPatchMorphLOFContrastAD
    clf = None
    try:
        clf = MaskedPatchMorphLOFContrastAD(
            input_c=data_test.shape[1], win_size=win_size,
            patch_size=patch_size, d_model=d_model, n_heads=n_heads,
            depth=depth, d_ff=d_ff, dropout=dropout,
            mask_groups=mask_groups, mask_block=mask_block,
            lr=lr, batch_size=batch_size, num_epochs=epochs,
            patience=patience, win_periods=win_periods, win_cap=win_cap,
            window_stride_div=window_stride_div,
            min_train_windows=min_train_windows,
            max_train_windows=max_train_windows,
            channel_latent_threshold=channel_latent_threshold,
            n_latents=n_latents, lse_temp=lse_temp, chan_agg=chan_agg,
            exclude_quiet_channels=exclude_quiet_channels,
            level_weight=level_weight, level_gate_sigma=level_gate_sigma,
            level_window=level_window, level_combine=level_combine,
            score_smooth=score_smooth, sigma_floor_frac=sigma_floor_frac,
            calib_shrinkage=calib_shrinkage, dilation_kernel=dilation_kernel,
            amp=amp,
            morph_beta=morph_beta, morph_chan_cap=morph_chan_cap,
            morph_ntrain=morph_ntrain,
            morph_lof_k=morph_lof_k, morph_lof_alpha=morph_lof_alpha,
            contrast_w=contrast_w, contrast_w_mult=contrast_w_mult,
            contrast_gamma=contrast_gamma, contrast_q=contrast_q)
        return _fit_and_score(clf, data_train, data_test)
    finally:
        del clf
        gc.collect()
        torch.cuda.empty_cache()


def run_MaskedPatchMorphLOFContrastAgg(data_train, data_test, win_size=None,
                                       patch_size=16, d_model=128, n_heads=8, depth=4,
                                       d_ff=None, dropout=0.0,
                                       mask_groups=4, mask_block=1,
                                       lr=5e-4, batch_size=32, epochs=400, patience=10,
                                       win_periods=2, win_cap=384,
                                       window_stride_div=8, min_train_windows=512,
                                       max_train_windows=4000,
                                       channel_latent_threshold=128, n_latents=16,
                                       lse_temp=0.5, chan_agg='softmax',
                                       exclude_quiet_channels=False,
                                       level_weight=1.0, level_gate_sigma=1.0,
                                       level_window=0, level_combine='add', score_smooth=0,
                                       sigma_floor_frac=0.05, calib_shrinkage=0.0,
                                       dilation_kernel=0, amp=True,
                                       morph_beta=1.0, morph_chan_cap=4, morph_ntrain=1200,
                                       morph_lof_k=4, morph_lof_alpha=0.25,
                                       contrast_w=None, contrast_w_mult=64,
                                       contrast_gamma=0.5, contrast_q=0.99,
                                       agg_delta=0.5, agg_chan_cap=4, agg_ntrain=1200,
                                       agg_ks_sample=20000, agg_chunk=4,
                                       score_smooth_scales=None, level_windows=None,
                                       dilation_at='base'):
    """Stage 5 (LEAP): + per-channel novelty over a criterion-based channel pool. See models/MaskedPatchMorphLOFContrastAgg/."""
    import torch
    import gc
    from .models.MaskedPatchMorphLOFContrastAgg import MaskedPatchMorphLOFContrastAggAD
    clf = None
    try:
        clf = MaskedPatchMorphLOFContrastAggAD(
            input_c=data_test.shape[1], win_size=win_size,
            patch_size=patch_size, d_model=d_model, n_heads=n_heads,
            depth=depth, d_ff=d_ff, dropout=dropout,
            mask_groups=mask_groups, mask_block=mask_block,
            lr=lr, batch_size=batch_size, num_epochs=epochs,
            patience=patience, win_periods=win_periods, win_cap=win_cap,
            window_stride_div=window_stride_div,
            min_train_windows=min_train_windows,
            max_train_windows=max_train_windows,
            channel_latent_threshold=channel_latent_threshold,
            n_latents=n_latents, lse_temp=lse_temp, chan_agg=chan_agg,
            exclude_quiet_channels=exclude_quiet_channels,
            level_weight=level_weight, level_gate_sigma=level_gate_sigma,
            level_window=level_window, level_combine=level_combine,
            score_smooth=score_smooth, sigma_floor_frac=sigma_floor_frac,
            calib_shrinkage=calib_shrinkage, dilation_kernel=dilation_kernel,
            amp=amp,
            morph_beta=morph_beta, morph_chan_cap=morph_chan_cap,
            morph_ntrain=morph_ntrain,
            morph_lof_k=morph_lof_k, morph_lof_alpha=morph_lof_alpha,
            contrast_w=contrast_w, contrast_w_mult=contrast_w_mult,
            contrast_gamma=contrast_gamma, contrast_q=contrast_q,
            agg_delta=agg_delta, agg_chan_cap=agg_chan_cap, agg_ntrain=agg_ntrain,
            agg_ks_sample=agg_ks_sample, agg_chunk=agg_chunk,
            score_smooth_scales=score_smooth_scales, level_windows=level_windows,
            dilation_at=dilation_at)
        return _fit_and_score(clf, data_train, data_test)
    finally:
        del clf
        gc.collect()
        torch.cuda.empty_cache()
