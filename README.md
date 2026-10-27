# S-LEAP - Latent Embedding Anomaly Patch-Transformer

![License](https://img.shields.io/badge/AGPL--3.0-red?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

**LEAP** (implementation name: `MaskedPatchMorphLOFContrastAgg`) is a
semi-supervised time-series anomaly detector that replaces reconstruction with
**masked patch prediction**. Patches are hidden from the encoder and predicted
from the remaining temporal and cross-channel context, so the prediction error
measures how well the learned normal dynamics explain each observation, and
the model cannot simply copy an anomaly back. On top of that error LEAP adds
multiple label-free evidence sources: density-calibrated latent novelty, local
temporal contrast and per-channel novelty.

Under the TSB-AD protocol [6], with VUS-PR as the primary metric and three random
seeds:

- **TSB-AD-M** (180 held-out multivariate series): **0.5179 ± 0.0022**, a
  24.5 % improvement over the strongest locally measured baseline (PaAno [2],
  0.4161).
- **TSB-AD-U** (350 held-out univariate series): **0.6194 ± 0.0016** with the
  multi-scale univariate variant, ahead of all 20 locally measured baselines,
  including PaAno<sub>PAI</sub> (+6.1 %).

The name reflects the four ideas that carry the model: novelty is measured
in the **Latent** space of the encoder, via **Embedding** distance, to
score an **Anomaly** relative to windows the model has seen before, on top
of a masked-**Patch** **Transformer** base.

## Results

Mean VUS-PR ± sample standard deviation over three seeds (2024, 2025, 2026).
All rows were measured locally under the same protocol and file lists; the
paper also reports published-only results (e.g. VACE [1], 0.48 on TSB-AD-M).
The full comparison in the paper covers 21 baselines on TSB-AD-M and 23 on
TSB-AD-U, among them KAN-AD [3] and CrossAD [4].

| TSB-AD-M (180 series)    | VUS-PR              |     | TSB-AD-U (350 series)    | VUS-PR              |
|--------------------------|---------------------|-----|--------------------------|---------------------|
| **LEAP**                 | **0.5179 ± 0.0022** |     | **LEAP**                 | **0.6194 ± 0.0016** |
| PaAno [2]                | 0.4161 ± 0.0064     |     | PaAno<sub>PAI</sub>      | 0.5839 ± 0.0016     |
| PaAno<sub>PAI</sub>      | 0.4045 ± 0.0024     |     | PaAno [2]                | 0.5165 ± 0.0025     |
| USAD                     | 0.4028 ± 0.0019     |     | PatchTST                 | 0.4569 ± 0.0025     |
| xLSTMAD-R [5]            | 0.3881 ± 0.0011     |     | Sub-PCA                  | 0.4173 ± 0.0000     |
| TimesNet                 | 0.3874 ± 0.0022     |     | USAD                     | 0.3959 ± 0.0017     |

A Bayesian signed-rank test (ROPE ± 0.05 VUS-PR) gives LEAP a posterior
probability of superiority above 99.9 % against each of the five strongest
measured baselines on both tracks.

## Model Stages

LEAP is built as five nested stages. Each stage adds one term to the score
and never discards the evidence of the previous ones. Setting its
coefficient to zero reduces the model exactly to its predecessor, so the
chain doubles as a controlled ablation.

| Stage | Adds                                          | Folder                           | Removed by | VUS-PR     | Δ       | W/L/T     | p-value |
|:-----:|-----------------------------------------------|----------------------------------|:----------:|:----------:|:-------:|:---------:|:-------:|
| 1     | Masked-patch prediction + local moments       | `MaskedPatchTransformer`         | —          | 0.4611     | —       | —         | —       |
| 2     | Latent shape novelty                          | `MaskedPatchMorph`               | β = 0      | 0.4954     | +0.0343 | 113/55/12 | 1.2e-5  |
| 3     | Local-density calibration of the novelty      | `MaskedPatchMorphLOF`            | α = 0      | 0.5034     | +0.0080 | 84/42/54  | 9.5e-5  |
| 4     | Local temporal contrast                       | `MaskedPatchMorphLOFContrast`    | γ = 0      | 0.5074     | +0.0040 | 87/35/58  | 2.8e-6  |
| 5     | Per-channel novelty over a criterion pool     | `MaskedPatchMorphLOFContrastAgg` | δ = 0      | **0.5171** | +0.0097 | 90/51/39  | 3.4e-5  |

TSB-AD-M held-out set, single seed (2024). Δ and W/L/T are against the
previous stage; p-values are two-sided paired Wilcoxon signed-rank tests on
per-series VUS-PR. All four increments stay significant after Bonferroni
correction.

1. **Masked-patch base.** Each window is split into a channel × time grid of
   patches. A deterministic complementary group mask, staggered across
   channels, hides one group per pass. Every time step is therefore predicted
   exactly once, always from context, with the same visible fraction in
   training and inference. Errors are calibrated per channel, aggregated
   with a soft maximum, smoothed at event scale and summed with local-moment
   evidence, which captures sustained level and dispersion shifts.
2. **Latent novelty.** Distance of a window's encoder embedding (top-variance
   channels, concatenated, ℓ2-normalized) to a bank of up to 1,200 training
   windows. It catches windows that are locally predictable but globally
   unfamiliar.
3. **Density calibration.** That distance is divided by the local spacing of
   the bank (LOF-style), so a sparse but legitimate region of the normal
   manifold is not scored as anomalous.
4. **Local contrast.** Each point is compared with a running median of the
   score instead of the whole series, which improves precision on long,
   non-stationary recordings.
5. **Per-channel pool.** Latent novelty is computed per channel over the
   union of channels picked by six label-free criteria (train variance, rate
   above the train p99, range change, mean absolute z-score, std change,
   train-vs-test KS statistic) and averaged. This recovers anomalies in quiet
   channels that the variance-based pick of Stage 2 misses.

Everything is label-free: fitting and scoring use only the series and its
known train/test split, never the ground-truth labels. The `detector.py`
docstring of each folder explains what changed from the class it extends.

## Hyperparameters

A single configuration per track is used for all held-out series, with no per-series or per-dataset tuning. All design choices, including architectural configurations, structural parameters, and fusion coefficients, were evaluated on the TSB-AD tuning subsets (20 multivariate / 48 univariate series). The resulting configurations were fixed before held-out evaluation, while adaptive quantities were computed per series using predefined rules based on the dominant period or patch length, without further optimization. The values below correspond to those reported in the paper and are implemented in the MaskedPatchMorphLOFContrastAgg entries of TSB_AD/HP_list.py. The last column identifies the corresponding key in HP_list.py or the detector constructor; fixed denotes a constant defined directly in the detector code.

### Shared by both tracks

| Group        | Symbol             | Meaning                                    | Value                                  | Key                                       |
|--------------|--------------------|--------------------------------------------|----------------------------------------|-------------------------------------------|
| Architecture | p                  | patch length                               | 16                                     | `patch_size`                              |
|              | D                  | model width                                | 128                                    | `d_model`                                 |
|              | H                  | attention heads                            | 8                                      | `n_heads`                                 |
|              | B                  | encoder blocks                             | 4                                      | `depth`                                   |
|              | d<sub>ff</sub>     | feed-forward width                         | 4D = 512                               | `d_ff` (`None` → 4D)                      |
|              | —                  | dropout                                    | 0.0                                    | `dropout`                                 |
| Channel axis | C<sub>lat</sub>    | latent-bottleneck threshold                | 128                                    | `channel_latent_threshold`                |
|              | K                  | number of channel latents                  | 16                                     | `n_latents`                               |
|              | τ                  | channel-aggregation temperature (softmax)  | 0.5                                    | `lse_temp`, `chan_agg='softmax'`          |
| Masking      | G                  | masking groups (masked fraction 1/G)       | 4                                      | `mask_groups`                             |
|              | b                  | mask block, in patches                     | 1                                      | `mask_block`                              |
| Windowing    | L                  | window length                              | min(2P̂, 384), multiple of p            | `win_periods`, `win_cap`                  |
|              | Δ                  | window stride                              | L/8                                    | `window_stride_div`                       |
|              | —                  | min. / max. training windows               | 512 / 4000                             | `min_train_windows`, `max_train_windows`  |
|              | κ                  | dilation width                             | L/10                                   | `dilation_kernel=0` (auto)                |
| Optimization | η<sub>0</sub>      | learning rate (AdamW)                      | 5 × 10⁻⁴                               | `lr`                                      |
|              | —                  | weight decay                               | 10⁻⁴                                   | *fixed*                                   |
|              | —                  | batch size                                 | 32                                     | `batch_size`                              |
|              | —                  | max. epochs / early-stopping patience      | 400 / 10                               | `epochs`, `patience`                      |
|              | —                  | LR decay factor / patience                 | 0.5 / 3                                | *fixed*                                   |
|              | —                  | gradient clipping / validation split       | 1.0 / 0.2                              | *fixed* / `validation_size`               |
| Calibration  | ρ<sub>σ</sub>      | degenerate-channel floor fraction          | 0.05                                   | `sigma_floor_frac`                        |
|              | λ                  | error-spread shrinkage                     | 0.0                                    | `calib_shrinkage`                         |
|              | ρ<sub>f</sub>      | error-spread floor fraction                | 0.1                                    | *fixed*                                   |
|              | ρ<sub>ω</sub>      | edge-weight Gaussian fraction              | 0.25                                   | *fixed*                                   |
|              | ρ<sub>γ</sub>      | contrast denominator floor                 | 10⁻³                                   | *fixed*                                   |
| Fusion       | w<sub>ℓ</sub>      | local-moment weight (`add`)                | 1.0                                    | `level_weight`, `level_combine='add'`     |
|              | β                  | latent-novelty weight                      | 1.0                                    | `morph_beta`                              |
|              | χ                  | channels in the concatenated embedding     | 4                                      | `morph_chan_cap`                          |
|              | N<sub>bank</sub>   | latent bank size                           | 1200                                   | `morph_ntrain`                            |
|              | k, α               | LOF neighbours, local-density weight       | 4, 0.25                                | `morph_lof_k`, `morph_lof_alpha`          |
|              | ν                  | contrast width, W<sub>γ</sub> = νL         | 64                                     | `contrast_w_mult`                         |
|              | χ′                 | channels per criterion in the pool         | 4                                      | `agg_chan_cap`                            |

### Multivariate model (TSB-AD-M)

The full five-stage LEAP.

| Symbol         | Meaning                            | Value               | Key                         |
|----------------|------------------------------------|---------------------|-----------------------------|
| W<sub>s</sub>  | score smoothing width              | 8p = 128            | `score_smooth=0` (auto)     |
| W<sub>m</sub>  | local-moment width                 | max(8, 4p) = 64     | `level_window=0` (auto)     |
| γ              | local-contrast weight              | 1.5                 | `contrast_gamma`            |
| δ              | per-channel novelty weight         | 1.5                 | `agg_delta`                 |

### Univariate model (TSB-AD-U)

With a single channel (C = 1), LEAP is adapted in three ways:

1. **Multi-scale readout.** The masked-prediction error is smoothed at several
   widths and the local moments are computed at several windows. Each scale is
   rescaled to unit range, the maximum response is kept per time step, and
   tolerance dilation is applied afterwards. This avoids a fixed width, whose
   optimum varied independently of series length, window size and training
   duration.
2. **No local contrast** (γ = 0): it adds no signal beyond the multi-scale
   scoring.
3. **No channel pool** (δ = 0): the pool reduces to the only channel, which
   makes Stage 5 redundant with Stage 3. Latent novelty embeds that channel,
   with density calibration unchanged.

| Symbol         | Meaning                                   | Value               | Key                                    |
|----------------|-------------------------------------------|---------------------|----------------------------------------|
| W<sub>s</sub>  | score smoothing widths                    | {1, 8, 32, 128}     | `score_smooth_scales=(1, 8, 32, 128)`  |
| W<sub>m</sub>  | local-moment widths                       | {16, 64, 256}       | `level_windows=(16, 64, 256)`          |
| κ              | tolerance dilation, on the final score    | L/10                | `dilation_at='end'`                    |
| γ              | local-contrast weight                     | 0                   | `contrast_gamma=0.0`                   |
| δ              | per-channel novelty weight                | 0                   | `agg_delta=0.0`                        |


## Layout

```
/
  run_detector_m.py     entry point for TSB-AD-M (multivariate)
  run_detector_u.py     entry point for TSB-AD-U (univariate)
  TSB_AD/
    model_wrapper.py    dispatch: run_<Model>(data_train, data_test, **hp)
    HP_list.py          hyperparameter dictionaries per model and track
    evaluation/         get_metrics (VUS-PR and the rest)
    utils/              sliding-window period estimation, etc.
    models/
      MaskedPatchTransformer/           Stage 1: masked-patch base + local moments
      MaskedPatchMorph/                 Stage 2: + latent shape novelty
      MaskedPatchMorphLOF/              Stage 3: + density calibration
      MaskedPatchMorphLOFContrast/      Stage 4: + local contrast
      MaskedPatchMorphLOFContrastAgg/   Stage 5: + per-channel pool (LEAP)
  Datasets/File_List/   file lists the entry scripts default to
```

## Usage

```bash
python run_detector_m.py --dataset_dir <path to TSB-AD-M CSVs> \
                         --file_list <file list CSV> [--seed 2024] [--save]

python run_detector_u.py --dataset_dir <path to TSB-AD-U CSVs> \
                         --file_list <file list CSV> [--seed 2024] [--save]
```

Each data CSV: last column `Label`, filename encodes the train/test split
as `..._tr_<N>_...csv` (the first N rows are training). The default file
lists are TSB-AD evaluation lists, `TSB-AD-M-Eva.csv` and `TSB-AD-U-Eva.csv`.
The data CSVs are distributed with TSB-AD [6].

`run_detector_m.py` prints the metrics of each file and their mean; `--save`
also writes per-file scores (`.npy`) and a metrics table (`.csv`) under
`eval/`. `run_detector_u.py` always writes the per-file scores and skips files
already scored; `--save` adds the metrics table. Both scripts record total,
training and inference time per series.

To run an earlier stage of the chain, pass `--AD_Name` (e.g.
`MaskedPatchMorphLOF`); its hyperparameters come from `HP_list.py`.

## Requirements and Reproducibility

Python 3.11, `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`,
`statsmodels`.

- The paper's runs used PyTorch 2.10.0, CUDA 12.8 and cuDNN 9.10.2 on an NVIDIA
Tesla V100 (32 GB), with a peak GPU memory of 15 GB.

- **Seeds.** The paper averages seeds 2024, 2025 and 2026. Both scripts take
  `--seed` (default 2024), which seeds PyTorch, NumPy and Python's `random`
  and enables deterministic cuDNN; run them once per seed to reproduce the
  three-seed means.
- **Tuning.** Architecture and fusion coefficients were chosen only on the
  TSB-AD tuning subsets; the held-out sets were used only for the final
  results. Each stage was accepted only with a positive mean gain and a
  significant paired Wilcoxon test, against a recomputed reference run.

- **LEAP** (`MaskedPatchMorphLOFContrastAgg`) is the final model of the line.
Its multivariate and univariate entries in `HP_list.py` match the
configuration reported in the paper. Note: `amp: True` enables BF16 autocast on CUDA, and
 `run_detector_m.py` enables TF32, whereas the paper's runs are reported in
  FP32. Set `amp: False` to run in FP32; the TF32 flags only take effect on
  Ampere or newer GPUs.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{moyamota2026leaplatentembeddinganomaly,
      title={LEAP: a Latent Embedding Anomaly Patch-Transformer for time-series anomaly detection}, 
      author={Cristhian Moya-Mota and Ignacio Aguilera-Martos and Diego García-Gil and Julián Luengo},
      year={2026},
      eprint={2610.35190},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.35190}, 
}
```

DaSCI Institute, University of Granada.

## References

Top competitors and sources for TSB-AD datasets and algorithms:

[1] A. D. Cencillo, L. Concepción, I. Triguero, J. Luengo, VACE: Learning geometrically structured representations for time series anomaly detection, arXiv:2605.23504, 2026 (best published detector on TSB-AD-M).

[2] J. Park, S. Kang, PaAno: Patch-based representation learning for time-series anomaly detection, in: Proc. ICLR, 2026.

[3] Q. Zhou et al., KAN-AD: Time series anomaly detection with Kolmogorov–Arnold networks, in: Proc. ICML, PMLR 267, 2025, pp. 79136–79149.

[4] B. Li et al., CrossAD: Time series anomaly detection with cross-scale associations and cross-window modeling, in: Advances in NeurIPS, 2025.

[5] K. Faber et al., xLSTMAD: A powerful xLSTM-based method for anomaly detection, in: Proc. IEEE ICDM, 2025.

[6] Q. Liu, J. Paparrizos, The elephant in the room: Towards a reliable time-series anomaly detection benchmark, in: Advances in NeurIPS 37, Datasets and Benchmarks Track, 2024. Benchmark, datasets and evaluation code: <https://github.com/TheDatumOrg/TSB-AD>. The entry scripts and the evaluation code in `TSB_AD/` are adapted from that repository (Apache-2.0).
