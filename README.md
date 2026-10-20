# Background-Input Leakage in Video Foreground Segmentation

Repository: https://github.com/TompsonL/gnet-leakage-audit
Archived DOI: https://doi.org/10.5281/zenodo.23493953

Code and per-video results for the audit paper *"Background-Input Leakage in Video
Foreground Segmentation: An Empirical Audit and a Frozen Causal Protocol"*.

This package contains the controlled testbed (a reproduced GNet), the frozen
causal evaluation protocol, the leakage-control and robustness scripts, the
video splits, and machine-readable per-video results. It does **not** contain the
datasets or trained checkpoints.

## Layout

```
train.py eval.py train_real.py bench.py diag_oracle_bg.py eval_subsense.py
models/     GNet, TSAM, reliability gate and boundary module
data/       CDnet/SBI loaders, causal background estimators, synthetic engine
utils/      metrics, frozen protocol, boundary F1, losses
scripts/    experiment orchestration and analysis
splits/     video-level train/holdout/CV/A/B splits
logs/       per-video CSV + JSON sidecars (protocol metadata) from the paper
paper/      LaTeX source and compiled PDF
REPRODUCE.md  full environment, data, training and evaluation instructions
```

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python3 -m pytest -q tests        # 9 protocol/metrics/background tests
python3 eval.py --weights /path/to/last.pt --root /path/to/CDnet14 \
  --bg-source causal --frame-stride 1 --out logs/eval.csv
```

Datasets follow their original licences and are not redistributed. See
`REPRODUCE.md` for the expected directory layout and download scripts. Training
commands are in `REPRODUCE.md` (about 80 min per model on one NVIDIA L20).

## Headline numbers (CDnet 2014, category-average F1)

| Setting | F1 |
|---|---:|
| SuBSENSE (packaged masks) | 0.7287 |
| Current-frame-only network | 0.2209 |
| Synthetic GNet, causal background, 3 seeds | 0.2901 ± 0.0414 |
| + reliability gate / boundary / full | 0.2404 / 0.2491 / 0.2603 |
| Non-learning causal residual threshold | 0.3918 |
| Domain GNet, causal background (53 videos) | 0.5607 |

Same-weight background switching (4 held-out categories, stride 3): the domain
model drops from 0.6652 (leaky) to 0.4511 (causal), while the synthetic model
changes by 0.013. The causal column is unchanged at stride 1 (0.4526 / 0.3297).
See `paper/main.pdf` and `logs/` for the full tables.

## Notes

- Scripts resolve their repository root dynamically; datasets are expected under
  `datasets/` as described in `REPRODUCE.md`.
- `scripts/eval_causal_med.sh` is deprecated: its per-frame background/reliability
  cache needs about 340 GB on full CDnet. The L2a comparison is reported from the
  small-sample tuning script `scripts/tune_bg.py`.
- No third-party manuscript or report is redistributed with this package.

## License

Code, scripts, splits and per-video results are released under the MIT
License (see `LICENSE`). The manuscript text and figures in `paper/` are
copyright (c) 2026 Jichao Liu and are not covered by the MIT license; they are
shared for review and reproduction.

## Contact

Please open an issue or contact the authors for questions about the protocol or
the per-video results.
