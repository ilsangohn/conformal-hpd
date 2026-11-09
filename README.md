# Coverage and shape of conformal highest-density sets

Code and results for the paper

> Ilsang Ohn. *Coverage and shape of conformal highest-density sets: a minimax theory.* 2026.

The code implements two-stage calibration of highest-density predictive sets (Algorithm 1 of the paper), one-stage
calibration (HPD-split) and the baselines of the numerical experiments, and it reproduces every table and figure of the
paper and its supplement. It uses only NumPy, SciPy and Matplotlib (tested with Python 3.11, NumPy 1.23, SciPy 1.11 and
Matplotlib 3.8).

## Layout

| Folder | Contents |
| --- | --- |
| `sim/` | Simulation engine (`sim_rank.py`), bike-sharing application (`realdata_rank.py`), table and figure scripts, and the modules they import |
| `scripts/` | Job scripts with the settings of the paper |
| `results/` | Result files of all runs used in the paper (one JSON line per method; `.err` holds the run log and its resource use) |
| `data/` | UCI Bike Sharing Dataset, hourly records (`hour.csv`) |
| `figures/`, `tables/` | Figures and tables of the paper, as produced by the scripts below |

## Reproducing the tables and figures from the saved results

Run from the repository root:

```sh
python sim/make_tables_main.py                     # tables of the main experiment (tables/main_*.tex)
python sim/fig_main.py                             # coverage error and shape loss (figures/main_plane.pdf)
python sim/fig_cov_curves.py                       # coverage along a segment (figures/cov_curves_main.pdf)
FIG1_DIR=results/fig1_main_srv python sim/fig_residual_main_plot.py   # coverage residual (figures/residual_fig_main.pdf)
python sim/fig_alg_rank.py                         # illustration of Algorithm 1 (figures/alg_rank.pdf)
(cd sim && python fig_sep_smooth.py)               # smooth perturbation of Theorem 7.2 (figures/sep_smooth.pdf)
CELL_RCP=results/bike_1031_cellrcp python sim/make_table_bike_rank.py \
    results/bike_1031 tables/table3.tex results/bike_summary.json   # bike-sharing table
python sim/realdata_rank_fig.py                    # bike-sharing sets; refits the first split (figures/bike_sets.pdf)
```

## Rerunning the experiments

Each call runs one job on one core; the jobs are independent and can run in parallel.

```sh
scripts/run_main.sh MODEL DY SEED     # main experiment; MODEL in {gaussian, multimodal, manifold}, DY in {2, 5}, SEED in 20..39
scripts/run_curves.sh MODEL DY SEED   # coverage along the segment, after run_main.sh for the same job
scripts/run_bike.sh SPLIT             # bike-sharing application, SPLIT in 0..9
FIG1_OUT=results/fig1_main_srv python sim/fig_residual_main.py SEED   # residual figure, SEED in 20..29
```

A job of the main experiment takes about 0.3 to 1 hour for `DY=2` and 1 to 12 hours for `DY=5`, with at most about
5 GB of memory. The simulation models, sample sizes and tuning grids are described in Appendix G of the supplement and
in the docstrings of `sim/sim_rank.py` and `sim/realdata_rank.py`.

Notes on the saved results:

- The Gaussian model is in `results/main_n5k_gaussrej`, the other two models in `results/main_n5k_matched`;
  `sim/main_data.py` merges them.
- The coverage-curve runs in `results/curves_n5k_matched` reuse the fits of the main experiment and read its one-stage
  thresholds from the main result files (`TAU_FILE` in `scripts/run_curves.sh`).
- For the bike-sharing table, the partition rows of two-stage (quantile) come from the partition-only runs in
  `results/bike_1031_cellrcp`; the current `sim/realdata_rank.py` also writes this row in a full run.
- Some modules (`sim_cov.py`, `sim_final.py`, `sim_robust.py`, `sim_sdr.py`, `sim_local.py`, `sim_gauss.py`, `llfast.py`) are
  imported for shared helpers and contain code of earlier experiments that the paper does not use.

## Data

`data/hour.csv` and `data/Readme.txt` are the hourly records of the Bike Sharing Dataset from the UCI Machine Learning
Repository, distributed under CC BY 4.0: Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5W894. See also Fanaee-T and Gama (2014), Event labeling combining ensemble detectors and
background knowledge, Progress in Artificial Intelligence 2, 113-127.

## License

The code is released under the MIT License. The data keep their original license (CC BY 4.0).
