"""Bike-sharing table with rank calibration (layout of tab:bike), mean +- s.e. over splits, and a JSON summary.

usage: python make_table_bike_rank.py RESULTS_DIR OUT_TEX OUT_JSON [EXT_DIR]
  RESULTS_DIR  rank_bike_<seed>.jsonl / .err of realdata_rank.py (all methods; the final runs use the default
               TUNE_GRID=ext, i.e. extended LP and KDE grids)
  EXT_DIR      optional: runs restricted to ONLY=lp,kde,cell whose rows replace the LP, KDE and partition rows of
               RESULTS_DIR (the other rows do not depend on these grids); not used for the final table
Environment CELL_RCP=DIR (optional): partition-only runs (ONLY=cell) of the same splits whose row CELL-RCP is added;
their partition rows must reproduce those of RESULTS_DIR (the largest difference is printed).
Rows: NN and LP: one-stage (HPD-x), Algorithm 1 (-RANKISO: isotropic frozen-target CV, as in the simulations), two-stage
(own covariate) (-LOC), two-stage (quantile) (-RCP); partition: one-stage (CELL), Algorithm 1 (CELL-RANKISO), two-stage
(own covariate) (CELL-LOC), two-stage (quantile) (CELL-RCP, with CELL_RCP); baselines: CHCDS,
ELLIPSOID, RCP-ELLIPSOID, PCP, RCP-PCP, OT-CP. The JSON summary has all rows (also -RANKFIX and -A),
per-split values, the selected tuning parameters with grid-edge flags, and per-split wall time and peak memory.
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np

res_dir, out_tex, out_json = sys.argv[1:4]
ext_dir = sys.argv[4] if len(sys.argv) > 4 else None


def load(d):
    rows, info, cost = defaultdict(dict), {}, {}
    for f in sorted(glob.glob(os.path.join(d, "rank_bike_*.jsonl"))):
        for l in open(f):
            if l.startswith("{"):
                r = json.loads(l)
                if r["method"] == "INFO":
                    info[r["seed"]] = r
                else:
                    rows[r["method"]][r["seed"]] = r
        e = f[:-6] + ".err"
        if os.path.exists(e):
            t = open(e).read(); c = {}
            m = re.search(r"([\d.]+) real", t); c["wall_s"] = float(m.group(1)) if m else None
            m = re.search(r"(\d+)\s+maximum resident set size", t); c["max_rss_gb"] = int(m.group(1)) / 1e9 if m else None
            m = re.search(r"(\d+)\s+peak memory footprint", t); c["peak_footprint_gb"] = int(m.group(1)) / 1e9 if m else None
            cost[int(re.search(r"rank_bike_(\d+)\.jsonl", f).group(1))] = c
    return rows, info, cost


rows, info, cost = load(res_dir)
EXT_ROWS = ("HPD-LP", "HPD-KDE", "CELL")
if ext_dir:
    rows_e, info_e, cost_e = load(ext_dir)
    for m in list(rows):
        if m.startswith(EXT_ROWS):
            del rows[m]
    rows.update(rows_e)
seeds = sorted(info)
if os.environ.get("CELL_RCP"):
    rows_c = load(os.environ["CELL_RCP"])[0]
    diff = max(abs(rows_c[m][s_][k] - rows[m][s_][k]) for m in ("CELL", "CELL-RANKISO", "CELL-LOC") for s_ in seeds
               for k in ("marg", "cerr_mean", "cerr_max", "area"))
    print(f"partition rows of CELL_RCP vs RESULTS_DIR: largest difference {diff:.1e}")
    rows["CELL-RCP"] = rows_c["CELL-RCP"]
assert all(set(rows[m]) == set(seeds) for m in rows), {m: sorted(rows[m]) for m in rows}

TWO = r"\cref{alg:local}"
GROUPS = [("NN", [("HPD-NN", "one-stage"), ("HPD-NN-RANKISO", TWO), ("HPD-NN-LOC", "two-stage (own covariate)"),
                  ("HPD-NN-RCP", "two-stage (quantile)")]),
          ("LP", [("HPD-LP", "one-stage"), ("HPD-LP-RANKISO", TWO), ("HPD-LP-LOC", "two-stage (own covariate)"),
                  ("HPD-LP-RCP", "two-stage (quantile)")]),
          ("partition", [("CELL", "one-stage"), ("CELL-RANKISO", TWO), ("CELL-LOC", "two-stage (own covariate)")]
                         + ([("CELL-RCP", "two-stage (quantile)")] if os.environ.get("CELL_RCP") else [])),
          ("baselines", [("CHCDS", "CHCDS"), ("ELLIPSOID", "local ellipsoid"), ("RCP-ELLIPSOID", "RCP (ellipsoid score)"), ("PCP", "PCP"),
                          ("RCP-PCP", "RCP (PCP score)"), ("OT-CP", "OT-CP")])]
se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def stats(m):
    d = rows[m]; out = {"n": len(d)}
    for k in ("marg", "cerr_mean", "cerr_max", "area"):
        v = np.array([d[s][k] for s in seeds]); out[k] = {"mean": float(v.mean()), "se": se(v), "per_split": v.tolist()}
    rel = np.array([d[s]["area"] / rows["HPD-NN"][s]["area"] for s in seeds])
    out["area_rel_nn"] = {"mean": float(rel.mean()), "se": se(rel), "per_split": rel.tolist()}
    for k in ("hx", "k", "lp_hx", "lp_hy", "m", "hy", "kde_hx", "kde_hy", "nn_epochs", "cv_edge", "k_edge", "time_s", "tau"):
        if k in d[seeds[0]]:
            out[k] = [d[s].get(k) for s in seeds]
    return out


def pm(st, k, scale=1.0):
    return rf"${st[k]['mean'] * scale:.3f}${{\scriptsize${{}}\pm{st[k]['se'] * scale:.3f}$}}"


S = {m: stats(m) for m in rows}
lines = []
for gi, (gname, lst) in enumerate(GROUPS):
    for j, (m, lab) in enumerate(lst):
        st = S[m]
        lines.append(f"{gname if j == 0 else ''} & {lab} & {pm(st, 'marg')} & {pm(st, 'cerr_mean', 10)} & "
                     f"{pm(st, 'cerr_max', 10)} & {pm(st, 'area')} & {pm(st, 'area_rel_nn')}\\\\")
    if gi < len(GROUPS) - 1:
        lines.append(r"\addlinespace[2pt]")
floor = float(np.mean([info[s]["sampling_floor"] for s in seeds]))
n_test = info[seeds[0]]["n_test"]
ext_lp = ext_dir is not None or any(info[s_].get("tune_grid", "sim") == "ext" for s_ in seeds)
grid_note = ("" if not ext_lp else
             " The bandwidths of the LP and of \\cref{alg:local} are chosen from grids extended below those of the simulations.")
tex = r"""\begin{table}[!htbp]
\caption{Bike-sharing demand: mean $\pm$ standard error over """ + f"{len(seeds)}" + r""" random splits; the sampling floor of the mean
group-coverage deviation is about $""" + f"{10 * floor:.2f}" + r"""$ on the $\times10$ scale. Columns marked ($\times10$) report ten times the value. Areas are in units of log-counts squared.""" + grid_note + r"""}\label{tab:bike}
\centering\small
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{llccccc}
\toprule
\multicolumn{2}{l}{method} & marginal coverage & \multicolumn{2}{c}{$|$group coverage$-0.9|$ ($\times10$)} & area & area relative\\
 & & & mean & max & & to NN, one-stage\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(out_tex, "w").write(tex)


# ---------------- tuning summary with grid-edge flags
def tuning(info):
    T = defaultdict(list)
    for s in seeds:
        I = info.get(s)
        if I is None:
            continue
        for tag in ("fit_full", "fit_A"):
            if tag in I:
                f = I[tag]
                if f.get("nn_epochs"):
                    T[f"nn_epochs_{tag}"].append((f["nn_epochs"], f["nn_edge"]))
                if f.get("lp_hx") is not None:
                    T[f"lp_{tag}"].append(((f["lp_hx"], f["lp_hy"]), f["lp_edge"]))
        if "kde" in I:
            T["kde"].append(((I["kde"]["kde_hx"], I["kde"]["kde_hy"]), I["kde"]["kde_edge"]))
        for tag in ("cell_full", "cell_A"):
            if tag in I:
                T[tag].append(((I[tag]["m"], I[tag]["hy"]), I[tag]["cell_edge"]))
        for m in ("HPD-NN-RANK", "HPD-LP-RANK", "CELL-RANK", "HPD-NN-LOC", "HPD-LP-LOC", "CELL-LOC"):
            if m in I:
                T[m + " h"].append((I[m]["hx"], I[m]["cv_edge"]))
        for m in ("HPD-NN-RANKFCV", "HPD-LP-RANKFCV", "CELL-RANKFCV", "HPD-NN-RANKISO", "HPD-LP-RANKISO", "CELL-RANKISO"):
            if m in I:
                T[m + " h"].append((I[m]["hx"], I[m]["fcv_edge"]))
        for m in ("HPD-NN-RCP", "HPD-LP-RCP", "RCP-ELLIPSOID", "RCP-PCP"):
            if m in I:
                T[m + " k"].append((I[m]["k"], I[m]["k_edge"]))
    return dict(T)


summary = {"seeds": seeds, "results_dir": os.path.abspath(res_dir), "ext_dir": os.path.abspath(ext_dir) if ext_dir else None,
           "sizes": {k: info[seeds[0]][k] for k in ("n_A", "n_reg", "n_val", "n_cal", "n_test", "grid")},
           "md5_sim_rank": sorted({info[s]["md5_sim_rank"] for s in seeds}),
           "sampling_floor_mean": floor, "group_sizes_split0": info[seeds[0]]["group_sizes"],
           "methods": S, "tuning_sim_grid": tuning(info), "cost_sim_grid": cost,
           "h_fixed": info[seeds[0]].get("h_fixed")}
if ext_dir:
    summary["tuning_ext_grid"] = tuning(info_e); summary["cost_ext_grid"] = cost_e
    summary["ext_rows"] = sorted(m for m in rows if m.startswith(EXT_ROWS))
json.dump(summary, open(out_json, "w"), indent=1, default=str)
print("\n".join(lines))
print(f"sampling floor (x10): {10 * floor:.3f}; n_test {n_test}; splits {seeds}")
diag = ["HPD-NN-RANK", "HPD-LP-RANK", "CELL-RANK", "HPD-NN-RANKFIX", "HPD-LP-RANKFIX", "CELL-RANKFIX", "HPD-NN-A", "HPD-LP-A", "CELL-A"]
for b in ("HPD-NN", "HPD-LP", "CELL"):
    diag += [b + "-RANKFCV"] + sorted((m for m in S if m.startswith(b + "-RANKH")), key=lambda m: float(m.split("RANKH")[1]))
for m in diag:
    if m in S:
        st = S[m]
        print(f"{m:16s} marg {st['marg']['mean']:.3f} cerr {10 * st['cerr_mean']['mean']:.3f}+-{10 * st['cerr_mean']['se']:.3f} "
              f"max {10 * st['cerr_max']['mean']:.3f} area {st['area']['mean']:.3f}+-{st['area']['se']:.3f}")
