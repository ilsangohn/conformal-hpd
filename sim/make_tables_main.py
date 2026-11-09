"""Tables of the main experiment (Section 8 and app:numerics): run from the project root,
python sim/make_tables_main.py [RESULT_ROOT].

  tables/main_paired.tex                     one-stage conditional coverage error and the paired difference
                                             two-stage - one-stage, per model and d_Y (main text)
  tables/main_{gaussian,multimodal,manifold}.tex   all methods, blocks d_Y = 2 and 5 stacked, with the volume ratio
                                             sum Leb(C_hat)/sum Leb(C*) over the random covariates (appendix)
  tables/main_samefit.tex                    one-stage on all fitting points, one-stage and two-stage with the same
                                             fit on the base sample (appendix)
Data: main_data.load (results/main_n5k_matched and, for the Gaussian model, results/main_n5k_gaussrej; jobs with exit=0, SEEDS=a-b). Conditional coverage error
= region-averaged, times 100; shape loss and volume relative to Leb(C*), averaged over the evaluation covariates.
Two-stage = rank calibration with the isotropic frozen-target CV bandwidth (-RANKISO). Mean +- standard error over the
replications; in main_paired a difference is set in bold when it lies more than two standard errors from zero (the
verdict rule registered in results_main_n5k/README.md).
"""
import sys
import numpy as np
from main_data import MODELS, DYS, load, cerr, cell

runs = load(sys.argv[1] if len(sys.argv) > 1 else None)
EST = (("NN", "HPD-NN"), ("LP", "HPD-LP"), ("partition", "CELL"))
ROWS = [("\\textit{true density}", "\\textit{one-stage}", "ORACLE-COND"),
        ("", "\\textit{global threshold}", "ORACLE-MARG"),
        None,
        ("NN", "one-stage", "HPD-NN"), ("", "\\cref{alg:local}", "HPD-NN-RANKISO"),
        ("", "two-stage (own covariate)", "HPD-NN-LOC"), ("", "two-stage (quantile)", "HPD-NN-RCP"),
        None,
        ("LP", "one-stage", "HPD-LP"), ("", "\\cref{alg:local}", "HPD-LP-RANKISO"),
        ("", "two-stage (own covariate)", "HPD-LP-LOC"), ("", "two-stage (quantile)", "HPD-LP-RCP"),
        None,
        ("partition", "one-stage", "CELL"), ("", "\\cref{alg:local}", "CELL-RANKISO"),
        ("", "two-stage (own covariate)", "CELL-LOC"), ("", "two-stage (quantile)", "CELL-RCP"),
        None,
        ("baselines", "CHCDS", "CHCDS"), ("", "local ellipsoid", "ELLIPSOID"), ("", "RCP (ellipsoid score)", "RCP-ELLIPSOID"),
        ("", "PCP", "PCP"), ("", "RCP (PCP score)", "RCP-PCP"), ("", "OT-CP", "OT-CP")]
WORDS = {10: "ten", 20: "twenty"}


def num(v, sig=3):
    """sig significant digits; powers of ten from 10^4 on and below 10^-3"""
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
        e = int(np.floor(np.log10(abs(v)))); return f"{v / 10 ** e:.{sig - 1}f}\\times10^{{{e}}}"
    if v == 0 or abs(v) >= 10 ** (sig - 1):
        return f"{v:.0f}"
    return f"{v:.{max(0, sig - 1 - int(np.floor(np.log10(abs(v)))))}f}"


def pm(vals, fmt=num, bold=False):
    v = np.asarray(vals, float); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    s = f"${fmt(m)}${{\\scriptsize${{}}\\pm{fmt(se) if fmt is not num else num(se, 2)}$}}"
    return f"{{\\boldmath{s}}}" if bold and abs(m) > 2 * se else s


f2 = lambda v: f"{v:.2f}"
COUNTS = {(disp, d): len(cell(runs, n, d)) for n, disp in MODELS for d in DYS}


def reps(keys, model=True):
    """'twenty replications', with the counts of the cells that have fewer (runs still missing) in parentheses"""
    c = {k: COUNTS[k] for k in keys}; full = max(c.values())
    short = [f"{v} for {f'the {k[0]} model with ' if model else ''}$d_Y={k[1]}$" for k, v in c.items() if 0 < v < full]
    return f"{WORDS.get(full, str(full))} replications" + (f" ({', '.join(short)})" if short else "")


nrep = reps(list(COUNTS))

# ---------------- main text: paired differences
L = ["\\begin{table}[!t]",
     f"\\caption{{Region-averaged conditional coverage error ($\\times100$) of one-stage calibration and the paired "
     f"difference \\cref{{alg:local}} $-$ one-stage, mean $\\pm$ standard error over {nrep}. Bold: "
     f"more than two standard errors from zero.}}\\label{{tab:paired}}",
     "\\centering\\small", "\\begin{tabular}{llcccccc}", "\\toprule",
     " & & \\multicolumn{2}{c}{NN} & \\multicolumn{2}{c}{LP} & \\multicolumn{2}{c}{partition}\\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
     "model & $d_Y$ & one-stage & difference & one-stage & difference & one-stage & difference\\\\", "\\midrule"]
for i, (name, disp) in enumerate(MODELS):
    for j, d in enumerate(DYS):
        R = cell(runs, name, d)
        if not R:
            continue
        line = f"{disp if j == 0 else ''} & ${d}$"
        for _, pre in EST:
            line += f" & {pm([100 * cerr(r[pre]) for r in R], f2)} & " \
                    f"{pm([100 * (cerr(r[pre + '-RANKISO']) - cerr(r[pre])) for r in R], f2, bold=True)}"
        L.append(line + "\\\\")
    if i < len(MODELS) - 1:
        L.append("\\addlinespace[2pt]")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open("tables/main_paired.tex", "w").write("\n".join(L) + "\n")

# ---------------- appendix: all methods, one table per model with the blocks d_Y = 2 and d_Y = 5 stacked
for name, disp in MODELS:
    C = {d: cell(runs, name, d) for d in DYS}
    L = ["\\begin{table}[!t]",
         f"\\caption{{{disp[0].upper() + disp[1:]} model: marginal coverage, region-averaged conditional coverage error "
         f"($\\times100$), shape loss $\\cS^\\circ$ relative to $\\Leb(\\Cstar)$, averaged over the evaluation covariates, "
         f"and the volume ratio $\\sum_j\\Leb(\\hC(X_j))/\\sum_j\\Leb(\\Cstar(X_j))$ over the random covariates $X_j$ of "
         f"the marginal coverage; mean $\\pm$ standard error over {reps([(disp, d) for d in DYS], model=False)}.}}"
         f"\\label{{tab:main-{name}}}",
         "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
         "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%", "\\begin{tabular}{llcccc}", "\\toprule",
         "\\multicolumn{2}{l}{method} & marg.\\ coverage & cov.\\ error & shape loss & volume ratio\\\\"]
    for d in DYS:
        if not C[d]:
            continue
        L += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{$d_Y={d}$}}\\\\", "\\addlinespace[2pt]"]
        for row in ROWS:
            if row is None:
                if not L[-1].startswith("\\addlinespace"):
                    L.append("\\addlinespace[2pt]")
                continue
            a_, b_, m = row
            if not all(m in r for r in C[d]):
                continue
            L.append(f"{a_} & {b_} & {pm([r[m]['marg'] for r in C[d]], lambda v: f'{v:.3f}')} & "
                     f"{pm([100 * cerr(r[m]) for r in C[d]], f2)} & {pm([r[m]['shape'] for r in C[d]])} & "
                     f"{pm([r[m]['volM'] for r in C[d]])}\\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open(f"tables/main_{name}.tex", "w").write("\n".join(L) + "\n")

# ---------------- appendix: same fit
L = ["\\begin{table}[!t]",
     f"\\caption{{The same density fit with one-stage calibration and with \\cref{{alg:local}}: region-averaged "
     f"conditional coverage error ($\\times100$) and shape loss $\\cS^\\circ$ relative to $\\Leb(\\Cstar)$ of one-stage "
     f"calibration with the density fitted on all $4000$ fitting points (all), and of one-stage calibration (one) "
     f"and \\cref{{alg:local}} (two) with the density fitted on the $3000$ points of two-stage calibration; mean "
     f"$\\pm$ standard error over {nrep}.}}\\label{{tab:samefit}}",
     "\\centering\\small", "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%",
     "\\begin{tabular}{lllcccccc}", "\\toprule",
     " & & & \\multicolumn{3}{c}{cond.\\ cov.\\ error ($\\times100$)} & \\multicolumn{3}{c}{shape loss $\\cS^\\circ$}\\\\",
     "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
     "model & $d_Y$ & estimate & all & one & two & all & one & two\\\\", "\\midrule"]
for i, (name, disp) in enumerate(MODELS):
    for j, d in enumerate(DYS):
        R = cell(runs, name, d)
        if not R:
            continue
        for k, (lab, pre) in enumerate(EST):
            ms = (pre, pre + "-A", pre + "-RANKISO")
            L.append(f"{disp if j == 0 and k == 0 else ''} & {f'${d}$' if k == 0 else ''} & {lab} & "
                     + " & ".join(pm([100 * cerr(r[m]) for r in R], f2) for m in ms) + " & "
                     + " & ".join(pm([r[m]["shape"] for r in R]) for m in ms) + "\\\\")
        if (i, j) != (len(MODELS) - 1, len(DYS) - 1):
            L.append("\\addlinespace[2pt]")
L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
open("tables/main_samefit.tex", "w").write("\n".join(L) + "\n")
print("replications per cell:", {f"{n}{d}": len(cell(runs, n, d)) for n, _ in MODELS for d in DYS})
