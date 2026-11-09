"""Results of the main experiment at n = 5000, shared by fig_main.py and make_tables_main.py.

load(root=None) -> {(model, d_Y, seed): {method: row}} from root/{gaussian,multimodal,manifold}/dY{2,5}/rank_*.jsonl
(default: multimodal and manifold from results/main_n5k_matched, the rerun of 11-01 in which the calibration, auxiliary
and evaluation HPD values use the same M proposal points, with CELL-RCP; Gaussian from results/main_n5k_gaussrej, the
rerun of 11-03 with the same settings that draws its responses by rejection, as the other two models do). A job counts
only if its .err ends with exit=0. Environment SEEDS=a-b keeps the seeds a, ..., b only.
"""
import glob, json, os

MODELS = (("gaussian", "Gaussian"), ("multimodal", "multimodal"), ("manifold", "manifold"))
DYS = (2, 5)
ROOT = "results/main_n5k_matched"
GAUSS_ROOT = "results/main_n5k_gaussrej"
SEEDS = os.environ.get("SEEDS")


def keep(s):
    if not SEEDS:
        return True
    a, b = map(int, SEEDS.split("-"))
    return a <= s <= b


def read(root, names=None):
    out = {}
    for name in names or [n for n, _ in MODELS]:
        for f in glob.glob(f"{root}/{name}/dY*/rank_*.jsonl"):
            e = f[:-6] + ".err"
            if not os.path.exists(e) or "# exit=0" not in open(e).read():
                continue
            rows = [json.loads(l) for l in open(f) if l.startswith("{")]
            if rows and keep(rows[0]["seed"]):
                out[(name, rows[0]["dY"], rows[0]["seed"])] = {r["method"]: r for r in rows}
    return out


def load(root=None):
    if root:
        return read(root)
    return {**read(ROOT, ["multimodal", "manifold"]), **read(GAUSS_ROOT, ["gaussian"])}


def cell(runs, name, d):
    """the runs of one model and d_Y, in seed order"""
    return [runs[k] for k in sorted(runs) if k[0] == name and k[1] == d]


cerr = lambda r: (r["cerrR"] + r["cerrO"]) / 2       # region-averaged conditional coverage error
