import json

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": s.strip("\n").splitlines(True)}

cells = []

cells.append(md("""
# The Winner's Curse of Planning: pilot experiments (ICWM 2027)

**What this notebook is.** The cheap first stage of the paper plan. It tests, in a controlled environment where the
true dynamics are known, whether (1) a planner's predicted cost gets more optimistic as search grows,
(2) true success saturates or reverses when some candidates are off the model's data support,
(3) a label-free *cross-fit* estimator can measure that optimism, and (4) bias-aware planners help at matched compute.

**What it is not.** Evidence about DINO-WM / LeWM / TD-MPC2. Those released-checkpoint runs (stage 2) only start if the
gates in the last section pass here.

**Where things run**

| Part | Where | Cost |
|---|---|---|
| A. Theory simulation | local CPU | free |
| B. Learned-model pilot (E1, E3, E4) | local CPU (about 25 min on 2 cores) | free |
| C. Same pilot, scaled up | Modal GPU (`modal_app.py`) | start with `--smoke`, then check the printed cost |
| D. Gates | local | free |

Related work you must cite and position against (found in the literature check): *Search or Scoring? Diagnosing Why
Visual World-Model Planners Fail* (arXiv 2611.25347) already states the i.i.d. optimism result
sigma*sqrt(2 ln N) and the optimizer's-curse framing. This pilot therefore treats Result 1 as background and tests what
is left: the off-support extension (Result 2), the label-free estimator, and budget-aware planning.
"""))

cells.append(code("""
import os, sys, subprocess, importlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
import wcp_core as W, wcp_plots as P
importlib.reload(W); importlib.reload(P)
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)

def run(cmd):
    # Run a command and stream its output INTO this cell. (Plain subprocess.run sends output to the
    # Jupyter server log, which is why Modal runs can look like they did nothing.)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print(f"\\n[exit code {p.returncode}]")
    return p.returncode

print("ready | working directory:", os.getcwd())
assert os.path.exists("modal_app.py"), "Open the notebook from inside the icwm_pilot folder (modal_app.py must be next to it)."
"""))

cells.append(md("## A. Theory simulation (Results 1 and 2)\nResult 1 is the standard i.i.d. winner's-curse formula. Result 2 is the off-support extension: a small share of candidates has large model error, so the *winning group* switches as N grows."))
cells.append(code("""
Ns_t = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384]
gauss = W.theory_gaussian([4, 16, 64, 256, 1024, 4096], sigma=0.5, reps=3000)
two = W.theory_two_group(Ns_t, s=1.0, sig_lo=0.3, sig_hi=3.0, p_hi=0.02, reps=2000)
fig = P.fig_theory(gauss, two); fig.savefig("figures/fig_theory.png", dpi=160, bbox_inches="tight"); plt.show()
g, q = [r["gain_mc"] for r in two], [r["gain_pred"] for r in two]
print("simulated peak of true gain at N* =", Ns_t[int(np.argmax(g))], "| predicted N* =", Ns_t[int(np.argmax(q))],
      "| corr =", round(float(np.corrcoef(g, q)[0, 1]), 3))
"""))

cells.append(md("""
## B. Learned-model pilot on CPU
Environment: 2D point mass with a swirl field, a high-drag disc and bouncing walls. Offline data comes only from
x < 0.6, so the right side (including the drag disc) is **off-support**. Two independent ensembles of 5 MLP dynamics
models; 36 start-goal pairs each (18 *near* goals in-support, 18 *far* goals off-support). Every model also acts as a
proposer while the other four act as held-out evaluators (this is the cross-fit estimator).
"""))
cells.append(code("""
# ~10-12 min per ensemble; the two ensembles run in parallel. Skipped if results already exist.
need = [s for s in (0, 1) if not os.path.exists(f"results/planners_e{s}.csv")]
procs = [subprocess.Popen([sys.executable, "run_pilot.py", "--seed", str(s), "--M", "36"]) for s in need]
for p_ in procs: p_.wait()
df = pd.concat([pd.read_csv(f"results/sweep_e{s}.csv") for s in (0, 1)], ignore_index=True)
dp = pd.concat([pd.read_csv(f"results/planners_e{s}.csv") for s in (0, 1)], ignore_index=True)
# pairs differ per ensemble: make pair ids unique so the cluster bootstrap treats them as separate units
for d_ in (df, dp): d_["pair"] = d_["ensemble"] * 1000 + d_["pair"]
print(len(df), "sweep rows;", len(dp), "planner rows")
"""))

cells.append(md("### E1: optimism and success versus search budget"))
cells.append(code("""
fig = P.fig_scaling(df, iters_list=(1, 3)); fig.savefig("figures/fig_E1_scaling.png", dpi=160, bbox_inches="tight"); plt.show()
print(W.summarize(df[df.iters == 3], ["tag", "N"], "optimism").round(4).to_string(index=False))
"""))

cells.append(md("### E3: can optimism be estimated without the environment?"))
cells.append(code("""
fig, q = P.fig_estimator(df); fig.savefig("figures/fig_E3_estimator.png", dpi=160, bbox_inches="tight"); plt.show()
print({k: round(v, 3) for k, v in q.items()})
"""))

cells.append(md("### E4: planners at matched compute\n`single` = plain CEM; `meanens`/`lcb` = ensemble planners (5x model evaluations); `crossfit` = propose with one member, re-rank top candidates with the others; `budget_select` = run several N and keep the plan the held-out members rate best."))
cells.append(code("""
fig = P.fig_planners(dp); fig.savefig("figures/fig_E4_planners.png", dpi=160, bbox_inches="tight"); plt.show()
print(P.planner_table(dp).round(3).to_string(index=False))
"""))

cells.append(md("""
## C. Scale up on Modal GPUs from this notebook

One-time setup (in the VS Code terminal, not in this notebook): `pip install modal`, then `modal setup` and sign in.
Then run the cells below **in order**. Always do the smoke run first and read the cost line it prints. Output now streams into the cell; if a cell prints nothing, something is wrong with the kernel, not Modal.
"""))
cells.append(code("""
# 1) Is Modal installed, and which workspace are you logged into? (Compare the workspace name with the
#    one selected at the top-left of modal.com/apps. Runs only show up in the workspace you are logged into.)
run([sys.executable, "-m", "modal", "--version"])
run([sys.executable, "-m", "modal", "profile", "current"])
"""))
cells.append(code("""
# 2) SMOKE RUN: one tiny ensemble on one L4 GPU. Output streams below. Expect: image build (first time
#    takes a few minutes), then 'trained in ...s', then a cost line, then 'saved results/gpu_*_smoke.csv'.
rc = run([sys.executable, "-m", "modal", "run", "modal_app.py", "--smoke"])
"""))
cells.append(code("""
# 3) FULL RUN (3 ensembles in parallel on L4s). Set RUN_FULL=True only after the smoke run worked.
RUN_FULL = False
if RUN_FULL:
    run([sys.executable, "-m", "modal", "run", "modal_app.py", "--seeds", "3", "--M", "60"])
"""))
cells.append(code("""
# 4) Where did the results go? Local CSVs, the Modal volume, and Modal's own app list.
print("local result files:", sorted(f for f in os.listdir("results") if f.startswith("gpu_")))
run([sys.executable, "-m", "modal", "volume", "ls", "wcp-results"])
run([sys.executable, "-m", "modal", "app", "list"])
"""))
cells.append(code("""
# 5) Load GPU results (full run if present, otherwise the smoke run) and reuse the same figures.
tag = "full" if os.path.exists("results/gpu_sweep_full.csv") else ("smoke" if os.path.exists("results/gpu_sweep_smoke.csv") else None)
if tag is None:
    print("No GPU result CSVs in ./results yet. Run cell 2 and read its output for errors.")
else:
    dg, dgp = pd.read_csv(f"results/gpu_sweep_{tag}.csv"), pd.read_csv(f"results/gpu_planners_{tag}.csv")
    for d_ in (dg, dgp): d_["pair"] = d_["ensemble"] * 1000 + d_["pair"]
    print(f"loaded {tag} run:", len(dg), "sweep rows,", len(dgp), "planner rows")
    print(dg.groupby(["tag", "N"])[["pred", "oracle", "optimism", "success"]].mean().round(3))
    if tag == "full":
        P.fig_scaling(dg, iters_list=(1, 3, 6)).savefig("figures/fig_E1_scaling_gpu.png", dpi=160, bbox_inches="tight")
        P.fig_estimator(dg)[0].savefig("figures/fig_E3_estimator_gpu.png", dpi=160, bbox_inches="tight")
        P.fig_planners(dgp).savefig("figures/fig_E4_planners_gpu.png", dpi=160, bbox_inches="tight")
        plt.show()
"""))

cells.append(md("""
## D. Decision gates (pre-registered in the plan)
Computed automatically with cluster-bootstrap 95% CIs over start-goal pairs. Do not change the thresholds after looking.

* **G1** optimism at the largest N exceeds optimism at the smallest N (CI above 0).
* **G2** success at the largest N versus a mid-range N: reports REVERSAL / SATURATED / STILL IMPROVING.
* **G3** cross-fit estimate tracks true optimism (Spearman >= 0.3 within each budget cell), and flags failures better than the predicted cost alone (AUROC >= 0.7).
* **G4** a bias-aware planner beats single-model CEM on far goals (CI above 0), with its compute multiple reported.
"""))
cells.append(code("""
G = P.gates(df, dp)
print(G.round(3).to_string(index=False))
"""))

cells.append(md("""
## Stage 2 (only if the gates pass): released checkpoints
Not implemented here on purpose: it depends on external repos whose current APIs I could not verify.
Plan: (a) E0 reproduce a published Push-T success number with the released DINO-WM / JEPA-WM checkpoint before any new claim;
(b) rerun E1 and E3 with a reduced N sweep (planning is slow: one public report puts DINO-WM at roughly 5,800 s for 50
trajectories on an L40S); (c) TD-MPC2 released checkpoints with an MPPI sample sweep. Use `modal_app.py` as the template:
swap the MLP ensemble for the released model and keep `W.oracle`-style simulator scoring.
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("winners_curse_pilot.ipynb", "w"), indent=1)
print("wrote winners_curse_pilot.ipynb with", len(cells), "cells")
