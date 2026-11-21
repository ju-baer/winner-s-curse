"""run_pilot.py -- CPU pilot (E1, E3, E4) for one independent ensemble.

Usage:  python run_pilot.py --seed 0 --M 36 [--smoke]
Writes  results/sweep_e{seed}.csv, results/planners_e{seed}.csv, results/models_e{seed}.pkl
"""
import argparse, os, pickle, time, warnings
import numpy as np
import wcp_core as W

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--M", type=int, default=36, help="number of start-goal pairs (half near, half far)")
ap.add_argument("--K", type=int, default=5)
ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

Ns_sweep = [16, 64, 256, 1024, 4096, 16384]
iters_sweep = [1, 3]
Ns_plan = [16, 64, 256, 1024, 4096]
if a.smoke:
    a.M, Ns_sweep, iters_sweep, Ns_plan = 4, [16, 256], [1, 3], [16, 256]

t0 = time.time()
mp = f"{a.out}/models_e{a.seed}.pkl"
if os.path.exists(mp):
    models, pos = pickle.load(open(mp, "rb"))
    print("loaded cached models", flush=True)
else:
    S, A, S2 = W.make_data(n_traj=1000, T=60, seed=a.seed)
    models = W.train_ensemble(S, A, S2, K=a.K, hidden=(64, 64), max_iter=40 if a.smoke else 150, seed=a.seed)
    pos = S[:, :2]
    pickle.dump((models, pos), open(mp, "wb"))
    print(f"trained ensemble seed {a.seed} in {time.time()-t0:.0f}s", flush=True)

pairs = W.make_pairs(M=a.M, seed=1000 + a.seed, data_pos=pos)   # different pairs per ensemble

print("E1/E3 sweep ...", flush=True)
df = W.run_sweep(models, pairs, Ns_sweep, iters_sweep, seed=a.seed)
df["ensemble"] = a.seed
df.to_csv(f"{a.out}/sweep_e{a.seed}.csv", index=False)

print("E4 planners ...", flush=True)
dp = W.run_planners(models, pairs, Ns_plan, iters=3, proposers=[0, 1, 2], seed=a.seed,
                    select_grid=Ns_plan)
dp["ensemble"] = a.seed
dp.to_csv(f"{a.out}/planners_e{a.seed}.csv", index=False)
print(f"done in {time.time()-t0:.0f}s", flush=True)
