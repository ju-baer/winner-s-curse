"""
wcp_core.py -- core code for the "Winner's Curse of Planning" pilot.

Self-contained (numpy + sklearn + scipy only). Everything the notebook needs:
  * a 2D point-mass environment with a swirl field, a high-drag zone and bouncing walls
  * offline data collected from only the left part of the arena (so the right side is OFF-SUPPORT)
  * a bootstrap ensemble of MLP dynamics models (trained with sklearn, run with fast numpy forward)
  * CEM planner and the planners compared in the paper plan (single, mean-ensemble, LCB,
    cross-fit re-rank, held-out budget selection)
  * the cross-fit optimism estimator
  * theory simulation (Gaussian baseline + two-group off-support extension)
  * analysis helpers (cluster bootstrap, Spearman, AUROC)

Honesty note: this is a controlled *pilot* environment. It is NOT evidence for the paper's claims
on real latent world models (DINO-WM / LeWM / TD-MPC2); it is the cheap first stage that decides
whether those expensive runs are worth doing.
"""
from __future__ import annotations
import zlib
import numpy as np
from scipy.stats import norm, spearmanr
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------- environment
DT = 0.1
H = 20            # planning horizon (env steps)
BLOCK = 4         # actions are piecewise-constant over BLOCK steps
NB = H // BLOCK   # number of action blocks per plan
GOAL_TOL = 0.10   # success: true final position within this distance of the goal


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def swirl(p):
    x, y = p[..., 0] - 0.5, p[..., 1] - 0.5
    g = np.exp(-(x * x + y * y) / 0.08)
    return np.stack([-y * g, x * g], -1) * 3.0


def drag(p):
    d2 = (p[..., 0] - 0.75) ** 2 + (p[..., 1] - 0.3) ** 2
    return 1.0 + 4.0 * _sig((0.15 ** 2 - d2) / 0.004)


def step(s, a):
    """True dynamics. s: (...,4) = (x, y, vx, vy); a: (...,2) in [-1,1]."""
    p, v = s[..., :2], s[..., 2:]
    a = np.clip(a, -1.0, 1.0)
    acc = 2.0 * a + swirl(p) - 0.8 * drag(p)[..., None] * v
    v2 = v + DT * acc
    p2 = p + DT * v2
    hit = (p2 < 0) | (p2 > 1)
    p2 = np.clip(p2, 0.0, 1.0)
    v2 = np.where(hit, -0.3 * v2, v2)
    return np.concatenate([p2, v2], -1).astype(np.float32)


def expand(Z):
    """(N, NB, 2) block actions -> (N, H, 2) per-step actions."""
    return np.repeat(Z, BLOCK, axis=1)


def cost_from_state(s, g):
    """Goal-reaching cost used by both the oracle and the models."""
    d = s[..., :2] - g
    return (d * d).sum(-1) + 0.05 * (s[..., 2:] ** 2).sum(-1)


def true_rollout(s0, A):
    s = np.repeat(np.asarray(s0, np.float32)[None], A.shape[0], 0)
    for t in range(A.shape[1]):
        s = step(s, A[:, t])
    return s


def oracle(s0, Z, g):
    """True cost and true goal distance of block-plans Z (N,NB,2) executed open-loop."""
    sf = true_rollout(s0, expand(Z))
    return cost_from_state(sf, g), np.linalg.norm(sf[:, :2] - g, axis=-1)


# ----------------------------------------------------------------------------- data
def make_data(n_traj=600, T=60, seed=0, x_max_start=0.5, x_cut=0.6):
    """Random-action rollouts from the left part of the arena. Transitions touching x > x_cut are
    dropped, so the right side (incl. the drag zone) is genuinely OFF-SUPPORT for the models."""
    rng = np.random.default_rng(seed)
    s = np.zeros((n_traj, 4), np.float32)
    s[:, 0] = rng.uniform(0.0, x_max_start, n_traj)
    s[:, 1] = rng.uniform(0.0, 1.0, n_traj)
    s[:, 2:] = rng.normal(0, 0.1, (n_traj, 2))
    S, A, S2 = [], [], []
    a = rng.uniform(-1, 1, (n_traj, 2)).astype(np.float32)
    for t in range(T):
        if t % 3 == 0:
            a = rng.uniform(-1, 1, (n_traj, 2)).astype(np.float32)
        s2 = step(s, a)
        S.append(s); A.append(a); S2.append(s2)
        s = s2
    S, A, S2 = np.concatenate(S), np.concatenate(A), np.concatenate(S2)
    if x_cut is not None:
        keep = (S[:, 0] < x_cut) & (S2[:, 0] < x_cut)
        S, A, S2 = S[keep], A[keep], S2[keep]
    return S, A, S2


# ----------------------------------------------------------------------------- models
class MLPDyn:
    """Fast numpy forward for a trained sklearn MLP predicting normalised state deltas."""

    def __init__(self, reg, xm, xs, ym, ys):
        self.W = [w.astype(np.float32) for w in reg.coefs_]
        self.b = [b.astype(np.float32) for b in reg.intercepts_]
        self.xm, self.xs, self.ym, self.ys = [np.asarray(a, np.float32) for a in (xm, xs, ym, ys)]

    def step(self, s, a):
        h = (np.concatenate([s, a], -1) - self.xm) / self.xs
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
        return s + h * self.ys + self.ym

    def final_state(self, s0, A):
        s = np.repeat(np.asarray(s0, np.float32)[None], A.shape[0], 0)
        for t in range(A.shape[1]):
            s = self.step(s, A[:, t])
        return s


def train_ensemble(S, A, S2, K=5, hidden=(64, 64), max_iter=120, seed=0, verbose=False):
    """Bootstrap ensemble of K MLP dynamics models."""
    from sklearn.neural_network import MLPRegressor
    X = np.concatenate([S, A], -1)
    Y = S2 - S
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    Xn, Yn = (X - xm) / xs, (Y - ym) / ys
    models = []
    rng = np.random.default_rng(seed)
    for k in range(K):
        idx = rng.integers(0, len(X), len(X))
        reg = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                           learning_rate_init=2e-3, batch_size=256, max_iter=max_iter,
                           random_state=seed * 100 + k, tol=1e-6, n_iter_no_change=max_iter)
        reg.fit(Xn[idx], Yn[idx])
        models.append(MLPDyn(reg, xm, xs, ym, ys))
        if verbose:
            print(f"member {k}: final train loss {reg.loss_:.4f}")
    return models


def one_step_rmse(models, S, A, S2, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.choice(len(S), min(n, len(S)), replace=False)
    return [float(np.sqrt(((m.step(S[i], A[i]) - S2[i]) ** 2).mean())) for m in models]


# ----------------------------------------------------------------------------- planners
def pred_cost(model, s0, Z, g):
    return cost_from_state(model.final_state(s0, expand(Z)), g)


def cem(costfn, N, iters, rng, elite_frac=0.1, std0=0.8):
    """Returns last population Z (N,NB,2) and its model costs. iters=1 is random shooting."""
    mu = np.zeros((NB, 2), np.float32)
    sd = np.full((NB, 2), std0, np.float32)
    ne = max(4, int(N * elite_frac))
    for it in range(iters):
        Z = np.clip(mu + sd * rng.standard_normal((N, NB, 2)).astype(np.float32), -1, 1)
        c = costfn(Z)
        if it < iters - 1:
            E = Z[np.argsort(c)[:ne]]
            mu, sd = E.mean(0), E.std(0) + 1e-3
    return Z, c


def evals_of(N, iters, n_models=1):
    """Model step-evaluations used by one CEM call (the compute unit used for matching)."""
    return N * iters * H * n_models


def plan(method, models, k, s0, g, N, iters, rng, lam=1.0, m_elite=None):
    """Return (best_plan Z* (NB,2), model-evals used). `k` is the proposer index."""
    K = len(models)
    others = [j for j in range(K) if j != k]
    if method == "single":                       # CEM on one model
        Z, c = cem(lambda Z: pred_cost(models[k], s0, Z, g), N, iters, rng)
        return Z[np.argmin(c)], evals_of(N, iters)
    if method == "meanens":                      # CEM on the ensemble-mean cost
        f = lambda Z: np.mean([pred_cost(m, s0, Z, g) for m in models], 0)
        Z, c = cem(f, N, iters, rng)
        return Z[np.argmin(c)], evals_of(N, iters, K)
    if method == "lcb":                          # pessimistic: mean + lam * std across members
        def f(Z):
            C = np.stack([pred_cost(m, s0, Z, g) for m in models])
            return C.mean(0) + lam * C.std(0)
        Z, c = cem(f, N, iters, rng)
        return Z[np.argmin(c)], evals_of(N, iters, K)
    if method == "crossfit":                     # propose with k, re-rank top-m with the others
        Z, c = cem(lambda Z: pred_cost(models[k], s0, Z, g), N, iters, rng)
        m = m_elite or max(8, N // 10)
        top = np.argsort(c)[:m]
        Ce = np.mean([pred_cost(models[j], s0, Z[top], g) for j in others], 0)
        return Z[top[np.argmin(Ce)]], evals_of(N, iters) + m * H * len(others)
    raise ValueError(method)


def plan_budget_select(models, k, s0, g, Ns, iters, rng):
    """Held-out budget selection: run proposer-k CEM at each N, keep the plan the OTHER members
    rate best. Cost = sum over the N grid."""
    others = [j for j in range(len(models)) if j != k]
    best, best_c, ev = None, np.inf, 0
    for N in Ns:
        Z, c = cem(lambda Z: pred_cost(models[k], s0, Z, g), N, iters, rng)
        z = Z[np.argmin(c)]
        ce = float(np.mean([pred_cost(m, s0, z[None], g)[0] for m in (models[j] for j in others)]))
        ev += evals_of(N, iters) + H * len(others)
        if ce < best_c:
            best, best_c = z, ce
    return best, ev


# ----------------------------------------------------------------------------- start/goal pairs
def make_pairs(M=40, seed=0, data_pos=None):
    """Half 'near' goals (in-support), half 'far' goals (off-support)."""
    rng = np.random.default_rng(seed)
    pairs = []
    tree = cKDTree(data_pos) if data_pos is not None else None
    for i in range(M):
        far = i >= M // 2
        while True:
            s0 = np.array([rng.uniform(0.05, 0.45), rng.uniform(0.1, 0.9), 0, 0], np.float32)
            gx = rng.uniform(0.65, 0.95) if far else rng.uniform(0.15, 0.5)
            g = np.array([gx, rng.uniform(0.1, 0.9)], np.float32)
            if np.linalg.norm(g - s0[:2]) >= 0.3:
                break
        sup = float(tree.query(g, k=10)[0].mean()) if tree is not None else np.nan
        pairs.append(dict(pair=i, tag="far" if far else "near", s0=s0, g=g, support_d=sup))
    return pairs


# ----------------------------------------------------------------------------- experiments
def _seed(*args):
    """Deterministic across processes (python's hash() of str is randomised)."""
    return zlib.crc32(repr(args).encode()) % (2 ** 32)


def run_sweep(models, pairs, Ns, iters_list, seed=0, proposers=None, progress=True):
    """E1/E3: for every pair, proposer k, N, iters: plan with member k, record predicted cost,
    oracle cost, the other members' rating (cross-fit) and their spread."""
    import pandas as pd, time
    K = len(models)
    proposers = list(range(K)) if proposers is None else proposers
    rows, t0 = [], time.time()
    for pr in pairs:
        s0, g = pr["s0"], pr["g"]
        for k in proposers:
            others = [j for j in range(K) if j != k]
            for iters in iters_list:
                for N in Ns:
                    rng = np.random.default_rng(_seed(seed, pr["pair"], k, iters, N))
                    Z, c = cem(lambda Z: pred_cost(models[k], s0, Z, g), N, iters, rng)
                    z = Z[np.argmin(c)][None]
                    Co = np.array([pred_cost(models[j], s0, z, g)[0] for j in others])
                    oc, od = oracle(s0, z, g)
                    rows.append(dict(pair=pr["pair"], tag=pr["tag"], support_d=pr["support_d"],
                                     proposer=k, N=N, iters=iters, evals=evals_of(N, iters),
                                     pred=float(c.min()), oracle=float(oc[0]), dist=float(od[0]),
                                     oth_mean=float(Co.mean()), oth_std=float(Co.std()),
                                     success=int(od[0] < GOAL_TOL)))
        if progress and pr["pair"] % 5 == 0:
            print(f"  pair {pr['pair']+1}/{len(pairs)}  elapsed {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["optimism"] = df["oracle"] - df["pred"]            # >0: model was too optimistic
    df["est_optimism"] = df["oth_mean"] - df["pred"]      # cross-fit estimate (no env needed)
    return df


def run_planners(models, pairs, Ns, iters=3, methods=("single", "meanens", "lcb", "crossfit"),
                 proposers=None, seed=0, select_grid=None, progress=True):
    """E4: compare planners; records oracle result and model-eval compute."""
    import pandas as pd, time
    K = len(models)
    proposers = list(range(K)) if proposers is None else proposers
    rows, t0 = [], time.time()
    for pr in pairs:
        s0, g = pr["s0"], pr["g"]
        for k in proposers:
            for meth in methods:
                for N in Ns:
                    rng = np.random.default_rng(_seed(seed, "p", pr["pair"], k, meth, N))
                    z, ev = plan(meth, models, k, s0, g, N, iters, rng)
                    oc, od = oracle(s0, z[None], g)
                    rows.append(dict(pair=pr["pair"], tag=pr["tag"], proposer=k, method=meth, N=N,
                                     evals=ev, oracle=float(oc[0]), dist=float(od[0]),
                                     success=int(od[0] < GOAL_TOL)))
            if select_grid is not None:
                rng = np.random.default_rng(_seed(seed, "sel", pr["pair"], k))
                z, ev = plan_budget_select(models, k, s0, g, select_grid, iters, rng)
                oc, od = oracle(s0, z[None], g)
                rows.append(dict(pair=pr["pair"], tag=pr["tag"], proposer=k, method="budget_select",
                                 N=int(max(select_grid)), evals=ev, oracle=float(oc[0]),
                                 dist=float(od[0]), success=int(od[0] < GOAL_TOL)))
        if progress and pr["pair"] % 5 == 0:
            print(f"  pair {pr['pair']+1}/{len(pairs)}  elapsed {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- theory simulation
def z_n(n):
    """Expected max of n standard normals (Blom approximation); 0 for n<1."""
    n = np.asarray(n, float)
    return np.where(n >= 1, norm.ppf((np.maximum(n, 1) - 0.375) / (np.maximum(n, 1) + 0.25)), 0.0)


def theory_gaussian(Ns, mu=0.0, s=1.0, sigma=0.5, reps=4000, seed=0):
    """Result 1: i.i.d. errors. Returns MC and closed-form optimism / true gain per N (cost minimised)."""
    rng = np.random.default_rng(seed)
    out = []
    for N in Ns:
        J = mu + s * rng.standard_normal((reps, N))
        Jh = J + sigma * rng.standard_normal((reps, N))
        i = Jh.argmin(1)
        r = np.arange(reps)
        out.append(dict(N=N, opt_mc=float((J[r, i] - Jh[r, i]).mean()),
                        gain_mc=float(mu - J[r, i].mean()),
                        opt_th=float(sigma ** 2 / np.sqrt(s ** 2 + sigma ** 2) * z_n(N)),
                        gain_th=float(s ** 2 / np.sqrt(s ** 2 + sigma ** 2) * z_n(N))))
    return out


def two_group_prediction(N, s, sig_lo, sig_hi, p_hi):
    """Soft group-competition prediction (no simulation of candidates).
    Each group g has n_g = N*p_g candidates whose predicted cost is ~ N(0, s^2+sig_g^2); the winner is
    the group with the lower minimum. Given the winning predicted value m from group g, the true cost
    is E[J | J_hat=m] = s^2/(s^2+sig_g^2) * m  (regression to the mean). Returns (expected true gain,
    P(off-support group wins))."""
    m = np.linspace(-14, 3, 6000)
    dm = m[1] - m[0]
    groups = {"lo": (sig_lo, N * (1 - p_hi)), "hi": (sig_hi, N * p_hi)}
    cdf, pdf, v = {}, {}, {}
    for g, (sg, n) in groups.items():
        v[g] = s ** 2 + sg ** 2
        Phi = norm.cdf(m / np.sqrt(v[g]))
        n = max(n, 1e-9)
        cdf[g] = 1.0 - (1.0 - Phi) ** n          # CDF of the minimum of n draws
        pdf[g] = np.gradient(cdf[g], dm)
    gain, p_hi_win = 0.0, 0.0
    for g, o in (("lo", "hi"), ("hi", "lo")):
        w = pdf[g] * (1.0 - cdf[o])              # density that g gives the overall minimum at m
        gain += float((w * (-(s ** 2 / v[g]) * m)).sum() * dm)
        if g == "hi":
            p_hi_win = float(w.sum() * dm)
    return gain, p_hi_win


def theory_two_group(Ns, s=1.0, sig_lo=0.3, sig_hi=3.0, p_hi=0.02, reps=3000, seed=0):
    """Result 2: a small fraction p_hi of candidates is off-support with large model error.
    The winning group switches from in-support to off-support as N grows -> true gain falls.
    Returns Monte-Carlo true gain, and a group-competition prediction of the same curve."""
    rng = np.random.default_rng(seed)
    out = []
    for N in Ns:
        gains, frac_hi = [], []
        for _ in range(reps):
            hi = rng.random(N) < p_hi
            sig = np.where(hi, sig_hi, sig_lo)
            J = s * rng.standard_normal(N)
            Jh = J + sig * rng.standard_normal(N)
            i = Jh.argmin()
            gains.append(-J[i]); frac_hi.append(float(hi[i]))
        gain_pred, p_hi_pred = two_group_prediction(N, s, sig_lo, sig_hi, p_hi)
        out.append(dict(N=N, gain_mc=float(np.mean(gains)), p_hi_wins_mc=float(np.mean(frac_hi)),
                        gain_pred=float(gain_pred), p_hi_wins_pred=float(p_hi_pred)))
    return out


# ----------------------------------------------------------------------------- analysis helpers
def cluster_boot(df, col, by="pair", n=2000, seed=0, groupby=None):
    """Mean and 95% CI of `col`, resampling pairs (the independent unit)."""
    rng = np.random.default_rng(seed)
    per = df.groupby(by)[col].mean().values
    if len(per) == 0:
        return np.nan, np.nan, np.nan
    bs = [rng.choice(per, len(per), replace=True).mean() for _ in range(n)]
    return float(per.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarize(df, keys, col):
    import pandas as pd
    rows = []
    for key, g in df.groupby(keys):
        m, lo, hi = cluster_boot(g, col)
        rows.append({**(dict(zip(keys, key)) if isinstance(key, tuple) else {keys[0]: key}),
                     col: m, "lo": lo, "hi": hi, "n_pairs": g["pair"].nunique()})
    return pd.DataFrame(rows)


def estimator_quality(df):
    """E3: does the cross-fit estimate track the true optimism, and does it predict failure?"""
    from sklearn.metrics import roc_auc_score
    out = {}
    rho, _ = spearmanr(df["est_optimism"], df["optimism"])
    out["spearman_all"] = float(rho)
    # within-N (removes the trivial 'both grow with N' trend)
    rs = [spearmanr(g["est_optimism"], g["optimism"])[0] for _, g in df.groupby(["N", "iters"]) if len(g) > 10]
    out["spearman_within_cell_mean"] = float(np.nanmean(rs))
    fail = 1 - df["success"].values
    if 0 < fail.sum() < len(fail):
        out["auroc_est_optimism"] = float(roc_auc_score(fail, df["est_optimism"]))
        out["auroc_oth_std"] = float(roc_auc_score(fail, df["oth_std"]))
        out["auroc_pred_cost_baseline"] = float(roc_auc_score(fail, df["pred"]))
        out["auroc_oth_mean"] = float(roc_auc_score(fail, df["oth_mean"]))
    return out
