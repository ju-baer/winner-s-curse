"""Figures + summary tables for the pilot. Works on CPU-pilot and Modal-GPU result DataFrames alike."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wcp_core as W

C = {"near": "#0072B2", "far": "#D55E00", "all": "#444444"}                 # colour-blind-safe
MC = {"single": "#444444", "meanens": "#009E73", "lcb": "#CC79A7", "crossfit": "#0072B2",
      "budget_select": "#D55E00"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def _band(ax, d, col, color, label, ls="-"):
    s = W.summarize(d, ["N"], col).sort_values("N")
    ax.plot(s["N"], s[col], ls, color=color, label=label, lw=1.8, marker="o", ms=3)
    ax.fill_between(s["N"], s["lo"], s["hi"], color=color, alpha=0.15, lw=0)


def fig_scaling(df, iters_list=(1, 3)):
    """E1: optimism and success vs search budget N, near (in-support) vs far (off-support) goals."""
    fig, ax = plt.subplots(2, len(iters_list), figsize=(4.6 * len(iters_list), 6.2), sharex=True)
    ax = np.atleast_2d(ax).reshape(2, -1)
    for j, it in enumerate(iters_list):
        d = df[df.iters == it]
        for tag in ("near", "far"):
            _band(ax[0, j], d[d.tag == tag], "optimism", C[tag], f"{tag} goals")
            _band(ax[1, j], d[d.tag == tag], "success", C[tag], f"{tag} goals")
        ax[0, j].axhline(0, color="k", lw=0.6)
        ax[0, j].set_title(f"CEM iterations = {it}")
        ax[0, j].set_ylabel("optimism = true cost - predicted cost")
        ax[1, j].set_ylabel("true success rate")
        ax[1, j].set_xlabel("search budget N (candidates per iteration)")
        for a in ax[:, j]:
            a.set_xscale("log")
    ax[0, 0].legend(frameon=False)
    fig.suptitle("E1: does more search buy more optimism, and does success follow?", y=1.0)
    fig.tight_layout()
    return fig


def fig_theory(gauss, two):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    N = [r["N"] for r in gauss]
    ax[0].plot(N, [r["opt_mc"] for r in gauss], "o", color=C["near"], label="simulation")
    ax[0].plot(N, [r["opt_th"] for r in gauss], "-", color="k", label="closed form")
    ax[0].set_xscale("log"); ax[0].set_title("Result 1: optimism, i.i.d. errors")
    ax[0].set_xlabel("N"); ax[0].set_ylabel("optimism"); ax[0].legend(frameon=False)
    N2 = [r["N"] for r in two]
    ax[1].plot(N2, [r["gain_mc"] for r in two], "o", color=C["far"], label="simulation")
    ax[1].plot(N2, [r["gain_pred"] for r in two], "-", color="k", label="group-competition prediction")
    ax[1].set_xscale("log"); ax[1].set_title("Result 2: true gain, 2% off-support")
    ax[1].set_xlabel("N"); ax[1].set_ylabel("true gain of selected plan"); ax[1].legend(frameon=False)
    ax[2].plot(N2, [r["p_hi_wins_mc"] for r in two], "o", color=C["far"], label="simulation")
    ax[2].plot(N2, [r["p_hi_wins_pred"] for r in two], "-", color="k", label="prediction")
    ax[2].set_xscale("log"); ax[2].set_title("P(winner is an off-support plan)")
    ax[2].set_xlabel("N"); ax[2].legend(frameon=False)
    fig.tight_layout()
    return fig


def fig_estimator(df, N_show=(256, 4096)):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    d = df[(df.iters == 3) & (df.N.isin(N_show))]
    for tag in ("near", "far"):
        s = d[d.tag == tag]
        ax[0].scatter(s["est_optimism"], s["optimism"], s=10, alpha=0.5, color=C[tag], label=f"{tag} goals")
    lim = float(np.percentile(np.abs(np.r_[d["est_optimism"], d["optimism"]]), 99))
    ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    ax[0].set_xscale("symlog", linthresh=1e-3); ax[0].set_yscale("symlog", linthresh=1e-3)
    ax[0].set_xlabel("cross-fit estimate (needs no environment)"); ax[0].set_ylabel("true optimism (oracle)")
    ax[0].set_title("E3: estimated vs true optimism"); ax[0].legend(frameon=False)
    q = W.estimator_quality(df[df.iters == 3])
    names = ["auroc_est_optimism", "auroc_oth_std", "auroc_oth_mean", "auroc_pred_cost_baseline"]
    labels = ["cross-fit\nestimate", "evaluator\nspread", "evaluator\nmean cost", "predicted cost\n(baseline)"]
    vals = [q.get(n, np.nan) for n in names]
    ax[1].bar(labels, vals, color=["#0072B2", "#56B4E9", "#009E73", "#999999"])
    ax[1].axhline(0.5, color="k", lw=0.6)
    ax[1].set_ylim(0.3, 1); ax[1].set_ylabel("AUROC for predicting failure")
    ax[1].set_title("Which label-free signal flags failing plans?")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig, q


def fig_planners(dp, tag=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for j, t in enumerate(("near", "far")):
        d = dp[dp.tag == t]
        for meth, g in d.groupby("method"):
            s = g.groupby("N").agg(ev=("evals", "mean"), succ=("success", "mean"))
            if meth == "budget_select":
                ax[j].scatter(s["ev"], s["succ"], marker="*", s=130, color=MC[meth], label=meth, zorder=5)
            else:
                ax[j].plot(s["ev"], s["succ"], "-o", ms=3, color=MC[meth], label=meth)
        ax[j].set_xscale("log"); ax[j].set_title(f"{t} goals")
        ax[j].set_xlabel("model step-evaluations (compute)")
    ax[0].set_ylabel("true success rate"); ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E4: planners at matched compute", y=1.0)
    fig.tight_layout()
    return fig


def planner_table(dp):
    rows = []
    for (tag, meth), g in dp.groupby(["tag", "method"]):
        # best-compute-matched comparison: highest budget row per method
        gmax = g[g.N == g.N.max()]
        m, lo, hi = W.cluster_boot(gmax, "success")
        d, dlo, dhi = W.cluster_boot(gmax, "dist")
        rows.append(dict(tag=tag, method=meth, success=m, s_lo=lo, s_hi=hi, mean_dist=d,
                         evals=float(gmax["evals"].mean())))
    import pandas as pd
    return pd.DataFrame(rows).sort_values(["tag", "evals"])


# ----------------------------------------------------------------------------- pre-registered gates
def _paired(df, a_mask, b_mask, keys, col):
    """Mean(col | a) - mean(col | b) matched on `keys`; returns a frame with a 'pair' column."""
    A = df[a_mask].groupby(keys)[col].mean().rename("a")
    B = df[b_mask].groupby(keys)[col].mean().rename("b")
    j = A.to_frame().join(B, how="inner").reset_index()
    j["diff"] = j["a"] - j["b"]
    return j


def gates(df, dp=None, iters=3):
    """Evaluate the decision gates from the plan, on whichever results are passed in.
    Returns a list of dicts (gate, metric, value, CI, verdict)."""
    out = []
    d = df[df.iters == iters]
    Nlo, Nhi = d.N.min(), d.N.max()
    for tag in ("near", "far"):
        s = d[d.tag == tag]
        j = _paired(s, s.N == Nhi, s.N == Nlo, ["pair", "proposer", "ensemble"], "optimism") if "ensemble" in s else \
            _paired(s, s.N == Nhi, s.N == Nlo, ["pair", "proposer"], "optimism")
        m, lo, hi = W.cluster_boot(j.rename(columns={"diff": "x"}), "x")
        out.append(dict(gate="G1 optimism grows with N", subset=f"{tag} goals",
                        metric=f"optimism(N={Nhi}) - optimism(N={Nlo})", value=m, lo=lo, hi=hi,
                        verdict="PASS" if lo > 0 else "FAIL"))
    for tag in ("near", "far"):
        s = d[d.tag == tag]
        keys = ["pair", "proposer", "ensemble"] if "ensemble" in s else ["pair", "proposer"]
        Nmid = sorted(s.N.unique())[len(s.N.unique()) // 2]
        j = _paired(s, s.N == Nhi, s.N == Nmid, keys, "success")
        m, lo, hi = W.cluster_boot(j.rename(columns={"diff": "x"}), "x")
        v = "REVERSAL" if hi < 0 else ("SATURATED" if lo <= 0 <= hi else "STILL IMPROVING")
        out.append(dict(gate="G2 success vs more search", subset=f"{tag} goals",
                        metric=f"success(N={Nhi}) - success(N={Nmid})", value=m, lo=lo, hi=hi, verdict=v))
    q = W.estimator_quality(d)
    out.append(dict(gate="G3 estimator tracks optimism", subset="all", metric="Spearman within (N,iters) cell",
                    value=q["spearman_within_cell_mean"], lo=np.nan, hi=np.nan,
                    verdict="PASS" if q["spearman_within_cell_mean"] >= 0.3 else "FAIL"))
    if "auroc_est_optimism" in q:
        ok = q["auroc_est_optimism"] >= 0.7 and q["auroc_est_optimism"] > q["auroc_pred_cost_baseline"]
        out.append(dict(gate="G3b estimator flags failures", subset="all",
                        metric="AUROC(est) vs AUROC(predicted cost)", value=q["auroc_est_optimism"],
                        lo=q["auroc_pred_cost_baseline"], hi=np.nan, verdict="PASS" if ok else "FAIL"))
    if dp is not None and len(dp):
        for tag in ("far",):
            s = dp[(dp.tag == tag) & (dp.N == dp.N.max())]
            keys = ["pair", "proposer", "ensemble"] if "ensemble" in s else ["pair", "proposer"]
            for meth in ("crossfit", "lcb", "meanens", "budget_select"):
                if meth not in set(s.method):
                    continue
                j = _paired(s, s.method == meth, s.method == "single", keys, "success")
                m, lo, hi = W.cluster_boot(j.rename(columns={"diff": "x"}), "x")
                ev = float(s[s.method == meth].evals.mean() / s[s.method == "single"].evals.mean())
                out.append(dict(gate="G4 planner vs single CEM", subset=f"{tag} goals, {meth} ({ev:.1f}x compute)",
                                metric="success difference", value=m, lo=lo, hi=hi,
                                verdict="PASS" if lo > 0 else ("TIE" if hi >= 0 else "WORSE")))
    import pandas as pd
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- support-continuous analysis
def support_interaction(df, iters=3, n_boot=1000, seed=0):
    """Does optimism grow with search MORE when the goal is further from the training data?
    OLS: optimism ~ 1 + log2(N) + support + log2(N)*support  (both centred/standardised).
    The interaction coefficient is the test. 95% CIs by cluster bootstrap over start-goal pairs.
    Uses the continuous `support_d` (mean distance of the goal to its 10 nearest training positions),
    which avoids relying on the near/far label alone."""
    import pandas as pd
    d = df[df.iters == iters].copy()
    d["lN"] = np.log2(d["N"]); d["lN"] -= d["lN"].mean()
    d["sz"] = (d["support_d"] - d["support_d"].mean()) / d["support_d"].std()
    d["ix"] = d["lN"] * d["sz"]

    def fit(x):
        X = np.column_stack([np.ones(len(x)), x["lN"], x["sz"], x["ix"]])
        return np.linalg.lstsq(X, x["optimism"].values, rcond=None)[0]

    b = fit(d)
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in d.groupby("pair")}
    keys = list(groups)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        bs.append(fit(pd.concat([groups[keys[i]] for i in pick])))
    bs = np.array(bs)
    names = ["intercept", "log2(N) slope", "support distance (SD)", "log2(N) x support (the test)"]
    return pd.DataFrame(dict(term=names, coef=b, lo=np.percentile(bs, 2.5, 0), hi=np.percentile(bs, 97.5, 0)))


def fig_support(df, iters=3, bins=3):
    """Optimism vs N for terciles of support distance."""
    d = df[df.iters == iters].copy()
    d["bin"] = np.digitize(d["support_d"], np.quantile(d["support_d"], np.linspace(0, 1, bins + 1)[1:-1]))
    names = {0: "closest to data", 1: "middle", 2: "furthest from data"}
    cols = ["#0072B2", "#999999", "#D55E00"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for b in range(bins):
        s = d[d.bin == b]
        _band(ax[0], s, "optimism", cols[b], names[b])
        _band(ax[1], s, "success", cols[b], names[b])
    ax[0].set_ylabel("optimism"); ax[1].set_ylabel("true success rate")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("search budget N")
    ax[0].legend(frameon=False)
    fig.suptitle("Optimism and success by distance of the goal from the training data", y=1.0)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- stricter check of the cross-fit result
def crossfit_check(df, dp, iters=3, Nplan=None):
    """Is the cross-fit gain real, or just 'more compute'? Compares cross-fit at the planner-grid maximum N
    against SINGLE-model CEM at that N and at larger N (up to 16x/4x more compute), matched on the same
    start-goal pair and proposer, per ensemble too. Needs the sweep (df) and planner (dp) tables."""
    import pandas as pd
    Nplan = Nplan or int(dp.N.max())
    cf = dp[(dp.method == "crossfit") & (dp.N == Nplan)]
    props = sorted(cf.proposer.unique())
    sw = df[(df.iters == iters) & (df.proposer.isin(props))]
    rows = []
    for tag in ("near", "far"):
        a_all = cf[cf.tag == tag]
        for Nref in sorted(n for n in sw.N.unique() if n >= Nplan):
            b_all = sw[(sw.tag == tag) & (sw.N == Nref)]
            for ens in ["all"] + sorted(a_all["ensemble"].unique()) if "ensemble" in a_all else ["all"]:
                a, b = a_all, b_all
                if ens != "all":
                    a, b = a[a.ensemble == ens], b[b.ensemble == ens]
                keys = ["pair", "proposer"]
                j = a.groupby(keys)["success"].mean().rename("a").to_frame().join(
                    b.groupby(keys)["success"].mean().rename("b"), how="inner").reset_index()
                if len(j) == 0:
                    continue
                j["x"] = j["a"] - j["b"]
                m, lo, hi = W.cluster_boot(j, "x")
                rows.append(dict(tag=tag, ensemble=ens, vs=f"single CEM N={Nref}",
                                 compute_ratio_single_over_crossfit=float(b["evals"].mean() / a["evals"].mean()),
                                 crossfit_minus_single=m, lo=lo, hi=hi,
                                 verdict="crossfit better" if lo > 0 else ("single better" if hi < 0 else "tie")))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- confound control for the support test
def add_pair_meta(df, M):
    """Re-create each start-goal pair (they are generated deterministically from the ensemble seed) and add
    `sg_dist` (start-to-goal distance) and `goal_x`. Needs df['pair'] = ensemble*1000 + pair_index and an
    'ensemble' column; M = number of pairs per ensemble (CPU pilot 36, GPU run 60)."""
    import pandas as pd
    rows = []
    for e in sorted(df["ensemble"].unique()):
        for p in W.make_pairs(M=M, seed=1000 + int(e), data_pos=None):
            rows.append(dict(ensemble=e, pair_idx=p["pair"],
                             sg_dist=float(np.linalg.norm(p["g"] - p["s0"][:2])), goal_x=float(p["g"][0])))
    d = df.copy()
    d["pair_idx"] = d["pair"] % 1000
    return d.merge(pd.DataFrame(rows), on=["ensemble", "pair_idx"], how="left")


def support_interaction_ctrl(df, ctrl="sg_dist", iters=3, n_boot=1000, seed=0):
    """Like support_interaction, but also lets optimism's growth with N depend on a control (default:
    start-goal distance). If the 'log2(N) x support' row survives, the effect is not just 'far goals are longer'."""
    import pandas as pd
    d = df[df.iters == iters].copy()
    d["lN"] = np.log2(d["N"]); d["lN"] -= d["lN"].mean()
    d["sz"] = (d["support_d"] - d["support_d"].mean()) / d["support_d"].std()
    d["cz"] = (d[ctrl] - d[ctrl].mean()) / d[ctrl].std()
    d["ix"], d["ic"] = d["lN"] * d["sz"], d["lN"] * d["cz"]
    cols = ["lN", "sz", "cz", "ix", "ic"]

    def fit(x):
        X = np.column_stack([np.ones(len(x))] + [x[c] for c in cols])
        return np.linalg.lstsq(X, x["optimism"].values, rcond=None)[0]

    b = fit(d)
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in d.groupby("pair")}
    keys = list(groups)
    bs = np.array([fit(pd.concat([groups[keys[i]] for i in rng.choice(len(keys), len(keys), replace=True)]))
                   for _ in range(n_boot)])
    names = ["intercept", "log2(N) slope", "support distance (SD)", f"{ctrl} (SD)",
             "log2(N) x support (the test)", f"log2(N) x {ctrl}"]
    return pd.DataFrame(dict(term=names, coef=b, lo=np.percentile(bs, 2.5, 0), hi=np.percentile(bs, 97.5, 0)))