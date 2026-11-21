"""
modal_app.py -- GPU version of the pilot for Modal (scaled-up E1/E3/E4).

Same environment, planners and output columns as the CPU pilot in wcp_core.py, but the
ensemble is trained with torch on the GPU and planning is batched on the GPU, so you can use
much larger N (up to 65,536), a bigger MLP and more pairs/ensembles.

Run from the notebook (or a VS Code terminal) inside this folder:
    modal run modal_app.py --smoke          # ~minutes; ALWAYS do this first and check the cost
    modal run modal_app.py --seeds 3 --n-pairs 60   # full run, 3 independent ensembles in parallel

NOTE (honest status): written and syntax-checked in a sandbox WITHOUT torch/Modal, so this file has
not been executed. The smoke run is the test. Expect to fix small API details if Modal/torch versions differ.
"""
import time
import modal

app = modal.App("wcp-pilot")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "pandas", "scikit-learn", "torch")
    .add_local_python_source("wcp_core")
)
vol = modal.Volume.from_name("wcp-results", create_if_missing=True)


@app.function(image=image, gpu="L4", timeout=3 * 3600, volumes={"/results": vol})
def run_gpu(cfg: dict) -> dict:
    import numpy as np, pandas as pd, torch
    import wcp_core as W

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(cfg["seed"])
    smoke = bool(cfg.get("smoke", False))
    K, hid = int(cfg.get("K", 5)), int(cfg.get("hidden", 128))
    steps = 300 if smoke else int(cfg.get("steps", 8000))
    M = 4 if smoke else int(cfg.get("M", 60))
    Ns = [16, 256] if smoke else cfg.get("Ns", [16, 64, 256, 1024, 4096, 16384, 65536])
    iters_list = [1, 3] if smoke else cfg.get("iters", [1, 3, 6])
    Ns_plan = [16, 256] if smoke else cfg.get("Ns_plan", [16, 64, 256, 1024, 4096])
    t0 = time.time()
    torch.manual_seed(seed)

    # ------------------------------------------------------------------ data + ensemble training
    S, A, S2 = W.make_data(n_traj=300 if smoke else 2000, T=60, seed=seed)
    X = np.concatenate([S, A], -1).astype(np.float32)
    Y = (S2 - S).astype(np.float32)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    Xt = torch.tensor((X - xm) / xs, device=dev)
    Yt = torch.tensor((Y - ym) / ys, device=dev)
    xm_t, xs_t, ym_t, ys_t = [torch.tensor(v, device=dev) for v in (xm, xs, ym, ys)]

    g_cpu = torch.Generator().manual_seed(seed)

    def init(i, o):
        return (torch.randn(K, i, o, generator=g_cpu) / i ** 0.5).to(dev).requires_grad_(True)

    def zeros(o):
        return torch.zeros(K, 1, o, device=dev).requires_grad_(True)

    P = [init(6, hid), zeros(hid), init(hid, hid), zeros(hid), init(hid, 4), zeros(4)]

    def fwd(P_, x):                                   # x: (k, B, 6) -> (k, B, 4)
        h = torch.relu(torch.baddbmm(P_[1], x, P_[0]))
        h = torch.relu(torch.baddbmm(P_[3], h, P_[2]))
        return torch.baddbmm(P_[5], h, P_[4])

    opt = torch.optim.Adam(P, lr=1e-3)
    n, B = Xt.shape[0], 512
    for it in range(steps):
        idx = torch.randint(0, n, (K, B), device=dev)          # independent minibatches per member
        loss = ((fwd(P, Xt[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 1000 == 0:
            print(f"[seed {seed}] step {it} loss {loss.item():.4f}", flush=True)
    for p in P:
        p.requires_grad_(False)
    print(f"[seed {seed}] trained in {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ planning (batched on GPU)
    BLOCK, NB, H = W.BLOCK, W.NB, W.H

    def member(k):
        return [p[k:k + 1] for p in P]

    @torch.no_grad()
    def pcost(Pk, s0, Z, g):                           # Z: (N, NB, 2) -> (N,) predicted cost
        N = Z.shape[0]
        Aexp = Z.repeat_interleave(BLOCK, dim=1)
        s = s0[None].expand(N, 4).clone()
        for t in range(H):
            x = (torch.cat([s, Aexp[:, t]], -1) - xm_t) / xs_t
            s = s + fwd(Pk, x[None])[0] * ys_t + ym_t
        d = s[:, :2] - g
        return (d * d).sum(-1) + 0.05 * (s[:, 2:] ** 2).sum(-1)

    @torch.no_grad()
    def cem(costfn, N, iters, gen, elite_frac=0.1, std0=0.8):
        mu = torch.zeros(NB, 2, device=dev)
        sd = torch.full((NB, 2), std0, device=dev)
        ne = max(4, int(N * elite_frac))
        for it_ in range(iters):
            Z = (mu + sd * torch.randn(N, NB, 2, device=dev, generator=gen)).clamp(-1, 1)
            c = costfn(Z)
            if it_ < iters - 1:
                E = Z[torch.topk(c, ne, largest=False).indices]
                mu, sd = E.mean(0), E.std(0) + 1e-3
        return Z, c

    def others_cost(k, s0, Z, g):                      # (K-1, N) costs by the other members
        return torch.stack([pcost(member(j), s0, Z, g) for j in range(K) if j != k])

    def plan(meth, k, s0, g, N, iters, gen):
        Pk = member(k)
        if meth == "single":
            Z, c = cem(lambda Z: pcost(Pk, s0, Z, g), N, iters, gen)
            return Z[c.argmin()], W.evals_of(N, iters)
        if meth == "meanens":
            f = lambda Z: torch.stack([pcost(member(j), s0, Z, g) for j in range(K)]).mean(0)
            Z, c = cem(f, N, iters, gen)
            return Z[c.argmin()], W.evals_of(N, iters, K)
        if meth == "lcb":
            def f(Z):
                C = torch.stack([pcost(member(j), s0, Z, g) for j in range(K)])
                return C.mean(0) + C.std(0, unbiased=False)
            Z, c = cem(f, N, iters, gen)
            return Z[c.argmin()], W.evals_of(N, iters, K)
        if meth == "crossfit":
            Z, c = cem(lambda Z: pcost(Pk, s0, Z, g), N, iters, gen)
            m = max(8, N // 10)
            top = torch.topk(c, min(m, N), largest=False).indices
            Ce = others_cost(k, s0, Z[top], g).mean(0)
            return Z[top[Ce.argmin()]], W.evals_of(N, iters) + m * H * (K - 1)
        raise ValueError(meth)

    def plan_select(k, s0, g, grid, iters, gen):
        best, best_c, ev = None, float("inf"), 0
        for N in grid:
            Z, c = cem(lambda Z: pcost(member(k), s0, Z, g), N, iters, gen)
            z = Z[c.argmin()]
            ce = float(others_cost(k, s0, z[None], g).mean())
            ev += W.evals_of(N, iters) + H * (K - 1)
            if ce < best_c:
                best, best_c = z, ce
        return best, ev

    # ------------------------------------------------------------------ pairs + experiments
    pairs = W.make_pairs(M=M, seed=1000 + seed, data_pos=S[:, :2])
    sweep_rows, plan_rows = [], []
    for pr in pairs:
        s0n, gn = pr["s0"], pr["g"]
        s0 = torch.tensor(s0n, device=dev)
        g = torch.tensor(gn, device=dev)
        for k in range(K):
            for iters in iters_list:
                for N in Ns:
                    gen = torch.Generator(device=dev).manual_seed(W._seed(seed, pr["pair"], k, iters, N))
                    Z, c = cem(lambda Z: pcost(member(k), s0, Z, g), N, iters, gen)
                    z = Z[c.argmin()]
                    Co = others_cost(k, s0, z[None], g)[:, 0].cpu().numpy()
                    oc, od = W.oracle(s0n, z.cpu().numpy()[None], gn)
                    sweep_rows.append(dict(
                        pair=pr["pair"], tag=pr["tag"], support_d=pr["support_d"], proposer=k, N=N,
                        iters=iters, evals=W.evals_of(N, iters), pred=float(c.min().item()),
                        oracle=float(oc[0]), dist=float(od[0]), oth_mean=float(Co.mean()),
                        oth_std=float(Co.std()), success=int(od[0] < W.GOAL_TOL), ensemble=seed))
        for k in range(min(K, 3)):
            for meth in ("single", "meanens", "lcb", "crossfit"):
                for N in Ns_plan:
                    gen = torch.Generator(device=dev).manual_seed(W._seed(seed, "p", pr["pair"], k, meth, N))
                    z, ev = plan(meth, k, s0, g, N, 3, gen)
                    oc, od = W.oracle(s0n, z.cpu().numpy()[None], gn)
                    plan_rows.append(dict(pair=pr["pair"], tag=pr["tag"], proposer=k, method=meth, N=N,
                                          evals=ev, oracle=float(oc[0]), dist=float(od[0]),
                                          success=int(od[0] < W.GOAL_TOL), ensemble=seed))
            gen = torch.Generator(device=dev).manual_seed(W._seed(seed, "sel", pr["pair"], k))
            z, ev = plan_select(k, s0, g, Ns_plan, 3, gen)
            oc, od = W.oracle(s0n, z.cpu().numpy()[None], gn)
            plan_rows.append(dict(pair=pr["pair"], tag=pr["tag"], proposer=k, method="budget_select",
                                  N=int(max(Ns_plan)), evals=ev, oracle=float(oc[0]), dist=float(od[0]),
                                  success=int(od[0] < W.GOAL_TOL), ensemble=seed))
        if pr["pair"] % 5 == 0:
            print(f"[seed {seed}] pair {pr['pair']+1}/{len(pairs)}  {time.time()-t0:.0f}s", flush=True)

    sw = pd.DataFrame(sweep_rows)
    sw["optimism"] = sw["oracle"] - sw["pred"]
    sw["est_optimism"] = sw["oth_mean"] - sw["pred"]
    pl = pd.DataFrame(plan_rows)
    tag = "smoke" if smoke else "full"
    sw.to_csv(f"/results/gpu_sweep_{tag}_e{seed}.csv", index=False)
    pl.to_csv(f"/results/gpu_planners_{tag}_e{seed}.csv", index=False)
    vol.commit()
    secs = time.time() - t0
    print(f"[seed {seed}] done in {secs:.0f}s (~${secs * 0.000222:.2f} at the L4 rate)", flush=True)
    return dict(sweep=sw.to_dict("records"), planners=pl.to_dict("records"), seconds=secs)


@app.local_entrypoint()
def main(smoke: bool = False, seeds: int = 3, n_pairs: int = 60):
    import os, pandas as pd
    os.makedirs("results", exist_ok=True)
    cfgs = [dict(seed=s, smoke=smoke, M=n_pairs) for s in range(100, 100 + (1 if smoke else seeds))]
    outs = list(run_gpu.map(cfgs))
    tag = "smoke" if smoke else "full"
    pd.DataFrame([r for o in outs for r in o["sweep"]]).to_csv(f"results/gpu_sweep_{tag}.csv", index=False)
    pd.DataFrame([r for o in outs for r in o["planners"]]).to_csv(f"results/gpu_planners_{tag}.csv", index=False)
    secs = sum(o["seconds"] for o in outs)
    print(f"saved results/gpu_*_{tag}.csv | total GPU seconds {secs:.0f} ~ ${secs * 0.000222:.2f} (L4, compute only)")