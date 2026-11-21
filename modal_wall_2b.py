"""
modal_wall_2b.py -- Phase 2b: the real-model test of the winner's curse / held-out re-ranking on released Wall checkpoints.

WHY a custom runner: the repo's eval costs ~395 s per episode (30 CEM iterations x 300 samples, fp32), so a sample-count
sweep through it is infeasible. This script reuses the repo's own model loading, env, goal sampling, objective and CEM
cost function, but runs a lean CEM (3 iterations, N = 16..1024) and scores every chosen plan in the simulator.

Per start/goal pair, for each proposer in {JEPA-WM, DINO-WM} (the other model is the held-out evaluator):
  single     : the proposer's own CEM winner at N samples                    (the planner everyone uses)
  crossfit   : top-m (m = max(8, N//10)) of that CEM re-ranked by the OTHER model's latent cost   (~1.1x compute)
  rankavg    : top-m re-ranked by (rank under proposer + rank under other)
Outcome = simulator end distance and success (< 4.5), exactly the repo's definition. The open-loop plan IS the whole
episode in this config (H=6, 6 actions stepped), so the simulator result is the true outcome of the chosen plan.
Support: distance of the goal to the nearest training-set positions (wall_single/states.pth).

Run (smoke first; prints shapes, timing and an fp32-vs-bf16 sanity check):
    python -m modal run modal_wall_2b.py --smoke
    python -m modal run --detach modal_wall_2b.py --pairs 48 --shards 4
Fetch if detached:   python -m modal volume get wcp-results out/ results/out/ --force

HONEST STATUS: written from the repo code your probes printed; never executed. The smoke run is the test.
"""
import modal

app = modal.App("wcp-wall-2b")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0", "libegl1", "libosmesa6-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/jepa-wms /opt/jepa-wms",
        "cd /opt/jepa-wms && pip install -q -e . 2>&1 | tail -3",
        "pip install -q pandas pyyaml 2>&1 | tail -1",
    )
)
vol = modal.Volume.from_name("wcp-results", create_if_missing=True)

ROOT = "/opt/jepa-wms"
DSET = "/results/jepawm_dset"
LOGS = "/results/jepawm_logs"
URLS = {
    "jepa": "https://dl.fbaipublicfiles.com/jepa-wms/wall_jepa-wm.pth.tar",
    "dino": "https://dl.fbaipublicfiles.com/jepa-wms/wall_dino-wm.pth.tar",
}
CFGS = {
    "jepa": "configs/evals/simu_env_planning/wall/jepa-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
    "dino": "configs/evals/simu_env_planning/wall/dino-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
}


@app.function(image=image, gpu="L4", timeout=3 * 3600, volumes={"/results": vol})
def shard(cfgd: dict) -> dict:
    import contextlib, math, os, sys, time, traceback, zlib
    os.environ.update(JEPAWM_DSET=DSET, JEPAWM_LOGS=LOGS, JEPAWM_HOME="/opt", JEPAWM_CKPT=LOGS,
                      JEPAWM_OSSCKPT=LOGS, PYTHONUNBUFFERED="1")
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import numpy as np, pandas as pd, torch, yaml
    from einops import rearrange
    from omegaconf import OmegaConf
    from src.utils.yaml_utils import expand_env_vars
    from evals.utils import make_datasets, prepare_obs
    from evals.simu_env_planning.eval import init_module
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.planning.common.parser import parse_cfg
    from evals.simu_env_planning.planning.gc_agent import GC_Agent
    from evals.simu_env_planning.planning.utils import make_td

    smoke = cfgd["smoke"]
    Ns = cfgd["Ns"]
    iters = 3
    amp = cfgd["amp"]
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    t00 = time.time()

    def log(*a):
        print(f"[shard {cfgd['shard']} {time.time()-t00:6.0f}s]", *a, flush=True)

    # ------------------------------------------------------------------ build both models (repo code, patched config)
    def build(key):
        cfg0 = yaml.safe_load(open(os.path.join(ROOT, CFGS[key])))
        cfg0["folder"] = f"{LOGS}/w2b/{key}"
        cfg0["tag"] = "w2b"
        cfg0["meta"]["eval_episodes"] = 1
        cfg0["logging"]["optional_plots"] = False
        cfg0["model_kwargs"]["checkpoint"] = URLS[key]
        cfg0["model_kwargs"]["pretrain_kwargs"]["heads_cfg"]["pretrain_dec_path"] = None   # decoders not needed
        cfg0["planner"]["decode_each_iteration"] = False
        args = expand_env_vars(cfg0)
        mk = args["model_kwargs"]
        cd, cda = mk.get("data", {}), mk.get("data_aug", {})
        dset, pre = make_datasets(cd, cda, 1, 0)
        args["frameskip"] = cd["custom"]["frameskip"]
        os.makedirs(f"{LOGS}/w2b/{key}", exist_ok=True)
        args["work_dir"] = f"{LOGS}/w2b/{key}"
        model = init_module(folder=args["folder"], checkpoint=mk["checkpoint"], module_name=mk["module_name"],
                            model_kwargs=mk["pretrain_kwargs"], wrapper_kwargs=mk.get("wrapper_kwargs", {}),
                            cfgs_data=cd, device=dev, action_dim=dset.action_dim, proprio_dim=dset.proprio_dim,
                            preprocessor=pre)
        cfg = OmegaConf.create(args)
        try:
            cfg = parse_cfg(cfg)
        except Exception:
            log("parse_cfg failed, using manual defaults:\n" + traceback.format_exc()[-600:])
            cfg.task_specification.multitask = False
            cfg.tasks = [cfg.task_specification.task]
        cfg.rank, cfg.world_size, cfg.device, cfg.local_seed = 0, 1, "cuda:0", int(cfg.meta.seed)
        cfg.planner.distribute_planner = False
        cfg.task_specification.goal_source = cfg.task_specification.get("goal_source", "random_state")
        cfg.action_ratio = cfg.frameskip // model.action_skip
        return cfg, model, dset, pre

    log("building JEPA-WM ...")
    cfg_j, model_j, dset_j, pre = build("jepa")
    log("building DINO-WM ...")
    cfg_d, model_d, dset_d, _ = build("dino")
    env = make_env(cfg_j)
    ag = {"jepa": GC_Agent(cfg_j, model_j, dset=dset_j, preprocessor=pre),
          "dino": GC_Agent(cfg_d, model_d, dset=dset_d, preprocessor=pre)}
    H, Ad = int(cfg_j.planner.horizon), int(model_j.action_dim)
    obs_type = cfg_j.task_specification.obs
    log(f"models ready. H={H} action_dim(model)={Ad} env.action_dim={env.action_dim} frameskip={cfg_j.frameskip}")

    # ------------------------------------------------------------------ support: distance to training positions
    st = torch.load(f"{DSET}/wall_single/states.pth").float().reshape(-1, 2)
    g = torch.Generator().manual_seed(0)
    train_pos = st[torch.randperm(st.shape[0], generator=g)[:20000]].to(dev)

    def support_d(pos):
        d = torch.cdist(torch.as_tensor(pos, dtype=torch.float32, device=dev)[None, :2], train_pos)[0]
        return float(d.topk(10, largest=False).values.mean())

    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else contextlib.nullcontext

    @torch.no_grad()
    def costs(a, z, A, chunk=512):
        out = []
        for i in range(0, A.shape[1], chunk):
            with ctx():
                out.append(a.planner.cost_function(A[:, i:i + chunk], z).float())
        return torch.nan_to_num(torch.cat(out), nan=1e6, posinf=1e6, neginf=-1e6)

    @torch.no_grad()
    def cem(a, z, N, gen):
        mean = torch.zeros(H, Ad, device=dev)
        std = torch.ones(H, Ad, device=dev)
        ne = max(4, int(0.1 * N))
        for it in range(iters):
            A = mean[:, None] + std[:, None] * torch.randn(H, N, Ad, device=dev, generator=gen)
            A[:, 0] = mean
            c = costs(a, z, A)
            if it < iters - 1:
                E = A[:, torch.topk(-c, ne).indices]
                mean, std = E.mean(1), E.std(1)
        return A, c

    def oracle(seed, init_s, state_g, plan):
        env.prepare(seed, init_s)
        act = rearrange(plan.detach().float().cpu(), "t (f d) -> (t f) d", d=env.action_dim)
        act = pre.denormalize_actions(act)
        obses, rews, dones, infos = env.step_multiple(act)
        ev = env.eval_state(state_g, infos[-1]["state"])
        return float(ev["state_dist"]), bool(ev["success"])

    def setup_pair(seed):
        init_s, goal_s = env.sample_random_init_goal_states(seed)
        g_obs, g_info = env.prepare(seed, goal_s)
        g_td = make_td(g_obs, g_info)
        i_obs, i_info = env.prepare(seed, init_s)
        i_td = make_td(i_obs, i_info)
        return init_s, goal_s, g_td, i_td, g_info["state"]

    rows = []
    seeds = [cfgd["base_seed"] + i for i in range(cfgd["shard"], cfgd["pairs"], cfgd["shards"])]
    if smoke:
        seeds = seeds[:2]
    for si, seed in enumerate(seeds):
        tp = time.time()
        init_s, goal_s, g_td, i_td, state_g = setup_pair(seed)
        zs = {}
        for key in ("jepa", "dino"):
            ag[key].set_goal(prepare_obs(obs_type, g_td))
            zs[key] = ag[key].model.encode(prepare_obs(obs_type, i_td).to(dev).unsqueeze(0), act=True)
        sd = support_d(goal_s)
        if smoke and si == 0:
            try:
                A = torch.randn(H, 64, Ad, device=dev)
                c32 = ag["jepa"].planner.cost_function(A, zs["jepa"]).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    c16 = ag["jepa"].planner.cost_function(A, zs["jepa"]).float()
                from scipy.stats import spearmanr
                log(f"SANITY fp32 vs bf16 cost: spearman={spearmanr(c32.cpu(), c16.cpu())[0]:.4f} "
                    f"max_rel_err={((c32-c16).abs()/(c32.abs()+1e-9)).max().item():.4f}")
                torch.cuda.synchronize(); t0 = time.time()
                costs(ag["jepa"], zs["jepa"], torch.randn(H, 1024, Ad, device=dev)); torch.cuda.synchronize()
                log(f"TIMING cost of 1024 plans (amp={amp}): {time.time()-t0:.2f}s")
                log(f"SHAPES z={tuple(zs['jepa']['visual'].shape) if hasattr(zs['jepa'],'keys') else tuple(zs['jepa'].shape)} "
                    f"A={tuple(A.shape)} cost={tuple(c32.shape)} init_s={np.asarray(init_s)} goal_s={np.asarray(goal_s)} support_d={sd:.2f}")
            except Exception:
                log("sanity block failed:\n" + traceback.format_exc()[-900:])
        for prop, oth in (("jepa", "dino"), ("dino", "jepa")):
            for N in Ns:
                gen = torch.Generator(device=dev).manual_seed(zlib.crc32(f"{seed}-{prop}-{N}".encode()))
                Z, c = cem(ag[prop], zs[prop], N, gen)
                m = max(8, N // 10)
                top = torch.topk(-c, min(m, N)).indices
                co = costs(ag[oth], zs[oth], Z[:, top])
                cp_top = c[top]
                i_single = int(c.argmin())
                i_cross = int(top[co.argmin()])
                rank_p = cp_top.argsort().argsort().float()
                rank_o = co.argsort().argsort().float()
                i_rank = int(top[(rank_p + rank_o).argmin()])
                po_single = float(costs(ag[oth], zs[oth], Z[:, i_single:i_single + 1])[0])
                for meth, idx, ev in (("single", i_single, N * iters), ("crossfit", i_cross, N * iters + m),
                                      ("rankavg", i_rank, N * iters + m)):
                    dist, ok = oracle(seed, init_s, state_g, Z[:, idx])
                    rows.append(dict(seed=seed, pair=si, proposer=prop, method=meth, N=N, evals=ev,
                                     pred_prop=float(c[idx]), pred_single=float(c[i_single]), pred_other_single=po_single,
                                     dist=dist, success=int(ok), support_d=sd, goal_x=float(goal_s[0]), goal_y=float(goal_s[1]),
                                     shard=cfgd["shard"]))
        log(f"pair {si+1}/{len(seeds)} seed={seed} support_d={sd:.1f} took {time.time()-tp:.0f}s")
        if si % 3 == 2:
            pd.DataFrame(rows).to_csv(f"/results/out/wall2b_{cfgd['tag']}_s{cfgd['shard']}.csv", index=False)
            vol.commit()
    df = pd.DataFrame(rows)
    os.makedirs("/results/out", exist_ok=True)
    df.to_csv(f"/results/out/wall2b_{cfgd['tag']}_s{cfgd['shard']}.csv", index=False)
    vol.commit()
    secs = time.time() - t00
    log(f"done in {secs:.0f}s (~${secs*0.000222:.2f} at the L4 rate)")
    return dict(csv=df.to_csv(index=False), seconds=secs)


@app.local_entrypoint()
def main(smoke: bool = False, pairs: int = 48, shards: int = 4, amp: bool = True,
         ns: str = "16,64,256,1024", base_seed: int = 777, tag: str = "run"):
    import os
    os.makedirs("results", exist_ok=True)
    Ns = [int(x) for x in ns.split(",")]
    if smoke:
        Ns, shards, tag = [16, 64], 1, "smoke"
    cfgs = [dict(shard=i, shards=shards, pairs=pairs, smoke=smoke, Ns=Ns, amp=amp, base_seed=base_seed, tag=tag)
            for i in range(shards)]
    outs = list(shard.map(cfgs))
    import io, pandas as pd
    df = pd.concat([pd.read_csv(io.StringIO(o["csv"])) for o in outs], ignore_index=True)
    path = f"results/wall2b_{tag}.csv"
    df.to_csv(path, index=False)
    secs = sum(o["seconds"] for o in outs)
    print(f"saved {path}: {len(df)} rows | total GPU seconds {secs:.0f} ~ ${secs*0.000222:.2f} (L4, compute only)")
    if not df.empty:
        g = df.groupby(["proposer", "method", "N"]).agg(success=("success", "mean"), dist=("dist", "mean"),
                                                        n=("success", "size")).round(3)
        print(g.to_string())