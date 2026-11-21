"""
modal_wall.py -- Phase 2a: run the released JEPA-WM / DINO-WM Wall checkpoints through the repo's OWN eval pipeline on Modal.

What this does (everything uses the repo's code; only the config is patched):
  stage "data"   : find + download the Wall dataset from the HF dataset repo, print the layout, dump the code
                   we still need to read (gc_agent.py, evals/utils.py, envs/init.py, wall env, dataset class).
  stage "repro"  : run the released config (CEM, 30 iters x 300 samples, H=6, randstate goals) for one model and
                   report success rate -> compare with the number published in the paper/README.
  stage "sweep"  : same, but num_samples in a grid (default 30,100,300,1000,3000) -> real-model scaling curve of the
                   simulator success rate vs planner compute. (Closed-form oracle, no optimism yet: that comes in 2b.)

Config patches (documented so nothing is hidden):
  * decoder heads removed (pretrain_dec_path: null) and decode_each_iteration/optional_plots off -> planning does not need them;
    this also avoids the ${JEPAWM_LOGS} state-head path that broke torch.hub.load in probe 2.
  * model checkpoint = released URL; log folder = /results/wall_runs/...; eval_episodes / num_samples from CLI.

Run (from this folder):
    modal run modal_wall.py --stage data
    modal run modal_wall.py --stage repro --model jepa --episodes 96
    modal run modal_wall.py --stage sweep --model jepa --episodes 48
Output: results/wall_<stage>_<model>.txt  (attach this to the chat)

HONEST STATUS: written from the code printed by your probes, never executed. Expect to fix paths on the first run;
the "data" stage exists precisely to show the layout the dataset class expects.
"""
import modal

app = modal.App("wcp-wall")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0", "libegl1", "libosmesa6-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/jepa-wms /opt/jepa-wms",
        "cd /opt/jepa-wms && pip install -q -e . 2>&1 | tail -3",
        "pip install -q huggingface_hub pyyaml pandas 2>&1 | tail -1",
    )
)
vol = modal.Volume.from_name("wcp-results", create_if_missing=True)
# The gated HF dataset needs your token INSIDE the container. A key stored in Modal is only visible to a function that
# lists its secret here. Name = the secret you created in the Modal dashboard (check: `modal secret list`).
# Override without editing: set WCP_HF_SECRET=<name> in the environment before `modal run`.
import os as _os
HF_SECRET = _os.environ.get("WCP_HF_SECRET", "huggingface-secret")

URLS = {
    "jepa": "https://dl.fbaipublicfiles.com/jepa-wms/wall_jepa-wm.pth.tar",
    "dino": "https://dl.fbaipublicfiles.com/jepa-wms/wall_dino-wm.pth.tar",
}
CFGS = {
    "jepa": "configs/evals/simu_env_planning/wall/jepa-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
    "dino": "configs/evals/simu_env_planning/wall/dino-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
}
ROOT = "/opt/jepa-wms"
DSET = "/results/jepawm_dset"
LOGS = "/results/jepawm_logs"


def _sh(cmd, n=8000, t=1800, cwd=ROOT):
    import subprocess
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=t)
        return (r.stdout + r.stderr)[-n:]
    except subprocess.TimeoutExpired:
        return "TIMEOUT: " + cmd


def _env():
    import os
    e = dict(os.environ)
    e.update(JEPAWM_DSET=DSET, JEPAWM_LOGS=LOGS, JEPAWM_HOME="/opt", JEPAWM_CKPT=LOGS, JEPAWM_OSSCKPT=LOGS,
             PYTHONPATH=ROOT, MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
    return e


def hf_token():
    import os
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(k):
            os.environ["HF_TOKEN"] = os.environ[k]
            return os.environ[k]
    raise RuntimeError("No HF token in the container. The Modal secret must contain a variable named HF_TOKEN "
                       "(or HUGGING_FACE_HUB_TOKEN). Seen env names: "
                       + ", ".join(sorted(k for k in os.environ if "HF" in k.upper() or "HUGG" in k.upper())))


def _link_and_resolve() -> str:
    """Symlink the extracted folder under the other places the repo might look, and print what dataset_paths() says."""
    import os
    real = f"{DSET}/wall/wall_single"
    for alt in (f"{DSET}/wall_single", f"{DSET}/Wall", f"{DSET}/wall_single/wall_single"):
        if not os.path.exists(alt) and os.path.isdir(real):
            try:
                os.makedirs(os.path.dirname(alt), exist_ok=True)
                os.symlink(real, alt)
            except OSError:
                pass
    code = ("import os; from src.datasets.utils.utils import dataset_paths as d; p=d(); "
            "print({k: p.get(k) for k in ('Wall','wall')})")
    import subprocess
    r = subprocess.run(["python", "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True)
    return "dataset_paths -> " + (r.stdout + r.stderr)[-600:]


def ensure_wall_data() -> str:
    """The repo's WallDataset needs an EXTRACTED folder (states.pth, actions.pth, door_locations.pth,
    wall_locations.pth, obses/episode_XXX.pth). HF ships it as wall/wall_single.zip, so unzip once."""
    import os, zipfile
    found = _sh(f"find {DSET} -name states.pth | head -3")
    if "states.pth" in found:
        return "wall data already extracted:\n" + found + "\n" + _link_and_resolve()
    z = f"{DSET}/wall/wall_single.zip"
    if not os.path.exists(z):
        return "wall_single.zip not found (run --stage data first)"
    with zipfile.ZipFile(z) as f:
        f.extractall(f"{DSET}/wall")
    return ("extracted wall_single.zip:\n" + _sh(f"find {DSET}/wall -maxdepth 3 | head -20; du -sh {DSET}/wall")
            + "\n" + _link_and_resolve())


def stage_data() -> str:
    import os
    from huggingface_hub import HfApi, snapshot_download
    out = []
    tok = hf_token()
    who = HfApi().whoami(token=tok)
    out.append(f"HF token OK, user={who.get('name')} type={who.get('auth',{}).get('accessToken',{}).get('role')}")
    api = HfApi(token=tok)
    files = api.list_repo_files("facebook/jepa-wms", repo_type="dataset")
    out.append(f"dataset repo has {len(files)} files; first 60:\n" + "\n".join(files[:60]))
    wall = [f for f in files if "wall" in f.lower()]
    out.append(f"\nwall-like files ({len(wall)}):\n" + "\n".join(wall[:80]))
    if wall:
        snapshot_download(repo_id="facebook/jepa-wms", repo_type="dataset", local_dir=DSET, token=tok,
                          allow_patterns=[f for f in wall])
    out.append("\n=== extract ===\n" + ensure_wall_data())
    out.append("\n=== tree of JEPAWM_DSET ===\n" + _sh(f"find {DSET} -maxdepth 4 -not -path '*/.cache/*' | head -60; du -sh {DSET}"))
    out.append("\n=== where the repo looks for the Wall data ===\n" + _sh(
        "sed -n 40,90p app/plan_common/datasets/__init__.py; grep -n 'def get_dataset_paths' -A 25 src/datasets/utils/utils.py", 6000))
    out.append("\n=== grep Wall dataset class / paths ===\n" + _sh(
        "grep -rniE 'wall' app/plan_common/datasets/*.py evals/utils.py | head -60", 8000))
    for f in ["evals/utils.py", "evals/simu_env_planning/planning/gc_agent.py", "evals/simu_env_planning/envs/init.py",
              "evals/simu_env_planning/envs/wall_env/envs/wall.py", "app/plan_common/datasets/wall_dset.py"]:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            out.append(f"\n##### FILE {f} #####\n" + open(p, errors="replace").read()[:26000])
        else:
            out.append(f"\n##### MISSING {f}; find: " + _sh(f"find . -iname '*{os.path.basename(f).split('.')[0]}*' | head"))
    return "\n".join(out)


def make_cfg(model, episodes, n_samples, tag, iters=30):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, CFGS[model])))
    cfg["folder"] = f"{LOGS}/wall_runs/{model}"
    cfg["tag"] = tag
    cfg["meta"]["eval_episodes"] = episodes
    cfg["logging"]["optional_plots"] = False
    cfg["logging"]["tqdm_silent"] = True
    cfg["model_kwargs"]["checkpoint"] = URLS[model]
    pk = cfg["model_kwargs"]["pretrain_kwargs"]
    pk["heads_cfg"]["pretrain_dec_path"] = None
    cfg["planner"]["decode_each_iteration"] = False
    cfg["planner"]["num_samples"] = int(n_samples)
    cfg["planner"]["num_elites"] = min(cfg["planner"]["num_elites"], int(n_samples))
    cfg["planner"]["iterations"] = int(iters)
    os.makedirs("/tmp/cfgs", exist_ok=True)
    path = f"/tmp/cfgs/wall_{model}_{tag}.yaml"
    yaml.safe_dump(cfg, open(path, "w"))
    return path


def run_eval(model, episodes, n_samples, tag, iters=30, timeout=3 * 3600):
    import subprocess, time
    path = make_cfg(model, episodes, n_samples, tag, iters)
    t0 = time.time()
    p = subprocess.Popen(["python", "-m", "evals.main", "--fname", path, "--debug", "--devices", "cuda:0"],
                         cwd=ROOT, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for ln in p.stdout:
        lines.append(ln)
        if "success" in ln.lower() or "Error" in ln or "Traceback" in ln or "combined_results" in ln:
            print(f"[{model} N={n_samples}] {ln.rstrip()[:300]}", flush=True)
        if time.time() - t0 > timeout:
            p.kill()
            break
    p.wait()
    txt = "".join(lines)
    keep = [l for l in txt.splitlines() if "combined_results" in l or "episode_success" in l or "ep_end_dist" in l
            or "Traceback" in l or "Error" in l]
    return dict(model=model, N=n_samples, episodes=episodes, rc=p.returncode, secs=time.time() - t0,
                key="\n".join(keep[-25:]), tail="\n".join(txt.splitlines()[-40:]))


import os  # noqa: E402  (used inside make_cfg; kept below defs for the Modal image import order)


@app.function(image=image, gpu="L4", timeout=4 * 3600, volumes={"/results": vol},
              secrets=[modal.Secret.from_name(HF_SECRET)])
def wall(stage: str, model: str = "jepa", episodes: int = 96, grid: str = "30,100,300,1000,3000") -> str:
    os.makedirs(DSET, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    hf_token()  # fail fast with a clear message if the secret is not attached
    if stage == "data":
        txt = stage_data()
    elif stage in ("repro", "sweep"):
        have = _sh(f"find {DSET} -maxdepth 3 | head -5") + "\n" + ensure_wall_data()
        txt = f"dataset dir check:\n{have}\n"
        if "wall" not in have.lower():
            txt += "dataset missing -> downloading (stage data)\n" + stage_data()[:3000] + "\n"
        Ns = [300] if stage == "repro" else [int(x) for x in grid.split(",")]
        for n in Ns:
            r = run_eval(model, episodes, n, tag=f"{stage}_N{n}")
            txt += f"\n===== {model} N={n} episodes={episodes} rc={r['rc']} secs={r['secs']:.0f} =====\n{r['key']}\n--- tail ---\n{r['tail']}\n"
            print(f"done N={n}: rc={r['rc']} secs={r['secs']:.0f}", flush=True)
            vol.commit()
    else:
        txt = "unknown stage"
    vol.commit()
    return txt


@app.local_entrypoint()
def main(stage: str = "data", model: str = "jepa", episodes: int = 96, grid: str = "30,100,300,1000,3000"):
    os.makedirs("results", exist_ok=True)
    txt = wall.remote(stage, model, episodes, grid)
    path = f"results/wall_{stage}_{model}.txt"
    open(path, "w", encoding="utf-8").write(txt)
    print(txt[-6000:])
    print(f"\nsaved {path} ({len(txt)} chars)")