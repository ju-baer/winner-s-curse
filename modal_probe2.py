"""
modal_probe2.py -- Phase 2, step 0b. Targeted: correct paths + try to install and load the Wall models on a GPU.

Run:  modal run modal_probe2.py     (one L4, ~10 min, ~$0.15)
Output: results/probe2_key.txt (small, attach this) and results/probe2_install.txt
"""
import modal

app = modal.App("wcp-probe2")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0", "libegl1", "libosmesa6-dev")
    .run_commands("git clone --depth 1 https://github.com/facebookresearch/jepa-wms /opt/jepa-wms")
)
vol = modal.Volume.from_name("wcp-results", create_if_missing=True)

FILES = [
    "evals/simu_env_planning/planning/planning/planner.py",
    "evals/simu_env_planning/planning/planning/objectives.py",
    "evals/simu_env_planning/planning/plan_evaluator.py",
    "evals/simu_env_planning/envs/wrappers/pixels.py",
    "evals/simu_env_planning/envs/wall_gym_wrap.py",
    "evals/simu_env_planning/eval.py",
    "configs/evals/simu_env_planning/wall/jepa-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
    "hubconf.py",
]


@app.function(image=image, gpu="L4", timeout=1800, volumes={"/results": vol})
def probe2() -> dict:
    import os, subprocess

    root = "/opt/jepa-wms"

    def sh(cmd, n=6000, t=900):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=root, timeout=t)
            return (r.stdout + r.stderr)[-n:]
        except subprocess.TimeoutExpired:
            return "TIMEOUT: " + cmd

    key = []
    key.append("=== install-related files ===\n" + sh("ls; ls requirements* environment* pyproject.toml setup.py 2>&1; find . -name 'requirements*' -o -name 'environment*.yml' | head"))
    key.append("=== README install section ===\n" + sh("grep -n -iE 'install|conda|uv |pip |env var|JEPAWM|mujoco|ffmpeg' README.md | head -60"))
    key.append("=== evals main / entry ===\n" + sh("ls evals; sed -n 1,80p evals/main.py"))
    key.append("=== wall data paths in config / how dataset is located ===\n" + sh("grep -rnE 'JEPAWM_DSET|JEPAWM_HOME|JEPAWM_LOGS|data_dir|dset_path|val_data' configs/evals/simu_env_planning/wall evals/simu_env_planning/eval.py app/plan_common/datasets/*.py | head -40"))
    for f in FILES:
        p = os.path.join(root, f)
        if os.path.exists(p):
            txt = open(p, errors="replace").read()
            key.append(f"\n##### FILE {f} ({len(txt)} chars) #####\n" + txt[:30000])
        else:
            key.append(f"\n##### MISSING {f} #####\n" + sh(f"find . -name '{os.path.basename(f)}' | head"))

    inst = []
    inst.append("=== pip install torch stack ===\n" + sh("pip install -q torch torchvision einops tensordict pyyaml omegaconf huggingface_hub 2>&1 | tail -5"))
    inst.append("=== try repo requirements / editable install ===\n" + sh(
        "(test -f requirements.txt && pip install -q -r requirements.txt 2>&1 | tail -15) ; (pip install -q -e . 2>&1 | tail -15)", t=1200))
    inst.append("=== import check ===\n" + sh("python -c \"import app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds as m; print('import ok')\" 2>&1 | tail -15"))
    inst.append("=== torch.hub load wall models ===\n" + sh(
        "python - <<'EOF'\nimport torch\nfor n in ['jepa_wm_wall','dino_wm_wall']:\n    try:\n        m,p=torch.hub.load('/opt/jepa-wms',n,source='local')\n        print(n,'loaded',type(m).__name__, sum(x.numel() for x in m.parameters())/1e6,'M params')\n    except Exception as e:\n        import traceback; traceback.print_exc(); print(n,'FAILED',repr(e)[:400])\nEOF", n=8000, t=1200))

    k, i = "\n".join(key), "\n".join(inst)
    os.makedirs("/results/probe", exist_ok=True)
    open("/results/probe/probe2_key.txt", "w", encoding="utf-8").write(k)
    open("/results/probe/probe2_install.txt", "w", encoding="utf-8").write(i)
    vol.commit()
    return dict(key=k, install=i)


@app.local_entrypoint()
def main():
    import os
    os.makedirs("results", exist_ok=True)
    r = probe2.remote()
    open("results/probe2_key.txt", "w", encoding="utf-8").write(r["key"])
    open("results/probe2_install.txt", "w", encoding="utf-8").write(r["install"])
    print(r["install"][-5000:])
    print(f"\nsaved results/probe2_key.txt ({len(r['key'])} chars) and results/probe2_install.txt")