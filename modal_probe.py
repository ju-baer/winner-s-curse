"""
modal_probe.py -- Phase 2, step 0: look inside facebookresearch/jepa-wms on a Modal container.

It does NOT run the sweep. It (1) clones the repo into the image, (2) prints the install recipe,
repo tree, planner / objective / model-wrapper code, config names and checkpoint hints, and
(3) saves everything to the Volume (wcp-results/probe/) and to results/probe_*.txt locally.
Paste the printed output (or results/probe_summary.txt) back so the real sweep can be written
against the actual API instead of guesses.

Run:  modal run modal_probe.py        (CPU only, ~$0.05, a few minutes for the image build)
"""
import modal

app = modal.App("wcp-probe")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/jepa-wms /opt/jepa-wms",
        "cd /opt/jepa-wms && git rev-parse HEAD > /opt/jepa-wms/.commit",
    )
)
vol = modal.Volume.from_name("wcp-results", create_if_missing=True)

KEY_GLOBS = [
    "README.md", "pyproject.toml", "setup.py", "requirements*.txt", "environment*.yml", "Makefile",
    "evals/simu_env_planning/README.md",
    "evals/simu_env_planning/planning/planner.py",
    "evals/simu_env_planning/planning/objectives.py",
    "evals/simu_env_planning/planning/*.py",
    "app/vjepa_wm/modelcustom/simu_env_planning/*.py",
]


@app.function(image=image, timeout=1200, volumes={"/results": vol})
def probe() -> dict:
    import glob, os, subprocess

    root = "/opt/jepa-wms"
    out = []

    def sh(cmd, n=4000):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=root)
        return (r.stdout + r.stderr)[:n]

    out.append("COMMIT: " + open(f"{root}/.commit").read().strip())
    out.append("=== TREE (depth 3, dirs) ===\n" + sh("find . -maxdepth 3 -type d -not -path './.git*' | sort", 6000))
    out.append("=== CONFIGS (eval) ===\n" + sh("find configs -type f | head -150", 6000))
    out.append("=== grep: checkpoints / model zoo / hub ===\n" + sh(
        "grep -rniE 'checkpoint|torch.hub|huggingface|hf_hub|\\.pth|\\.pt\\b' README.md evals/simu_env_planning/README.md | head -60", 6000))
    out.append("=== grep: planner entry points ===\n" + sh(
        "grep -rnE 'def |class ' evals/simu_env_planning/planning/planner.py evals/simu_env_planning/planning/objectives.py | head -120", 8000))
    out.append("=== grep: env step / success / eval_episodes ===\n" + sh(
        "grep -rnE 'eval_episodes|def step|success|rollout|decode_unroll|frameskip|action_repeat|plan_horizon|planning_horizon' evals/simu_env_planning --include=*.py | head -120", 10000))

    seen = set()
    for pat in KEY_GLOBS:
        for f in sorted(glob.glob(os.path.join(root, pat))):
            if f in seen or not os.path.isfile(f):
                continue
            seen.add(f)
            txt = open(f, errors="replace").read()
            out.append(f"\n##### FILE {os.path.relpath(f, root)} ({len(txt)} chars) #####\n" + txt[:20000])

    text = "\n".join(out)
    os.makedirs("/results/probe", exist_ok=True)
    open("/results/probe/probe_summary.txt", "w", encoding="utf-8").write(text)
    vol.commit()
    return dict(text=text)


@app.local_entrypoint()
def main():
    import os
    os.makedirs("results", exist_ok=True)
    r = probe.remote()
    path = "results/probe_summary.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(r["text"])
    print(r["text"][:6000])
    print(f"\n... full text ({len(r['text'])} chars) saved to {path}")