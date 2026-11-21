"""
modal_count.py -- count success/fail episodes from a finished or killed repo eval, straight from the Modal volume.
The repo saves video_agent_goal_succ.* or video_agent_goal_fail.* per episode, so counting those files gives the
partial success rate without re-running anything (CPU only, a few cents).

    modal run modal_count.py --run repro_N300 --model jepa
"""
import modal

app = modal.App("wcp-count")
vol = modal.Volume.from_name("wcp-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.10")


@app.function(image=image, volumes={"/results": vol}, timeout=600)
def count(run: str, model: str) -> str:
    import glob, os, re
    base = f"/results/jepawm_logs/wall_runs/{model}/simu_env_planning/{run}/wall-base"
    succ = fail = 0
    rows = []
    for d in sorted(glob.glob(f"{base}/ep_*"), key=lambda p: int(re.findall(r"ep_(\d+)", p)[0])):
        s = glob.glob(f"{d}/visualisation/video_agent_goal_succ.gif")
        f = glob.glob(f"{d}/visualisation/video_agent_goal_fail.gif")
        if s:
            succ += 1; rows.append((os.path.basename(d), 1))
        elif f:
            fail += 1; rows.append((os.path.basename(d), 0))
    n = succ + fail
    import math
    p = succ / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if n else float("nan")
    # Wilson 95% interval
    z = 1.96
    if n:
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        ci = f"[{c-h:.3f}, {c+h:.3f}]"
    else:
        ci = "n/a"
    return (f"run={run} model={model}: {succ} success / {fail} fail over {n} finished episodes -> "
            f"success rate {p:.3f} (Wilson 95% CI {ci})\nper-episode: " + " ".join(f"{a}:{b}" for a, b in rows))


@app.local_entrypoint()
def main(run: str = "repro_N300", model: str = "jepa"):
    print(count.remote(run, model))