# Winner's Curse of Planning: pilot code (ICWM 2027 plan)

Open `winners_curse_pilot.ipynb` in VS Code (Python kernel, any Python 3.10+ with numpy, scipy, pandas, scikit-learn, matplotlib).

| File | Purpose |
|---|---|
| `winners_curse_pilot.ipynb` | Main notebook: theory (A), CPU pilot (B), Modal GPU launch (C), decision gates (D) |
| `wcp_core.py` | Environment, ensemble training, planners, cross-fit estimator, theory simulation, stats helpers |
| `wcp_plots.py` | Figures, planner table, automatic gate evaluation |
| `run_pilot.py` | Command-line runner for one ensemble (what the notebook calls) |
| `modal_app.py` | GPU version for Modal (torch). Untested: run `--smoke` first |
| `results/` | CSVs from the CPU pilot (included) and later GPU runs |
| `figures/` | PNG figures |

## Modal setup (once)
1. `pip install modal`
2. `modal setup` (opens a browser to sign in)
3. In the notebook, run section C cells in order. The smoke run prints its approximate cost; check it before `RUN_FULL = True`.
4. Set a spending limit in the Modal dashboard before the full run.

## Status
* Verified by running here: theory simulation, CPU pilot (E1, E3, E4), analysis code.
* NOT verified: `modal_app.py` (torch and Modal were unavailable in the sandbox). The smoke run is the test.
* Not implemented: stage 2 (released DINO-WM / LeWM / TD-MPC2 checkpoints).
