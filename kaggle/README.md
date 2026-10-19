# Training on Kaggle without losing anything

This folder holds the notebook `ferromobile_train.ipynb`. Upload it to Kaggle (Create > New Notebook > File > Import Notebook) once per run you want to launch.

## How the durable save works

1. You start the notebook with **Save Version > Save & Run All (Commit)**. A committed run executes on Kaggle's servers. You can close the browser, lose the wifi or switch the computer off: the run continues.
2. Every 25 episodes, `train_dqn.py` writes a checkpoint (both networks, optimizer, epsilon, step counter, random states, replay buffer) and the logs, then sends the whole run folder as a new version of a **private** Kaggle dataset named `fm-<run name>` (for example `fm-dueling-softmax-ww5-seed0`).
3. Kaggle stops a session after 12 hours. The script stops itself cleanly before that (`FM_MAX_HOURS`, 11 h by default), saves a checkpoint and sends it. The commit ends normally.
4. To continue, run **Save & Run All (Commit)** again with the same settings. The script downloads the latest dataset version and resumes at the saved episode. A resumed run gives exactly the same result as an uninterrupted one (checked by `placement/test_reprise_exacte.py`).

What can be lost: if Kaggle kills a session unexpectedly (rare), you lose at most the episodes since the last checkpoint (25 episodes or less). They are replayed identically on the next run, and their log lines are rewritten, so no episode appears twice.

Files in each dataset version:

| File | Content |
|---|---|
| `train_log.csv` | one line per training episode |
| `eval_log.csv` | one line per greedy evaluation (every 25 episodes), with `return_ref` (return with the reference weights, comparable to Table 4.8) and `temps_decision_s` |
| `eval_trace_epN.csv` | every action of evaluation N |
| `console.log` | the full console output of every session |
| `run_etat.json` | last saved episode, `termine: true` when the run is finished |
| `dernier.pt`, `checkpoint_epN.pt`, `buffer.npz` | checkpoints |

## One-time setup

1. In the notebook: Settings > Accelerator **GPU T4**, Internet **on**.
2. Add-ons > Secrets: create `KAGGLE_USERNAME` (your user name) and `KAGGLE_KEY` (kaggle.com > Settings > API > Create New Token, then copy the `key` value of `kaggle.json`). Attach both secrets to the notebook.
3. Add Input: your DEM dataset (`ferromobile-dem`).
4. Edit cell 1 (run settings) only.

Without the secrets the notebook still trains, but prints a warning: checkpoints then stay in `/kaggle/working` and are lost if the session stops before the end.

## Runs needed for the thesis

Each line is one notebook (copy the notebook and change cell 1). Two notebooks with the same settings must not run at the same time: they would write to the same dataset.

| Priority | Settings (cell 1) | Thesis values it replaces |
|---|---|---|
| 1 | `FM_EPISODES=1000, FM_W_W=5, FM_ARCH=dueling, FM_SEED=0` | Fig. 4.6 (step 4), Fig. 4.7, Tables 4.7, 4.8, 4.9, 4.10, Figs. 4.8 to 4.12, abstract |
| 2 | `FM_EPISODES=1000, FM_W_W=5, FM_ARCH=vanilla, FM_SEED=0` | Vanilla rows of Tables 4.7 to 4.10, Section 4.9 |
| 3 | `FM_EPISODES=1000, FM_W_W=5, FM_ARCH=dueling, FM_SEED=1` | Table 4.7, second seed |
| 4 | `FM_EPISODES=300, FM_W_W=1, FM_ARCH=vanilla` | Table 4.13, Vanilla row |
| 5 | `FM_EPISODES=300, FM_W_W=1, FM_EXPLORATION=uniforme` | Table 4.13, uniform row |
| 6 | `FM_EPISODES=300, FM_W_W=1, FM_FOLD=1` (then 2, 3, 4) | Table 4.12, level 1 |

Budget: at the measured 115 s per episode, a 1000-episode run needs about 32 GPU hours, which is more than Kaggle's weekly GPU quota (about 30 hours). The corrected environment is faster (vectorized action mask, cached radio simulations), so check `duree_s` in `train_log.csv` after the first session and plan with the real figure. Report the new episode time and GPU hours in Sections 4.2.2 and 4.6.6.

## Evaluations (no GPU needed)

Download the checkpoint from the run's dataset (or attach the dataset as an input), then from `placement/`:

```
python evaluate_policy.py <run>/dernier.pt final petite        # Table 4.8 row, Table 4.11 (after experiences/small_instance.py)
python evaluate_policy.py <wW1 run>/dernier.pt budget          # Table 4.14 (b_t = B / B_train)
python evaluate_policy.py <wW1 run>/dernier.pt operateurs meteo # Table 4.12, level 2 and weather
python evaluate_policy.py <fold k run>/dernier.pt pli --pli k  # Table 4.12, level 1, with Coverage-Greedy on the same block
```

Results are appended to `resultats/evaluation_<run>.csv`.

## Scripts for the other thesis results

| Thesis | Script |
|---|---|
| Table 4.3 | `experiences/etat_initial_reel.py` |
| Fig. 4.2, Table 4.12 "before" column | `experiences/real_extras.py` |
| Fig. 4.5, Table 4.6 | `experiences/analyse_recompense_reelle.py` |
| Table 4.5 (and Trajectory-Greedy with w_W = 5) | `experiences/tg_variant.py` |
| Table 4.8: random, Coverage-Greedy, Trajectory-Greedy | `placement/baseline_aleatoire.py`; `placement/diagnostiquer_budget.py` (version used for the thesis) or `placement/baseline_couverture_gloutonne.py` (same rule, played in the environment); `placement/baseline_trajectoire_gloutonne.py` |
| Table 4.8 ILP row, Fig. 4.12, Table 4.15 | `experiences/ilp_reference.py`, `experiences/ilp_criteria.py` |
| Table 4.11 | `experiences/small_instance.py`, then `evaluate_policy.py ... petite` |
| Table 4.2 | `placement/test_*.py` |
