"""
criteres_communs.py : same metrics for every method, on the same environment.

Each final plan is replayed in FerroMobileEnv, then measured on the 1,752 points outside tunnels:
  q_mean, q_min : a white point counts as quality 0 (no service)
  N_H, N_O, N_T : transitions of the optimal cell sequence (same DP as the RL)
  N_cut         : connection losses = number of entries into a white zone
  N_H_cut       : N_H + N_cut (each loss forces a new attachment)
Random policy: same 5 seeds as baseline_aleatoire.py.

Run: python criteres_communs.py   (needs the DEM and crc-covlib, like the RL)
Output: resultats/criteres_communs.csv
"""
import ast
import os
import sys

import numpy as np
import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, os.path.join(RACINE, "placement"))
os.environ.setdefault("CRC_COVLIB_PATH", os.path.join(RACINE, "crc-covlib", "python-wrapper"))

from environment_rl import FerroMobileEnv, STOP   # noqa: E402

PLANS = os.path.join(ICI, "plans_gapfix")
env = FerroMobileEnv(scenario="S1", budget=10_000_000.0, t_max=150, sauvegarder_deploiements_rl=False)


def mesurer(nom, total, n_steps):
    dp = env._dernier_resultat_dp
    points, cells = env._fusionner_cells_par_point()
    qos_dp = {}
    for pid, cand, cel in zip(points, cells, dp["trajectory"]):
        qos_dp[pid] = next(c["qos"] for c in cand if c["cellule"] == cel)
    blancs = set(env._points_blancs)
    hors = sorted(env.points_hors_tunnel)
    q = np.array([0.0 if (p in blancs or p not in qos_dp) else qos_dp[p] for p in hors])
    est_blanc = [p in blancs for p in hors]
    n_cut = sum(1 for i, b in enumerate(est_blanc) if b and (i == 0 or not est_blanc[i - 1]))
    return {"methode": nom, "n_white": len(blancs), "q_mean": q.mean(), "q_min": q.min(),
            "N_H": dp["N_H"], "N_O": dp["N_O"], "N_T": dp["N_T"], "N_cut": n_cut,
            "N_H_cut": dp["N_H"] + n_cut, "cout": np.nan, "reward_total": total,
            "return_ref": np.nan,
            "n_steps": n_steps}


def rejouer(nom, plan):
    env.reset()
    total, n, done, info = 0.0, 0, False, {}
    for a in plan:
        if done:
            break
        _, r, done, info = env.step(a)
        total += r; n += 1
    if not done:
        _, r, done, info = env.step(STOP)
        total += r
    res = mesurer(nom, total, n)
    res["cout"] = info.get("cout_total", 0.0)
    res["return_ref"] = env.U_reference() - env.U0_reference + info.get("r_fin", 0.0)
    print(res, flush=True)
    return res


def plan_csv(fichier):
    df = pd.read_csv(os.path.join(PLANS, fichier))
    return [(int(r.ix), int(r.iy), r.generation, int(r.bande_mhz)) for r in df.itertuples()]


def plan_trace(fichier):
    df = pd.read_csv(os.path.join(PLANS, fichier))
    acts = [a if a == STOP else ast.literal_eval(a) for a in df.action]
    return [a for a in acts if a != STOP]


def plan_aleatoire(graine):
    rng, plan = np.random.default_rng(graine), []
    env.reset()
    done = False
    while not done:
        faisables = np.nonzero(env._actions_faisables())[0]
        a = env.actions[int(rng.choice(faisables))]
        _, _, done, _ = env.step(a)
        if a != STOP:
            plan.append(a)
    return plan


lignes = [rejouer("Existing network", [])]
for g in range(5):
    lignes.append(rejouer(f"Random seed {g}", plan_aleatoire(g)))
lignes += [rejouer("Coverage-Greedy", plan_csv("coverage_greedy.csv")),
           rejouer("Trajectory-Greedy", plan_csv("trajectory_greedy.csv")),
           rejouer("ILP", plan_csv("ilp.csv")),
           rejouer("Dueling DQN ep725", plan_trace("dueling_ep725.csv")),
           rejouer("Dueling DQN ep1000", plan_trace("dueling_ep1000.csv"))]

os.makedirs(os.path.join(RACINE, "resultats"), exist_ok=True)
df = pd.DataFrame(lignes)
df.to_csv(os.path.join(RACINE, "resultats", "criteres_communs.csv"), index=False)
print(df.to_string())
