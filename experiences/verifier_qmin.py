"""
verifier_qmin.py : where does q_min come from, for the Dueling (ep 725) and ILP plans.

For the 10 weakest points of the optimal cell sequence it prints the chosen cell, its QoS,
and the best QoS available at that point (without transition costs).
Run: python verifier_qmin.py   (needs the DEM and crc-covlib)
"""
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import criteres_communs as cc   # noqa: E402  (also replays every method once)

env = cc.env
for nom, plan in [("Dueling DQN ep725", cc.plan_trace("dueling_ep725.csv")),
                  ("ILP", cc.plan_csv("ilp.csv"))]:
    cc.rejouer(nom, plan)
    dp = env._dernier_resultat_dp
    points, cells = env._fusionner_cells_par_point()
    lignes = []
    for pid, cand, cel in zip(points, cells, dp["trajectory"]):
        q = next(c["qos"] for c in cand if c["cellule"] == cel)
        meilleur = max(cand, key=lambda c: c["qos"])
        lignes.append((q, pid, cel, meilleur["qos"], meilleur["cellule"], len(cand)))
    lignes.sort(key=lambda x: x[0])
    print(f"\n=== {nom}: 10 weakest points of the optimal sequence ===")
    for q, pid, cel, qb, cb, n in lignes[:10]:
        print(f"point {pid}: chosen {cel} q={q:.3f} | best available {cb} q={qb:.3f} | {n} candidates")
    n_moins = sum(1 for l in lignes if l[0] < 0.6)
    print(f"points with q < 0.6: {n_moins}; points where a better cell exists: "
          f"{sum(1 for l in lignes if l[3] > l[0] + 1e-9)}")
