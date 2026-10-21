"""
export_apres_deploiement.py : simulated links of a final plan, for the throughput figure.

Replays the Dueling DQN (ep 725) and ILP plans and writes, for every new link,
the point, the generation and the throughput (debit_adj_mbps).
Run: python export_apres_deploiement.py   (needs the DEM and crc-covlib)
Output: resultats/liens_<plan>.csv
"""
import os
import sys

import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, os.path.join(RACINE, "placement"))
os.environ.setdefault("CRC_COVLIB_PATH", os.path.join(RACINE, "crc-covlib", "python-wrapper"))
import ast  # noqa: E402
from environment_rl import FerroMobileEnv, STOP  # noqa: E402

env = FerroMobileEnv(scenario="S1", budget=10_000_000.0, t_max=150, sauvegarder_deploiements_rl=False)
plans = {
    "dueling_ep725": [ast.literal_eval(a) for a in pd.read_csv(os.path.join(ICI, "plans_gapfix", "dueling_ep725.csv")).action if a != STOP],
    "ilp": [(int(r.ix), int(r.iy), r.generation, int(r.bande_mhz))
            for r in pd.read_csv(os.path.join(ICI, "plans_gapfix", "ilp.csv")).itertuples()],
}
os.makedirs(os.path.join(RACINE, "resultats"), exist_ok=True)
for nom, plan in plans.items():
    env.reset()
    for a in plan:
        env.step(a)
    liens = pd.DataFrame([{"point_id": r["point_id"], "generation": r["cellule"][2],
                           "debit_adj_mbps": r["debit_adj_mbps"]} for r in env._resultats_hypo])
    liens.to_csv(os.path.join(RACINE, "resultats", f"liens_{nom}.csv"), index=False)
    print(nom, len(liens), "links, white left:", len(env._points_blancs))
