"""
etat_initial_reel.py : état initial réel I_0 (réseau existant, S1), tableau 4.3.

Pas besoin du DEM ni de crc-covlib.

Lancer : python etat_initial_reel.py <racine_repo>
"""
import json, sys, os
import pandas as pd
racine = sys.argv[1]
sys.path.insert(0, os.path.join(racine, "placement"))
from trajectory_dp import trajectory_dp
from utility import normaliser_metriques, compute_U, compute_r_fin

D_COV, LAMBDA = 1.0, 0.05
df_base = pd.read_csv(os.path.join(racine, "data/processed/dataset_base.csv"), low_memory=False)
df_e = pd.read_csv(os.path.join(racine, "data/processed/dataset_enrichi.csv"), low_memory=False)
pts = df_base[["point_id", "in_tunnel"]].drop_duplicates("point_id")
hors_tunnel = set(pts[~pts.in_tunnel].point_id)
n_total = len(hors_tunnel)

res = {}
for sc in ["S1", "S2", "S3", "S4"]:
    s = df_e[df_e.scenario_id == sc]
    reel = s[(~s.in_tunnel) & (s.ant_id.notna())]
    cells = {}
    for pid, g in reel[["point_id", "ant_lat", "ant_lon", "operateur", "generation", "bande_mhz", "qos", "debit_adj_mbps"]].groupby("point_id"):
        cells[pid] = [{"cellule": ((round(r.ant_lat, 6), round(r.ant_lon, 6)), r.operateur, r.generation, r.bande_mhz),
                       "qos": r.qos} for r in g.itertuples()]
    points = sorted(cells)
    dp = trajectory_dp(points, [cells[p] for p in points], LAMBDA, LAMBDA, LAMBDA)
    debit = s[~s.in_tunnel].groupby("point_id").debit_adj_mbps.max().reindex(list(hors_tunnel)).fillna(0)
    n_white = int((debit < D_COV).sum())
    m = normaliser_metriques(dp, len(points), n_white, n_total)
    U0 = compute_U(m, 1, 1, 1, 1, 1)
    r_fin0 = compute_r_fin(n_white=n_white, n_total=n_total, qos_min=dp["qos_min"], cost_total=0, budget=10e6,
                           Q_critique=0.5, R_couverture=1.0, R_qualite=0.5, R_succes=0.2, R_eff=0.2)
    # points blancs sans aucun candidat (n'entrent pas dans N_eval)
    blancs = set(debit[debit < D_COV].index)
    blancs_sans_candidat = len(blancs - set(points))
    # QoS par génération sur la trajectoire DP
    traj_gen = pd.Series([c[2] for c in dp["trajectory"]]).value_counts().to_dict()
    res[sc] = {"n_total_hors_tunnel": n_total, "n_eval": len(points), "n_white": n_white,
               "n_white_sans_candidat": blancs_sans_candidat,
               "qos_mean": round(dp["qos_mean"], 4), "qos_min": round(dp["qos_min"], 4),
               "N_H": dp["N_H"], "N_O": dp["N_O"], "N_T": dp["N_T"],
               "Q_hat": round(m["Q"], 4), "W_hat": round(m["W"], 4), "H_hat": round(m["H"], 4),
               "O_hat": round(m["O"], 4), "T_hat": round(m["T"], 4), "U0": round(U0, 4),
               "r_fin_si_STOP_immediat": round(r_fin0, 4), "generations_trajectoire_dp": traj_gen,
               "n_antennes_reelles": int(reel.ant_id.nunique())}
    print(sc, json.dumps(res[sc]))
_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resultats")
os.makedirs(_out, exist_ok=True)
json.dump(res, open(os.path.join(_out, "etat_initial_reel.json"), "w"), indent=2)
