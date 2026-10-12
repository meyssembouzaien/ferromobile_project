"""
ilp_criteria.py : minimum cost to leave at most k white zones (k = 20, 10, 5, 0), solved exactly
on the cached coverage of ilp_reference.py (same candidates, same cost model).
    min  sum C_radio x + sum C_site y
    s.t. x_a <= y_c ; z_p <= sum_{a covers p} x_a ; sum z_p >= N_white0 - k ; binary.
Usage: python ilp_criteria.py <repo_root> [--k 20,10,5,0] [--temps_max_s 900]
Output: <repo_root>/resultats/ilp_criteres.csv
"""
import argparse, os, pickle, sys
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

ap = argparse.ArgumentParser(); ap.add_argument("racine"); ap.add_argument("--k", default="20,10,5,0")
ap.add_argument("--temps_max_s", type=int, default=900); a = ap.parse_args()
sys.path.insert(0, os.path.join(a.racine, "placement"))
from cost_model import get_radio_cost, C_SITE_EUR
from grid_config import construire_grille_depuis_dataset

OUT = os.path.join(a.racine, "resultats")
cache = pickle.load(open(os.path.join(OUT, "ilp_couverture_cache.pkl"), "rb"))
actions = sorted(cache)
b = pd.read_csv(os.path.join(a.racine, "data/processed/dataset_base.csv"), low_memory=False)
e = pd.read_csv(os.path.join(a.racine, "data/processed/dataset_enrichi.csv"), low_memory=False)
s = e[(e.scenario_id == "S1") & (~e.in_tunnel)]
g = construire_grille_depuis_dataset(b)
exist = set()
for r in s[s.ant_id.notna()][["ant_lat", "ant_lon"]].drop_duplicates().itertuples():
    c = g.gps_vers_cellule(r.ant_lat, r.ant_lon)
    if c is not None:
        exist.add(c)

N0 = 332
pts = sorted(set().union(*cache.values())); ip = {p: i for i, p in enumerate(pts)}
nouv = sorted({(x[0], x[1]) for x in actions} - exist); iy = {c: i for i, c in enumerate(nouv)}
nA, nY, nZ = len(actions), len(nouv), len(pts); n = nA + nY + nZ
cost = np.r_[[get_radio_cost(x[2], x[3]) for x in actions], [C_SITE_EUR] * nY, [0.0] * nZ]

rows, cols, vals, lo, hi, r = [], [], [], [], [], 0
for j, x in enumerate(actions):                       # x_a <= y_c
    c = (x[0], x[1])
    if c in iy:
        rows += [r, r]; cols += [j, nA + iy[c]]; vals += [1, -1]; lo.append(-np.inf); hi.append(0); r += 1
cov = {p: [] for p in pts}
for j, x in enumerate(actions):
    for p in cache[x]:
        cov[p].append(j)
for p in pts:                                         # z_p <= sum x
    rows += [r] * (len(cov[p]) + 1); cols += cov[p] + [nA + nY + ip[p]]; vals += [-1] * len(cov[p]) + [1]
    lo.append(-np.inf); hi.append(0); r += 1

lignes = []
for k in [int(v) for v in a.k.split(",")]:
    A = sparse.csr_matrix((vals + [1] * nZ, (rows + [r] * nZ, cols + list(range(nA + nY, n)))), shape=(r + 1, n))
    res = milp(cost, constraints=LinearConstraint(A, lo + [N0 - k], hi + [np.inf]), integrality=np.ones(n),
               bounds=Bounds(0, 1), options={"time_limit": a.temps_max_s})
    ch = [x for j, x in enumerate(actions) if res.x is not None and res.x[j] > 0.5]
    lignes.append({"k_max_white": k, "statut": "Optimal" if res.status == 0 else str(res.message),
                   "cout_eur": float(cost @ np.round(res.x)) if res.x is not None else None,
                   "borne_inf_eur": getattr(res, "mip_dual_bound", None),
                   "deploiements": len(ch), "nouveaux_sites": len({(x[0], x[1]) for x in ch} - exist)})
    print(lignes[-1], flush=True)
    pd.DataFrame(lignes).to_csv(os.path.join(OUT, "ilp_criteres.csv"), index=False)
