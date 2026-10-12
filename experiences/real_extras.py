"""
real_extras.py : quantités exactes sans DEM (S1).

 1. DP du réseau existant pour plusieurs lambda (figure de l'Expérience D)
 2. zones blanches sans un opérateur (colonne "avant" du tableau 4.12)
 3. segments blancs dont la couverture donne ΔU < 0 selon la récompense

Lancer : python real_extras.py <racine_repo>
"""
import json, os, sys
import pandas as pd
root = sys.argv[1]; sys.path.insert(0, os.path.join(root, "placement"))
from trajectory_dp import trajectory_dp
from utility import normaliser_metriques, compute_U
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resultats")
os.makedirs(out, exist_ok=True)
b = pd.read_csv(os.path.join(root, "data/processed/dataset_base.csv"), low_memory=False)
e = pd.read_csv(os.path.join(root, "data/processed/dataset_enrichi.csv"), low_memory=False)
pts = b[["point_id", "in_tunnel"]].drop_duplicates("point_id").sort_values("point_id")
ht = pts[~pts.in_tunnel].point_id.tolist(); n_total = len(ht)
s = e[e.scenario_id == "S1"]; sh = s[~s.in_tunnel]
reel = sh[sh.ant_id.notna()]
def cells_of(df):
    return {pid: [{"cellule": ((round(r.ant_lat, 6), round(r.ant_lon, 6)), r.operateur, r.generation, r.bande_mhz), "qos": r.qos}
                  for r in g.itertuples()] for pid, g in df.groupby("point_id")}
cells = cells_of(reel)
res = {}
# 1. Balayage de lambda
rows = []
P = sorted(cells)
for lam in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
    dp = trajectory_dp(P, [cells[p] for p in P], lam, lam, lam)
    rows.append({"lambda": lam, "N_H": dp["N_H"], "N_O": dp["N_O"], "N_T": dp["N_T"], "q_mean": round(dp["qos_mean"], 4)})
    print(rows[-1])
pd.DataFrame(rows).to_csv(os.path.join(out, "real_lambda_sweep.csv"), index=False)
# 2. Retrait d'un opérateur
ops = {}
for op in sorted(reel.operateur.dropna().unique()):
    d = sh[sh.operateur != op]
    deb = d.groupby("point_id").debit_adj_mbps.max().reindex(ht).fillna(0)
    ops[op] = int((deb < 1.0).sum())
print("white zones without operator:", ops); res["white_without_operator"] = ops
# 3. Variantes de récompense : chaque segment blanc couvert par une cellule 4G
deb = sh.groupby("point_id").debit_adj_mbps.max().reindex(ht).fillna(0); blanc = set(deb[deb < 1].index)
segs, cur = [], []
for p in ht:
    if p in blanc: cur.append(p)
    elif cur: segs.append(cur); cur = []
if cur: segs.append(cur)
def U(extra, wW=1.0, op_hypo=None):
    c = {p: list(v) for p, v in cells.items()}
    for p, cell in extra.items(): c.setdefault(p, []).append(cell)
    pp = sorted(c); dp = trajectory_dp(pp, [c[p] for p in pp], 0.05, 0.05, 0.05)
    nw = sum(1 for p in blanc if p not in extra)
    m = normaliser_metriques(dp, len(pp), nw, n_total)
    return compute_U(m, 1, wW, 1, 1, 1)
for name, wW, op in [("reference", 1.0, None), ("w_W_doubled", 2.0, None), ("fifth_operator", 1.0, "HYPO")]:
    for g, q in [("3G", 0.655), ("4G", 0.847), ("5G", 0.911)]:
        U0 = U({}, wW); neg, negpts = 0, 0
        for i, seg in enumerate(segs):
            cell = {"cellule": (("h", i), op, g, 900), "qos": q}
            if U({p: cell for p in seg}, wW) - U0 < 0: neg += 1; negpts += len(seg)
        res[f"neg_segments_{name}_{g}"] = [neg, negpts]; print(name, g, neg, negpts)
json.dump(res, open(os.path.join(out, "real_extras.json"), "w"), indent=1)
