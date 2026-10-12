"""Analyse EXACTE de la récompense par étape r_t = ΔU sur les données réelles (S1),
pour une cellule hypothétique idéalisée qui couvre exactement un segment blanc.
Hypothèse explicite : QoS de la cellule ajoutée = QoS médiane observée des liens couvrants
réels de la même génération (3G 0.655, 4G 0.847, 5G 0.911). Aucune donnée DEM nécessaire.
Usage : python analyse_recompense_reelle.py <racine_repo>
"""
import json, os, sys
import pandas as pd
racine = sys.argv[1]
sys.path.insert(0, os.path.join(racine, "placement"))
from trajectory_dp import trajectory_dp
from utility import normaliser_metriques, compute_U

L = 0.05
QOS = {"3G": 0.655, "4G": 0.847, "5G": 0.911}
b = pd.read_csv(os.path.join(racine, "data/processed/dataset_base.csv"), low_memory=False)
e = pd.read_csv(os.path.join(racine, "data/processed/dataset_enrichi.csv"), low_memory=False)
pts = b[["point_id", "in_tunnel"]].drop_duplicates("point_id").sort_values("point_id")
ht = pts[~pts.in_tunnel].point_id.tolist()
n_total = len(ht)
s = e[e.scenario_id == "S1"]
reel = s[(~s.in_tunnel) & (s.ant_id.notna())]
cells = {pid: [{"cellule": ((round(r.ant_lat, 6), round(r.ant_lon, 6)), r.operateur, r.generation, r.bande_mhz), "qos": r.qos}
               for r in g.itertuples()] for pid, g in reel.groupby("point_id")}
debit = s[~s.in_tunnel].groupby("point_id").debit_adj_mbps.max().reindex(ht).fillna(0)
blanc = set(debit[debit < 1.0].index)

def U(extra):
    c = {p: list(v) for p, v in cells.items()}
    for p, cell in extra.items():
        c.setdefault(p, []).append(cell)
    points = sorted(c)
    dp = trajectory_dp(points, [c[p] for p in points], L, L, L)
    nw = sum(1 for p in blanc if p not in extra)
    m = normaliser_metriques(dp, len(points), nw, n_total)
    return compute_U(m, 1, 1, 1, 1, 1), m, dp

U0, m0, _ = U({})
# segments blancs contigus le long des points hors tunnel
segs, cur = [], []
for p in ht:
    if p in blanc:
        cur.append(p)
    elif cur:
        segs.append(cur); cur = []
if cur: segs.append(cur)
print("U0", round(U0, 5), "segments blancs", len(segs), "tailles", sorted(len(x) for x in segs))
rows = []
for i, seg in enumerate(segs):
    for g, q in QOS.items():
        cell = {"cellule": (("hypo", i), None, g, 900), "qos": q}
        u, m, dp = U({p: cell for p in seg})
        rows.append({"segment": i, "taille": len(seg), "generation": g, "dU": round(u - U0, 6),
                     "dW": round(m0["W"] - m["W"], 6), "dQ": round(m["Q"] - m0["Q"], 6),
                     "dH": round(m["H"] - m0["H"], 6), "dT": round(m["T"] - m0["T"], 6), "dO": round(m["O"] - m0["O"], 6)})
df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resultats")
os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, "analyse_dU_segments_blancs.csv"), index=False)
piv = df.pivot(index=["segment", "taille"], columns="generation", values="dU")
print(piv.to_string())
# couverture totale par génération
res = {"U0": U0, "segments": len(segs), "tailles": sorted(len(x) for x in segs)}
for g, q in QOS.items():
    ext = {}
    for i, seg in enumerate(segs):
        for p in seg:
            ext[p] = {"cellule": (("hypo", i), None, g, 900), "qos": q}
    u, m, dp = U(ext)
    res[f"couverture_totale_{g}"] = {"U_T": round(u, 4), "dU": round(u - U0, 4), "Q": round(m["Q"], 4), "H": round(m["H"], 4),
                                     "O": round(m["O"], 4), "T": round(m["T"], 4), "qos_min": round(dp["qos_min"], 4)}
    print(g, res[f"couverture_totale_{g}"])
res["segments_dU_negatif_3G"] = int((df[df.generation == "3G"].dU < 0).sum())
res["points_dans_segments_dU_negatif_3G"] = int(df[(df.generation == "3G") & (df.dU < 0)].taille.sum())
json.dump(res, open(os.path.join(out, "analyse_dU_resume.json"), "w"), indent=2)
print(res["segments_dU_negatif_3G"], res["points_dans_segments_dU_negatif_3G"])
