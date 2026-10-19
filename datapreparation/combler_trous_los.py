"""
combler_trous_los.py : reclassify short white gaps caused by the binary LOS test.

Rule (applied once, after prepare_data.py and before enrich_data.py):
  a run of at most MAX_POINTS consecutive white points (scenario S1, best throughput < 1 Mbps),
  with a covered point on each side and no tunnel inside, gets a link to the antenna that serves
  each neighbour, if that antenna is within its nominal range of every point of the gap.
  The link is computed with P.1812 exactly like a LOS link (calc_metrics, los_category = 1).
The binary LOS test is kept everywhere else.

Usage (needs the DEM and crc-covlib, like prepare_data.py):
  python combler_trous_los.py             # rewrites data/processed/dataset_base.csv
  then: python enrich_data.py --scenario ALL
"""
import os
import pandas as pd

import prepare_data as pdp

MAX_POINTS = 12          # 12 points x 20 m = 240 m
PAS_KM = 0.02            # sampling step of the line
D_COV = 1.0              # Mbps

BASE = os.path.join(pdp.OUTPUT_DIR, "dataset_base.csv")
ENRICHI = os.path.join(pdp.OUTPUT_DIR, "dataset_enrichi.csv")


def segments_blancs(base, enrichi):
    pts = base.drop_duplicates("point_id").sort_values("point_id").set_index("point_id")
    ht = pts[~pts.in_tunnel].index.tolist()
    s1 = enrichi[(enrichi.scenario_id == "S1") & (~enrichi.in_tunnel)]
    debit = s1.groupby("point_id").debit_adj_mbps.max().reindex(ht).fillna(0)
    blanc = set(debit[debit < D_COV].index)
    segs, cur = [], []
    for i, p in enumerate(ht):
        if p in blanc:
            cur.append(i)
        elif cur:
            segs.append(cur); cur = []
    if cur:
        segs.append(cur)
    return pts, ht, s1, segs


def main():
    base = pd.read_csv(BASE, low_memory=False)
    enrichi = pd.read_csv(ENRICHI, low_memory=False)
    pts, ht, s1, segs = segments_blancs(base, enrichi)
    ok = s1[s1.debit_adj_mbps >= D_COV]
    best = ok.loc[ok.groupby("point_id").debit_adj_mbps.idxmax()].set_index("point_id")

    nouvelles, combles, refuses = [], [], []
    for seg in segs:
        if len(seg) > MAX_POINTS or seg[0] == 0 or seg[-1] == len(ht) - 1:
            continue
        gauche, droite = ht[seg[0] - 1], ht[seg[-1] + 1]
        trou = [ht[i] for i in seg]
        # no tunnel inside: the two neighbours are exactly (len + 1) steps apart
        if abs(pts.distance_km[droite] - pts.distance_km[gauche] - (len(seg) + 1) * PAS_KM) > PAS_KM / 2:
            refuses.append((trou[0], len(trou), "tunnel ou saut"))
            continue
        if gauche not in best.index or droite not in best.index:
            refuses.append((trou[0], len(trou), "voisin non couvert"))
            continue
        antennes = best.loc[[gauche, droite]].drop_duplicates("ant_id")
        lignes = []
        for ant in antennes.itertuples():
            portee = pdp.get_profile(ant.generation, ant.bande_mhz)[0]
            dists = {p: pdp.haversine_km(pts.lat[p], pts.lon[p], ant.ant_lat, ant.ant_lon) for p in trou}
            if max(dists.values()) > portee:
                continue
            for p in trou:
                (perte_db, snr_base, snr_adj, debit_adj, rtt, pkt_loss, ber, delta_ber,
                 delta_vitesse, delta_snr, modele) = pdp.calc_metrics(
                    dists[p], ant.generation, ant.bande_mhz, pts.vegetation[p], 1, 80.0,
                    pt_lat=pts.lat[p], pt_lon=pts.lon[p], ant_lat=ant.ant_lat, ant_lon=ant.ant_lon,
                    ant_mat_m=pdp.MAT_HEIGHT_M.get(ant.generation, 35.0))
                if modele == "P1812_FAILED":
                    continue
                lignes.append({
                    "point_id": p, "lat": pts.lat[p], "lon": pts.lon[p],
                    "distance_km": pts.distance_km[p], "altitude_m": pts.altitude_m[p],
                    "vegetation": pts.vegetation[p], "in_tunnel": False,
                    "ant_id": str(ant.ant_id), "operateur": ant.operateur, "generation": ant.generation,
                    "bande_mhz": ant.bande_mhz, "commune": ant.commune, "ant_lat": ant.ant_lat,
                    "ant_lon": ant.ant_lon, "ant_alt_abs_m": ant.ant_alt_abs_m,
                    "dist_ant_km": round(dists[p], 4), "los_category": 1, "perte_db": perte_db,
                    "snr_db": snr_base, "snr_adjusted_db": snr_adj, "debit_adjusted": debit_adj,
                    "rtt_ms": rtt, "ber": ber, "delta_ber": delta_ber, "delta_vitesse": delta_vitesse,
                    "delta_snr_db": delta_snr, "packet_loss_pct": pkt_loss,
                    "path_loss_model": "P1812_TROU_LOS", "flag_debit_nul": int(debit_adj == 0.0)})
        if lignes:
            nouvelles += lignes
            combles.append((trou[0], len(trou), "+".join(antennes.ant_id.astype(str))))
        else:
            refuses.append((trou[0], len(trou), "antenne hors portee"))

    pts_combles = {l["point_id"] for l in nouvelles}
    ajout = pd.DataFrame(nouvelles)
    ajout["gen_num"] = ajout.generation.map({"2G": 1, "3G": 2, "4G": 3, "5G": 4}).fillna(0).astype(int)
    ajout["veg_num"] = ajout.vegetation.map({"plaine": 0, "urbain": 1, "foret_legere": 2, "foret_dense": 3}).fillna(0).astype(int)
    # each gap point loses its "no candidate" placeholder (other links, e.g. 2G, are kept) and its
    # new rows go right after its last row: enrich_data.py interpolates the weather along the row
    # order, so every other row keeps its position
    vide = base.point_id.isin(pts_combles) & (base.path_loss_model == "AUCUN_CANDIDAT")
    derniere = base[base.point_id.isin(pts_combles)].groupby("point_id").tail(1).index
    morceaux = []
    for i, ligne in base.iterrows():
        if not vide[i]:
            morceaux.append(ligne.to_frame().T)
        if i in derniere:
            morceaux.append(ajout[ajout.point_id == ligne.point_id])
    base = pd.concat(morceaux, ignore_index=True).astype(base.dtypes.to_dict(), errors="ignore")

    # zone_blanche recomputed as in prepare_data.py
    hors = ~base.in_tunnel.astype(bool)
    debit_max = base[hors].groupby("point_id").debit_adjusted.max()
    base["zone_blanche"] = pd.NA
    base.loc[hors, "zone_blanche"] = (base.loc[hors, "point_id"].map(debit_max) < pdp.DEBIT_COUVERTURE_MBPS).astype(int)
    base["row_id"] = range(len(base))
    base[pd.read_csv(BASE, nrows=0).columns].to_csv(BASE, index=False)

    print(f"{len(combles)} gaps filled ({len(pts_combles)} points, {len(nouvelles)} links); {len(refuses)} refused")
    for r in refuses:
        print("  refused:", r)
    pd.DataFrame(combles, columns=["premier_point", "n_points", "antennes"]).to_csv(
        os.path.join(pdp.OUTPUT_DIR, "trous_los_combles.csv"), index=False)


if __name__ == "__main__":
    main()
