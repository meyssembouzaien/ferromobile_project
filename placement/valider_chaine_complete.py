"""
valider_chaine_complete.py : validation de la chaîne d'évaluation sur données réelles.

Un déploiement test est ajouté au réseau réel, puis on compare AVANT et APRÈS :
simulation -> DP -> U(I) -> r_t -> coût -> r_fin.

Corrections :
    - compute_r_fin() appelé avec n_total, en arguments nommés.
    - antenne au centre de la cellule de grille, comme dans environment_rl.py
      (au point exact du trajet, P.1812 donne un cas dégénéré à distance nulle).
    - paramètres de r_fin identiques à ceux de environment_rl.py.

Lancer : python valider_chaine_complete.py
"""

import os

import pandas as pd

from grid_config import construire_grille_depuis_dataset
from simulateur_deploiement import simuler_deploiement
from trajectory_dp import trajectory_dp
from cost_model import get_total_cost
from utility import WEIGHTS_REFERENCE, normaliser_metriques, compute_U, compute_r_t, compute_r_fin

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")
DATASET_ENRICHI = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")

SCENARIO = "S1"
D_COV = 1.0
LAMBDA_H, LAMBDA_O, LAMBDA_T = 0.05, 0.05, 0.05   # pénalités de la DP (§10)

# Mêmes valeurs que environment_rl.py
Q_CRITIQUE = 0.5
R_COUVERTURE, R_QUALITE, R_SUCCES, R_EFF = 1.0, 0.5, 0.2, 0.2
BUDGET = 10_000_000.0

# Déploiement test
GENERATION_TEST = "4G"
BANDE_TEST = 1800


def cellules_par_point(df_reel):
    """Candidats réels de chaque point : {point_id: [cellules]}."""
    candidats = {}
    for pid, lignes in df_reel.groupby("point_id"):
        candidats[pid] = [
            {"cellule": ((round(r.ant_lat, 6), round(r.ant_lon, 6)), r.operateur, r.generation, r.bande_mhz),
             "qos": r.qos}
            for r in lignes.itertuples()
        ]
    return candidats


def evaluer(candidats, meilleur_debit, points_hors_tunnel):
    """DP + zones blanches + U(I) pour un réseau donné."""
    points = sorted(candidats)
    resultat = trajectory_dp(points, [candidats[p] for p in points], LAMBDA_H, LAMBDA_O, LAMBDA_T)

    debit = meilleur_debit.reindex(list(points_hors_tunnel)).fillna(0)   # pas de lien = 0 Mbps
    n_white = int((debit < D_COV).sum())
    n_eval = len(points)
    n_total = len(points_hors_tunnel)

    m = normaliser_metriques(resultat, n_eval, n_white, n_total)
    w = WEIGHTS_REFERENCE
    U = compute_U(m, w["w_Q"], w["w_W"], w["w_H"], w["w_O"], w["w_T"])
    return {"dp": resultat, "m": m, "U": U, "n_white": n_white, "n_eval": n_eval, "n_total": n_total}


def afficher(titre, e):
    dp, m = e["dp"], e["m"]
    print(f"\n{'=' * 50}\n{titre}\n{'=' * 50}")
    print(f"J*            : {dp['J_star']:.2f}")
    print(f"QoS moyenne   : {dp['qos_mean']:.4f}")
    print(f"QoS pire cas  : {dp['qos_min']:.4f}")
    print(f"N_H, N_O, N_T : {dp['N_H']}, {dp['N_O']}, {dp['N_T']}")
    print(f"N_eval        : {e['n_eval']}")
    print(f"N_white       : {e['n_white']} / {e['n_total']}")
    print(f"Q, W, H, O, T : {m['Q']:.4f}, {m['W']:.4f}, {m['H']:.4f}, {m['O']:.4f}, {m['T']:.4f}")
    print(f"U(I)          : {e['U']:.4f}")


def main():
    # 1. Données
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    df = pd.read_csv(DATASET_ENRICHI, low_memory=False)
    df = df[df["scenario_id"] == SCENARIO]

    df_points = df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")
    points_hors_tunnel = set(df_points[~df_points["in_tunnel"]]["point_id"])
    df_meteo = df[["point_id", "pluie_mm_h"]].drop_duplicates("point_id")
    grille = construire_grille_depuis_dataset(df_base)

    # 2. Réseau réel
    df_reel = df[(~df["in_tunnel"]) & (df["ant_id"].notna())]
    candidats = cellules_par_point(df_reel)
    debit_reel = df_reel.groupby("point_id")["debit_adj_mbps"].max()

    avant = evaluer(candidats, debit_reel, points_hors_tunnel)
    afficher("AVANT (réseau réel)", avant)

    # 3. Cellule test : celle du point blanc du milieu
    blancs = sorted(p for p in points_hors_tunnel if debit_reel.get(p, 0) < D_COV)
    point = df_points[df_points["point_id"] == blancs[len(blancs) // 2]].iloc[0]
    cellule = grille.gps_vers_cellule(point["lat"], point["lon"])
    lat_ant, lon_ant = grille.cellule_vers_gps(*cellule)   # centre de la cellule, comme le RL

    # 4. Simulation du déploiement
    resultats = simuler_deploiement(lat_ant, lon_ant, GENERATION_TEST, BANDE_TEST, df_points, df_meteo)
    resultats = [r for r in resultats if r["point_id"] in points_hors_tunnel]

    # 5. Coût (sites réels = cellules de grille)
    sites = {grille.gps_vers_cellule(la, lo) for la, lo in df_reel[["ant_lat", "ant_lon"]].drop_duplicates().values}
    sites.discard(None)
    cout = get_total_cost(cellule, GENERATION_TEST, BANDE_TEST, sites)

    print(f"\n{'=' * 50}\nDÉPLOIEMENT TEST\n{'=' * 50}")
    print(f"Cellule       : {cellule}, antenne en ({lat_ant:.4f}, {lon_ant:.4f})")
    print(f"Technologie   : {GENERATION_TEST} / {BANDE_TEST} MHz")
    print(f"Points servis : {len(resultats)}")
    print(f"Coût          : {cout:,.0f} € ({'site existant' if cellule in sites else 'nouveau site'})")

    # 6. Réseau après ajout
    candidats_apres = {p: list(c) for p, c in candidats.items()}
    for r in resultats:
        candidats_apres.setdefault(r["point_id"], []).append({"cellule": r["cellule"], "qos": r["qos"]})
    debit_hypo = pd.Series({r["point_id"]: r["debit_adj_mbps"] for r in resultats})
    debit_apres = pd.concat([debit_reel, debit_hypo], axis=1).max(axis=1)

    apres = evaluer(candidats_apres, debit_apres, points_hors_tunnel)
    afficher("APRÈS AJOUT", apres)

    # 7. Récompenses (arguments nommés : une erreur claire si la signature change)
    r_t = compute_r_t(apres["U"], avant["U"])
    r_fin = compute_r_fin(
        n_white=apres["n_white"], n_total=apres["n_total"],
        qos_min=apres["dp"]["qos_min"], cost_total=cout, budget=BUDGET,
        Q_critique=Q_CRITIQUE, R_couverture=R_COUVERTURE, R_qualite=R_QUALITE,
        R_succes=R_SUCCES, R_eff=R_EFF)

    print(f"\n{'=' * 50}\nIMPACT\n{'=' * 50}")
    print(f"ΔN_white      : {apres['n_white'] - avant['n_white']:+d}")
    print(f"ΔQoS moyenne  : {apres['dp']['qos_mean'] - avant['dp']['qos_mean']:+.4f}")
    print(f"ΔQoS pire cas : {apres['dp']['qos_min'] - avant['dp']['qos_min']:+.4f}")
    print(f"r_t = ΔU      : {r_t:+.4f}")
    print(f"r_fin         : {r_fin:+.4f}")
    print(f"Retour total  : {r_t + r_fin:+.4f}")


if __name__ == "__main__":
    main()