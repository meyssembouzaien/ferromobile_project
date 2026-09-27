"""
build_state_dynamic.py : canaux dynamiques de l'état (§15), recalculés après chaque action.

    16 canaux  infrastructure réelle (opérateur x génération)
     2 canaux  performance actuelle (couverture, QoS)
     1 canal   part des zones blanches restantes à moins de RAYON_BLANCS_M
     3 canaux  déploiements du RL (3G, 4G, 5G)

État complet = 9 canaux statiques + 22 dynamiques = 31 canaux.

Lancer : python build_state_dynamic.py
"""

import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from grid_config import construire_grille_depuis_dataset, RESOLUTION_M, VERS_METRES
from build_state import construire_canaux_statiques

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")
DATASET_ENRICHI = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")

OPERATEURS = ["ORANGE", "SFR", "BOUYGUES TELECOM", "FREE MOBILE"]
GENERATIONS = ["2G", "3G", "4G", "5G"]
GENERATIONS_RL = ["3G", "4G", "5G"]   # la 2G n'est pas dans l'espace d'action
D_COV = 1.0
RAYON_BLANCS_M = 2000                 # le meilleur site glouton couvre environ 1,3 km de voie

N_CANAUX_ETAT = 9 + 16 + 2 + 1 + len(GENERATIONS_RL)   # 31


def centres_cellules(grille):
    """Coordonnées (x, y) en mètres du centre de chaque cellule, ordre (ix, iy)."""
    i, j = np.meshgrid(np.arange(grille.nx), np.arange(grille.ny), indexing="ij")
    x = grille.x_min + (i.ravel() + 0.5) * RESOLUTION_M
    y = grille.y_min + (j.ravel() + 0.5) * RESOLUTION_M
    return np.column_stack([x, y])


def construire_canaux_infrastructure(grille, df_reel):
    """16 canaux : 1 là où le réseau réel a une antenne (opérateur, génération)."""
    canaux = np.zeros((16, grille.nx, grille.ny))
    index = {(o, g): k for k, (o, g) in enumerate((o, g) for o in OPERATEURS for g in GENERATIONS)}

    antennes = df_reel[["ant_lat", "ant_lon", "operateur", "generation"]].drop_duplicates()
    for r in antennes.itertuples():
        c = grille.gps_vers_cellule(r.ant_lat, r.ant_lon)
        if c is not None and (r.operateur, r.generation) in index:
            canaux[index[(r.operateur, r.generation)], c[0], c[1]] = 1.0
    return canaux


def construire_canaux_performance(grille, df_points, meilleur_par_point):
    """2 canaux : couverture et QoS par cellule.

    (-1, -1) : pas de point de trajet ; (0, -1) : points présents mais aucun couvert.
    """
    couverture = np.full((grille.nx, grille.ny), -1.0)
    qos_moy = np.full((grille.nx, grille.ny), -1.0)

    df = df_points.merge(meilleur_par_point, on="point_id", how="left")
    df["couvert"] = df["meilleur_debit"].fillna(0) >= D_COV

    par_cellule = {}
    for r in df.itertuples():
        c = grille.gps_vers_cellule(r.lat, r.lon)
        if c is not None:
            par_cellule.setdefault(c, []).append(r)

    for (i, j), lignes in par_cellule.items():
        couverts = [l for l in lignes if l.couvert]
        couverture[i, j] = len(couverts) / len(lignes)
        if couverts:
            qos_moy[i, j] = sum(l.meilleur_qos for l in couverts) / len(couverts)

    return np.stack([couverture, qos_moy])


def construire_canal_blancs_proches(grille, xy_points, est_blanc, rayon_m=RAYON_BLANCS_M):
    """1 canal : part des zones blanches restantes à moins de rayon_m de chaque cellule."""
    canal = np.zeros((1, grille.nx, grille.ny))
    n_blancs = int(est_blanc.sum())
    if n_blancs == 0:
        return canal
    arbre = cKDTree(xy_points[est_blanc])
    comptes = arbre.query_ball_point(centres_cellules(grille), r=rayon_m, return_length=True)
    canal[0] = (comptes / n_blancs).reshape(grille.nx, grille.ny)
    return canal


def construire_canaux_deploiements_rl(grille, infrastructure_deployee):
    """3 canaux (3G, 4G, 5G) : 1 là où le RL a déjà construit."""
    canaux = np.zeros((len(GENERATIONS_RL), grille.nx, grille.ny))
    for ix, iy, g, _ in infrastructure_deployee:
        canaux[GENERATIONS_RL.index(g), ix, iy] = 1.0
    return canaux


if __name__ == "__main__":
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    df = pd.read_csv(DATASET_ENRICHI, low_memory=False)
    df = df[df["scenario_id"] == "S1"]

    df_points = df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")
    df_meteo = df[["point_id", "pluie_mm_h", "temp_c"]].drop_duplicates("point_id")
    grille = construire_grille_depuis_dataset(df_base)
    df_reel = df[(~df["in_tunnel"]) & (df["ant_id"].notna())]

    stat = construire_canaux_statiques(grille, df_points, df_meteo)
    infra = construire_canaux_infrastructure(grille, df_reel)

    meilleur = df_reel.groupby("point_id").agg(meilleur_debit=("debit_adj_mbps", "max"),
                                               meilleur_qos=("qos", "max")).reset_index()
    perf = construire_canaux_performance(grille, df_points, meilleur)

    hors_tunnel = df_points[~df_points["in_tunnel"]]
    x, y = VERS_METRES.transform(hors_tunnel["lon"].values, hors_tunnel["lat"].values)
    debit = meilleur.set_index("point_id")["meilleur_debit"].reindex(hors_tunnel["point_id"]).fillna(0)
    est_blanc = (debit < D_COV).values
    blancs = construire_canal_blancs_proches(grille, np.column_stack([x, y]), est_blanc)

    rl = construire_canaux_deploiements_rl(grille, [(40, 119, "3G", 900)])

    etat = np.concatenate([stat, infra, perf, blancs, rl])
    assert etat.shape == (N_CANAUX_ETAT, grille.nx, grille.ny)
    assert not np.isnan(etat).any()
    assert rl.sum() == 1 and rl[0, 40, 119] == 1
    print(f"État complet : {etat.shape}")
    print(f"Zones blanches : {int(est_blanc.sum())}")
    print(f"Canal zones blanches proches : max {blancs.max():.3f}, cellules > 0 : {int((blancs > 0).sum())}")
    print("Toutes les vérifications passent.")