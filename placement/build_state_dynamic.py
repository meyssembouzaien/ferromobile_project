"""
build_state_dynamic.py — Construit les 18 canaux dynamiques (§15)
====================================================================
Infrastructure existante (16, opérateur x génération) et performance
actuelle (2, couverture + QoS). Recalculés après chaque action du RL,
contrairement aux 9 canaux statiques de build_state.py.

Couverture et QoS par cellule utilisent le MEILLEUR candidat disponible
à chaque point (réel + hypothétique confondus), pas seulement la
cellule que la DP choisirait — cohérent avec la définition de N_white
(§6, best débit dispo), pas avec Q(π*_λ) qui dépend d'un choix de
trajectoire (§11). Les canaux d'état décrivent la situation du réseau,
pas le résultat d'une optimisation de trajectoire.

Lancer : python build_state_dynamic.py (teste sur dataset_base.csv, S1)
"""

import os

import numpy as np
import pandas as pd

from grid_config import construire_grille_depuis_dataset
from build_state import construire_canaux_statiques, NOMS_CANAUX_STATIQUES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")
DATASET_ENRICHI = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")

OPERATEURS = ["ORANGE", "SFR", "BOUYGUES TELECOM", "FREE MOBILE"]
GENERATIONS = ["2G", "3G", "4G", "5G"]
D_COV = 1.0


def construire_canaux_infrastructure(grille, df_reel, cellules_hypothetiques=None):
    """16 canaux (opérateur x génération), présence binaire par cellule.

    df_reel : lignes du réseau réel (ant_lat, ant_lon, operateur, generation),
        une ligne par cellule radio existante (pas par point de trajet :
        dédupliquer d'abord si besoin).
    cellules_hypothetiques : liste optionnelle de dicts {"cellule": (site,
        operateur, generation, bande)} ajoutés cet épisode. operateur=None
        pour une cellule hypothétique (§9) : n'apparaît dans aucun des 16
        canaux, puisqu'aucun opérateur n'est modélisé pour elle.
    """
    canaux = np.zeros((16, grille.nx, grille.ny), dtype=float)
    index_og = {(o, g): k for k, (o, g) in
                enumerate((o, g) for o in OPERATEURS for g in GENERATIONS)}

    sites_reels = df_reel[["ant_lat", "ant_lon", "operateur", "generation"]].drop_duplicates()
    for _, r in sites_reels.iterrows():
        c = grille.gps_vers_cellule(r["ant_lat"], r["ant_lon"])
        cle = (r["operateur"], r["generation"])
        if c is not None and cle in index_og:
            canaux[index_og[cle], c[0], c[1]] = 1.0

    if cellules_hypothetiques:
        for item in cellules_hypothetiques:
            site, operateur, generation, _bande = item["cellule"]
            if operateur is None:
                continue  # cellule hypothétique : aucun opérateur modélisé (§9)
            lat, lon = site if isinstance(site, tuple) and len(site) == 2 else (None, None)
            if lat is None:
                continue
            c = grille.gps_vers_cellule(lat, lon)
            cle = (operateur, generation)
            if c is not None and cle in index_og:
                canaux[index_og[cle], c[0], c[1]] = 1.0

    return canaux


def construire_canaux_performance(grille, df_points, meilleur_par_point):
    """2 canaux : couverture (fraction de points couverts par cellule) et
    QoS (moyenne parmi les points couverts). Convention :
        (-1, -1)  cellule sans aucun point de trajet
        ( 0, -1)  points de trajet présents, mais aucun couvert
        ( C,  Q)  sinon
    """
    couverture = np.full((grille.nx, grille.ny), -1.0)
    qos_moy = np.full((grille.nx, grille.ny), -1.0)

    df = df_points.merge(meilleur_par_point, on="point_id", how="left")
    df["couvert"] = df["meilleur_debit"].fillna(0) >= D_COV

    par_cellule = {}
    for _, r in df.iterrows():
        c = grille.gps_vers_cellule(r["lat"], r["lon"])
        if c is None:
            continue
        par_cellule.setdefault(c, []).append(r)

    for c, lignes in par_cellule.items():
        n_total = len(lignes)
        couverts = [l for l in lignes if l["couvert"]]
        couverture[c[0], c[1]] = len(couverts) / n_total
        if couverts:
            qos_moy[c[0], c[1]] = sum(l["meilleur_qos"] for l in couverts) / len(couverts)
        # sinon : couverture=0 déjà posé, qos_moy reste -1 (convention "0 couvert")

    return np.stack([couverture, qos_moy])


def construire_etat_complet(canaux_statiques, canaux_infra, canaux_perf):
    """Empile les 3 blocs en un seul tenseur (27, nx, ny)."""
    return np.concatenate([canaux_statiques, canaux_infra, canaux_perf], axis=0)


def valider_canaux_dynamiques(canaux_infra, canaux_perf, grille):
    assert canaux_infra.shape == (16, grille.nx, grille.ny)
    assert canaux_perf.shape == (2, grille.nx, grille.ny)
    print(f"Infrastructure : {int(canaux_infra.sum())} cellules occupées au total (toutes combinaisons confondues)")
    print(f"Cellules avec plusieurs technologies au même endroit : {int((canaux_infra.sum(axis=0) > 1).sum())}")

    couverture, qos = canaux_perf
    hors_trajet = (couverture == -1) & (qos == -1)
    zero_couvert = (couverture == 0) & (qos == -1)
    normal = couverture > 0
    print(f"\nCellules hors trajet (-1,-1)      : {int(hors_trajet.sum())}")
    print(f"Cellules trajet, 0% couvert (0,-1) : {int(zero_couvert.sum())}")
    print(f"Cellules avec couverture > 0       : {int(normal.sum())}")
    assert int(hors_trajet.sum()) + int(zero_couvert.sum()) + int(normal.sum()) == grille.nx * grille.ny
    print("\nToutes les vérifications passent.")


if __name__ == "__main__":
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    df_enrichi = pd.read_csv(DATASET_ENRICHI, low_memory=False)
    df_s1 = df_enrichi[df_enrichi["scenario_id"] == "S1"]

    df_points = df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")
    df_meteo_s1 = df_enrichi[df_enrichi["scenario_id"] == "S1"][["point_id", "pluie_mm_h", "temp_c"]].drop_duplicates("point_id")
    grille = construire_grille_depuis_dataset(df_base)

    canaux_stat = construire_canaux_statiques(grille, df_points, df_meteo_s1)

    df_reel = df_s1[df_s1["ant_id"].notna()]
    canaux_infra = construire_canaux_infrastructure(grille, df_reel)

    meilleur_par_point = (df_s1.groupby("point_id")
                           .agg(meilleur_debit=("debit_adj_mbps", "max"))
                           .reset_index())
    # meilleure qos = qos de la ligne au meilleur debit, pas juste max(qos) isolement.
    # Certains points n'ont aucune ligne avec débit valide (aucun candidat réel) :
    # on les exclut avant idxmax, meilleur_par_point les aura quand même avec debit=NaN.
    df_s1_avec_debit = df_s1.dropna(subset=["debit_adj_mbps"])
    idx_meilleur = df_s1_avec_debit.loc[df_s1_avec_debit.groupby("point_id")["debit_adj_mbps"].idxmax()]
    meilleur_par_point = meilleur_par_point.merge(
        idx_meilleur[["point_id", "qos"]].rename(columns={"qos": "meilleur_qos"}), on="point_id", how="left")

    canaux_perf = construire_canaux_performance(grille, df_points, meilleur_par_point)

    etat = construire_etat_complet(canaux_stat, canaux_infra, canaux_perf)
    print(f"État complet : {etat.shape}\n")
    valider_canaux_dynamiques(canaux_infra, canaux_perf, grille)
