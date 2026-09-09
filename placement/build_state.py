"""
build_state.py — Construit les canaux statiques de l'état (§15)
==================================================================
Les 6 premiers canaux, qui ne changent jamais pendant un épisode :
environnement permanent (altitude, végétation, tunnel, distance à la
voie) et météo du scénario (pluie, température).

Pour chaque cellule de la grille, on prend les valeurs du point de
trajet le plus proche (recherche par arbre KD, rapide même sur toute
la grille). C'est une approximation simple : loin de la voie, le point
le plus proche n'est pas forcément représentatif du terrain exact à
cet endroit. Pour l'altitude spécifiquement, on pourrait un jour
interroger directement le DEM (comme le fait déjà `_get_elevation`
dans prepare_data.py, réutilisé par le simulateur) pour une valeur
plus précise, mais on part de cette version simple d'abord.

Convention : le tunnel est mis à 1 seulement si un point de trajet
DANS cette cellule précise (pas juste "le plus proche") est en tunnel
— contrairement aux autres canaux, un tunnel est un objet localisé, ne
pas l'étendre aux cellules voisines juste parce que leur point le plus
proche est tunnelé.

La végétation (`vegetation`) est catégorielle en texte (urbain,
foret_dense, foret_legere, plaine), pas numérique. Comme pour
opérateur/génération, un encodage ordinal (0,1,2,3) imposerait un ordre
arbitraire sans fondement physique vérifié. Encodée en one-hot, 4
canaux au lieu d'1 seul — ça fait passer le total de 6 à 9 canaux
statiques, donc 24 à 27 canaux au total. Changement réel, pas silencieux.

Lancer : python build_state.py (teste sur dataset_base.csv)
"""

import os

import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree

from grid_config import GrilleCorridor, RESOLUTION_M, VERS_METRES, construire_grille_depuis_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")
DATASET_ENRICHI = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")

CATEGORIES_VEGETATION = ["urbain", "foret_dense", "foret_legere", "plaine"]
NOMS_CANAUX_STATIQUES = (
    ["altitude"]
    + [f"vegetation_{v}" for v in CATEGORIES_VEGETATION]
    + ["tunnel", "distance_voie", "pluie", "temperature"]
)


def normaliser(valeurs, val_min, val_max):
    """Min-max vers [0,1]. Si min==max (valeur constante), renvoie 0 partout."""
    if val_max - val_min < 1e-9:
        return np.zeros_like(valeurs, dtype=float)
    return np.clip((valeurs - val_min) / (val_max - val_min), 0.0, 1.0)


def construire_canaux_statiques(grille, df_points, df_meteo_scenario):
    """Construit les 9 canaux statiques, shape (9, nx, ny).

    df_points : point_id, lat, lon, altitude_m, vegetation, in_tunnel
        (un point par ligne, comme dataset_base.csv).
    df_meteo_scenario : point_id, pluie_mm_h, temp_c, DÉJÀ filtré sur un
        seul scénario météo (la météo dépend du scénario, pas juste du point).
    """
    df = df_points.merge(df_meteo_scenario[["point_id", "pluie_mm_h", "temp_c"]], on="point_id")
    x, y = VERS_METRES.transform(df["lon"].values, df["lat"].values)
    arbre = cKDTree(np.column_stack([x, y]))

    # Centres de toutes les cellules de la grille, en un seul bloc (rapide).
    i_idx, j_idx = np.meshgrid(np.arange(grille.nx), np.arange(grille.ny), indexing="ij")
    x_centres = grille.x_min + (i_idx.ravel() + 0.5) * RESOLUTION_M
    y_centres = grille.y_min + (j_idx.ravel() + 0.5) * RESOLUTION_M

    distances, indices_plus_proche = arbre.query(np.column_stack([x_centres, y_centres]))

    canaux = np.zeros((9, grille.nx, grille.ny), dtype=float)

    altitude = df["altitude_m"].values[indices_plus_proche]
    canaux[0] = normaliser(altitude, df["altitude_m"].min(), df["altitude_m"].max()).reshape(grille.nx, grille.ny)

    # Végétation, one-hot : un canal par catégorie (indices 1 à 4).
    vegetation_plus_proche = df["vegetation"].values[indices_plus_proche]
    for k, categorie in enumerate(CATEGORIES_VEGETATION):
        canaux[1 + k] = (vegetation_plus_proche == categorie).astype(float).reshape(grille.nx, grille.ny)

    # Distance à la voie : directement la distance renvoyée par l'arbre KD,
    # normalisée par la marge du corridor (10km, cf grid_config.py).
    canaux[6] = normaliser(distances, 0, 10_000).reshape(grille.nx, grille.ny)

    pluie = df["pluie_mm_h"].values[indices_plus_proche]
    canaux[7] = normaliser(pluie, df["pluie_mm_h"].min(), df["pluie_mm_h"].max()).reshape(grille.nx, grille.ny)

    temp = df["temp_c"].values[indices_plus_proche]
    canaux[8] = normaliser(temp, df["temp_c"].min(), df["temp_c"].max()).reshape(grille.nx, grille.ny)

    # Tunnel : localisé, pas "le plus proche". On regroupe les points par
    # cellule (comme pour couverture/QoS) et on marque 1 si au moins un
    # point de CETTE cellule précise est en tunnel.
    df_tunnel = df[df["in_tunnel"]]
    for _, r in df_tunnel.iterrows():
        c = grille.gps_vers_cellule(r["lat"], r["lon"])
        if c is not None:
            canaux[5, c[0], c[1]] = 1.0

    return canaux


def valider_canaux_statiques(canaux, grille):
    """Vérifications de base avant de faire confiance à ces canaux."""
    assert canaux.shape == (9, grille.nx, grille.ny), f"Forme inattendue : {canaux.shape}"
    print(f"Forme du tenseur statique : {canaux.shape}")

    n_nan = np.isnan(canaux).sum()
    print(f"Valeurs NaN : {n_nan}")
    assert n_nan == 0, "Des NaN se sont glissés dans les canaux statiques"

    for i, nom in enumerate(NOMS_CANAUX_STATIQUES):
        print(f"  {nom:18s} : min={canaux[i].min():.3f}  max={canaux[i].max():.3f}  moyenne={canaux[i].mean():.3f}")

    # Les 4 canaux one-hot de végétation doivent sommer à 0 ou 1 partout
    # (jamais 2 catégories actives en même temps pour une cellule).
    somme_vegetation = canaux[1:5].sum(axis=0)
    assert np.all((somme_vegetation == 0) | (somme_vegetation == 1)), \
        "Une cellule a plusieurs catégories de végétation actives à la fois"
    print(f"\nOne-hot végétation cohérent : chaque cellule a 0 ou 1 catégorie active.")

    n_tunnel = int(canaux[5].sum())
    print(f"Cellules marquées tunnel : {n_tunnel}")
    assert n_tunnel > 0, "Aucune cellule tunnel trouvée, alors que la ligne en contient (à vérifier)"

    # Les cellules du corridor devraient, en général, être plus proches de
    # la voie (distance normalisée plus basse) que celles hors corridor.
    dist_dans_corridor = canaux[6][grille.corridor_mask].mean()
    dist_hors_corridor = canaux[6][~grille.corridor_mask].mean()
    print(f"\nDistance moyenne (normalisée) dans le corridor : {dist_dans_corridor:.3f}")
    print(f"Distance moyenne (normalisée) hors corridor     : {dist_hors_corridor:.3f}")
    assert dist_dans_corridor < dist_hors_corridor, "Incohérence : le corridor n'est pas plus proche de la voie que l'extérieur"

    print("\nToutes les vérifications passent.")


if __name__ == "__main__":
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    df_enrichi = pd.read_csv(DATASET_ENRICHI, low_memory=False)
    df_meteo_s1 = df_enrichi[df_enrichi["scenario_id"] == "S1"][["point_id", "pluie_mm_h", "temp_c"]].drop_duplicates("point_id")

    df_points = df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")
    grille = construire_grille_depuis_dataset(df_base)

    canaux = construire_canaux_statiques(grille, df_points, df_meteo_s1)
    valider_canaux_statiques(canaux, grille)
