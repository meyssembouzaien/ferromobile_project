"""
grid_config.py — Construit et valide la grille 2D du corridor
==================================================================
La grille est calculée à partir des vraies coordonnées du trajet, pas
d'une hypothèse "longueur x largeur" : la ligne n'est pas alignée sur
les axes est-ouest/nord-sud, donc on utilise une vraie projection
métrique (EPSG:2154, la même que prepare_data.py utilise déjà pour le
tampon du corridor), pas lat/lon traité comme un quadrillage carré.

Deux notions à ne pas confondre :
  - l'ÉTENDUE de la grille (le rectangle qui contient tout, utilisé pour
    dimensionner le tableau) = bounding box du trajet + marge de 10 km ;
  - le CORRIDOR D'ÉTUDE réel (§6 du cahier des charges) = un vrai tampon
    de 10 km autour de la ligne, une forme qui suit la voie, pas un
    rectangle. ~40% de l'étendue rectangulaire est hors de ce vrai
    corridor (vérifié sur données réelles) : ces cellules existent dans
    le tableau mais ne sont PAS des positions candidates valides pour
    une antenne. C'est à ça que sert corridor_mask.

Axes : x = est-ouest (EPSG:2154), y = nord-sud. nx = nombre de cellules
selon x, ny = nombre de cellules selon y. Le tenseur final aura la forme
(nx, ny, canaux) — à garder cohérent partout dans le code du CNN.

Lancer : python grid_config.py (fait tourner les vérifications ci-dessous)
"""

import os

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import LineString, Point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")

RESOLUTION_M = 250
MARGE_M = 10_000  # corridor d'étude, §6 du cahier des charges

VERS_METRES = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
VERS_GPS = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)


class GrilleCorridor:
    """Définit la grille, la conversion GPS <-> cellule, et le masque du
    vrai corridor d'étude (pas juste le rectangle qui le contient)."""

    def __init__(self, points_trajet_lat_lon):
        """points_trajet_lat_lon : liste/array de (lat, lon) DANS L'ORDRE
        du trajet (nécessaire pour construire la ligne du corridor)."""
        lats = np.array([p[0] for p in points_trajet_lat_lon])
        lons = np.array([p[1] for p in points_trajet_lat_lon])
        x, y = VERS_METRES.transform(lons, lats)

        # Étendue du tableau : bounding box + marge. Toujours plus large
        # que le vrai corridor (mathématiquement garanti), jamais plus
        # étroite : aucun point du vrai corridor ne peut tomber dehors.
        self.x_min = x.min() - MARGE_M
        self.y_min = y.min() - MARGE_M
        x_max = x.max() + MARGE_M
        y_max = y.max() + MARGE_M

        # +1 cellule de marge de sécurité : évite qu'un point pile sur la
        # borne max tombe hors grille par erreur d'arrondi.
        self.nx = int(np.ceil((x_max - self.x_min) / RESOLUTION_M)) + 1
        self.ny = int(np.ceil((y_max - self.y_min) / RESOLUTION_M)) + 1

        # Le vrai corridor : un tampon de 10 km autour de la ligne réelle,
        # pas autour de la bounding box. Sert à restreindre l'espace
        # d'action du RL aux positions réellement dans le corridor d'étude.
        ligne = LineString(list(zip(x, y)))
        self.corridor_geometrique = ligne.buffer(MARGE_M)
        self.corridor_mask = self._construire_masque_corridor()

    def _construire_masque_corridor(self):
        """Grille booléenne : True si le centre de la cellule est dans le
        vrai corridor de 10 km, False sinon (juste dans le rectangle)."""
        masque = np.zeros((self.nx, self.ny), dtype=bool)
        for i in range(self.nx):
            x_centre = self.x_min + (i + 0.5) * RESOLUTION_M
            for j in range(self.ny):
                y_centre = self.y_min + (j + 0.5) * RESOLUTION_M
                masque[i, j] = self.corridor_geometrique.covers(Point(x_centre, y_centre))
        return masque

    def gps_vers_cellule(self, lat, lon):
        """Convertit une position GPS en indices de cellule (i,j).
        Retourne None si la position tombe hors du tableau."""
        x, y = VERS_METRES.transform(lon, lat)
        i = int((x - self.x_min) / RESOLUTION_M)
        j = int((y - self.y_min) / RESOLUTION_M)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def cellule_vers_gps(self, i, j):
        """Convertit une cellule (i,j) vers les coordonnées GPS de son centre."""
        x = self.x_min + (i + 0.5) * RESOLUTION_M
        y = self.y_min + (j + 0.5) * RESOLUTION_M
        lon, lat = VERS_GPS.transform(x, y)
        return lat, lon


def construire_grille_depuis_dataset(df_base):
    """Construit la grille à partir de dataset_base.csv (colonnes point_id,
    lat, lon), triées par point_id pour respecter l'ordre du trajet.

    Suppose que chaque point_id correspond à une seule position GPS,
    puisque la grille dépend directement de l'ordre du trajet pour
    construire la ligne du corridor : vérifié explicitement ci-dessous.
    """
    incoherences = df_base.groupby("point_id")[["lat", "lon"]].nunique()
    n_incoherents = (incoherences > 1).sum().sum()
    assert n_incoherents == 0, f"{n_incoherents} point_id ont plusieurs coordonnées GPS différentes"

    points = df_base[["point_id", "lat", "lon"]].drop_duplicates("point_id").sort_values("point_id")
    return GrilleCorridor(points[["lat", "lon"]].values)


def valider(grille, df_base, nb_sites_attendu=None):
    """Vérifie la grille sous tous les angles utiles avant de construire
    les canaux du CNN par-dessus."""
    print(f"Grille : {grille.nx} x {grille.ny} cellules à {RESOLUTION_M}m "
          f"({grille.nx * grille.ny:,} cellules)")
    print(f"Cellules dans le vrai corridor de 10km : {grille.corridor_mask.sum():,} "
          f"({grille.corridor_mask.sum() / (grille.nx*grille.ny) * 100:.0f}% du tableau)")

    # 1. Tous les points de trajet dans la grille
    points = df_base[["lat", "lon"]].drop_duplicates()
    hors_grille = sum(1 for _, r in points.iterrows()
                       if grille.gps_vers_cellule(r["lat"], r["lon"]) is None)
    print(f"\nPoints de trajet hors grille : {hors_grille} / {len(points)}")
    assert hors_grille == 0, "Des points de trajet tombent hors de la grille"

    # 2. Tous les sites réels dans la grille, aucune collision
    #    (ant_lat, ant_lon) identifie déjà un site physique, pas une
    #    configuration : vérifié séparément (204 ant_id -> 29 sites uniques).
    sites = df_base[["ant_lat", "ant_lon"]].dropna().drop_duplicates()
    print(f"Sites physiques dans le dataset : {len(sites)}"
          + (f" (attendu : {nb_sites_attendu})" if nb_sites_attendu else ""))
    if nb_sites_attendu is not None:
        assert len(sites) == nb_sites_attendu, "Nombre de sites différent de ce qui était attendu"

    cellules_sites = {}
    collisions, hors_grille_sites = 0, 0
    for _, r in sites.iterrows():
        c = grille.gps_vers_cellule(r["ant_lat"], r["ant_lon"])
        if c is None:
            hors_grille_sites += 1
            continue
        if c in cellules_sites:
            collisions += 1
        cellules_sites[c] = cellules_sites.get(c, 0) + 1
    print(f"Sites hors grille : {hors_grille_sites} / {len(sites)}")
    print(f"Sites fusionnés à tort dans la même cellule : {collisions}")
    assert hors_grille_sites == 0, "Des sites réels tombent hors de la grille"
    assert collisions == 0, "Deux sites distincts tombent dans la même cellule : résolution trop grossière"

    # dans la grille (le tableau) != dans le vrai corridor (le tampon de
    # 10km) : la grille est volontairement plus grande, donc c'est une
    # vérification séparée, pas la même chose.
    sites_hors_corridor = 0
    for _, r in sites.iterrows():
        x, y = VERS_METRES.transform(r["ant_lon"], r["ant_lat"])
        if not grille.corridor_geometrique.covers(Point(x, y)):
            sites_hors_corridor += 1
    print(f"Sites physiques hors du vrai corridor de 10km : {sites_hors_corridor} / {len(sites)}")
    assert sites_hors_corridor == 0, "Un site réel existe en dehors du corridor d'étude de 10km"

    # 3. Aller-retour GPS -> cellule -> GPS -> cellule : doit redonner la même cellule
    echecs_aller_retour = 0
    for _, r in points.sample(min(100, len(points)), random_state=0).iterrows():
        c1 = grille.gps_vers_cellule(r["lat"], r["lon"])
        lat2, lon2 = grille.cellule_vers_gps(*c1)
        c2 = grille.gps_vers_cellule(lat2, lon2)
        if c1 != c2:
            echecs_aller_retour += 1
    print(f"Échecs du test aller-retour (sur 100 points testés) : {echecs_aller_retour}")
    assert echecs_aller_retour == 0, "La conversion GPS <-> cellule n'est pas cohérente"

    print("\nToutes les vérifications passent.")


if __name__ == "__main__":
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    grille = construire_grille_depuis_dataset(df_base)
    valider(grille, df_base, nb_sites_attendu=29)
