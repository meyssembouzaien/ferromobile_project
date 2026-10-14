"""
couverture_potentielle.py : quelles cellules couvriraient quels points blancs (portée + LOS).

Calculé une fois par ligne et par scénario, puis lu par l'environnement (canal de couverture potentielle).
Même test LOS que prepare_data.check_los_binaire, vectorisé sur le DEM :
12 échantillons/km (16 à 400), courbure k = 4/3, altitudes lues au point arrondi à 4 décimales,
mât selon la génération, véhicule à 4 m. Le seuil de débit n'est pas testé.

Sortie (npz) : cellules (indices ix * ny + iy), points (point_id des points blancs initiaux),
dist_km (cellules x points), los_<génération> (cellules x points, booléen).

Lancer : python couverture_potentielle.py   (écrit data/processed/couverture_potentielle_S1.npz)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datapreparation"))
from prepare_data import (MAT_HEIGHT_M, RECEIVER_HEIGHT_M, LOS_SAMPLE_PER_KM, LOS_MAX_SAMPLES,
                          LOS_RE_EFFECTIVE, TERRAIN_DATA_DIR, haversine_km_vec, get_portee_km)

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_DEFAUT = os.path.join(RACINE, "data", "processed", "couverture_potentielle_S1.npz")


class DEM:
    """Raster en mémoire ; lecture au plus proche pixel, comme rasterio.sample."""

    def __init__(self, chemin=None):
        import rasterio
        chemin = chemin or os.path.join(TERRAIN_DATA_DIR, "eu_dem_courpiere_ambert.tif")
        with rasterio.open(chemin) as ds:
            self.z = ds.read(1).astype(np.float64)
            self.inverse = ~ds.transform
            if ds.nodata is not None:
                self.z[self.z == ds.nodata] = np.nan

    def altitudes(self, lat, lon):
        lat, lon = np.round(lat, 4), np.round(lon, 4)
        col, ligne = self.inverse * (lon, lat)
        ligne, col = np.floor(ligne).astype(np.int64), np.floor(col).astype(np.int64)
        dedans = (ligne >= 0) & (ligne < self.z.shape[0]) & (col >= 0) & (col < self.z.shape[1])
        z = np.full(np.shape(lat), np.nan)
        z[dedans] = self.z[ligne[dedans], col[dedans]]
        return z


def _remplir_nan(z):
    """Comme check_los_binaire : interpolation le long du profil, 0 si tout manque."""
    for i in np.flatnonzero(np.isnan(z).any(axis=1)):
        ligne = z[i]
        ok = ~np.isnan(ligne)
        ligne[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), ligne[ok]) if ok.any() else 0.0
    return z


def los_profils(dem, lat_a, lon_a, sol_a, lat_p, lon_p, dist_km, mats):
    """LOS de plusieurs antennes vers un point. Renvoie un booléen (len(mats), n_antennes)."""
    n_ech = np.minimum(np.maximum(16, dist_km * LOS_SAMPLE_PER_KM) + 1, LOS_MAX_SAMPLES).astype(int)
    resultat = np.ones((len(mats), len(lat_a)), dtype=bool)
    for n in np.unique(n_ech):
        i = np.flatnonzero((n_ech == n) & (dist_km >= 0.05))
        if len(i) == 0 or n <= 2:
            continue
        t = np.linspace(0.0, 1.0, n)
        lats = lat_a[i, None] + (lat_p - lat_a[i, None]) * t
        lons = lon_a[i, None] + (lon_p - lon_a[i, None]) * t
        z = _remplir_nan(dem.altitudes(lats, lons))
        s = dist_km[i, None] * 1000.0 * t
        terrain = z + s * (dist_km[i, None] * 1000.0 - s) / (2.0 * LOS_RE_EFFECTIVE)
        bout = z[:, -1:] + RECEIVER_HEIGHT_M
        for k, mat in enumerate(mats):
            hauteur = (sol_a[i, None] + mat) + (bout - (sol_a[i, None] + mat)) * t
            resultat[k, i] = np.all(terrain[:, 1:-1] <= hauteur[:, 1:-1], axis=1)
    return resultat


def calculer(env, dem=None):
    """Portée + LOS de chaque cellule du corridor vers chaque point blanc initial de env."""
    from environment_rl import BANDES_PAR_GENERATION
    dem = dem or DEM()
    generations = sorted(BANDES_PAR_GENERATION)
    mats = [MAT_HEIGHT_M.get(g, 35.0) for g in generations]
    portee_max = max(get_portee_km(g, f) for g in generations for f in BANDES_PAR_GENERATION[g])

    cellules = np.unique(env._cellule_action)
    centres = np.array([env.grille.cellule_vers_gps(c // env.grille.ny, c % env.grille.ny) for c in cellules])
    lat_c, lon_c = centres[:, 0], centres[:, 1]
    sol_c = _remplir_nan(dem.altitudes(lat_c, lon_c)[None, :])[0]

    pts = env.df_points.set_index("point_id")
    points = list(env._points_blancs)
    dist = np.zeros((len(cellules), len(points)), dtype=np.float32)
    los = np.zeros((len(generations), len(cellules), len(points)), dtype=bool)
    debut = time.time()
    for j, p in enumerate(points):
        d = haversine_km_vec(pts.at[p, "lat"], pts.at[p, "lon"], lat_c, lon_c)
        dist[:, j] = d
        proches = np.flatnonzero(d <= portee_max)
        los[:, proches, j] = los_profils(dem, lat_c[proches], lon_c[proches], sol_c[proches],
                                         pts.at[p, "lat"], pts.at[p, "lon"], d[proches], mats)
        if (j + 1) % 25 == 0:
            print(f"  couverture potentielle : {j + 1}/{len(points)} points, {time.time() - debut:.0f} s")
    return {"cellules": cellules, "points": np.array(points), "dist_km": dist,
            **{f"los_{g}": los[k] for k, g in enumerate(generations)}}


def charger_ou_calculer(env, chemin=CHEMIN_DEFAUT):
    if os.path.exists(chemin):
        d = np.load(chemin, allow_pickle=False)
        return {k: d[k] for k in d.files}
    print(f"Couverture potentielle absente ({chemin}) : calcul (une seule fois)")
    debut = time.time()
    resultat = calculer(env)
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    np.savez_compressed(chemin, **resultat)
    print(f"Couverture potentielle : {len(resultat['cellules'])} cellules x {len(resultat['points'])} points blancs, "
          f"{time.time() - debut:.0f} s, écrite dans {chemin}")
    return resultat


if __name__ == "__main__":
    from environment_rl import FerroMobileEnv
    env = FerroMobileEnv(scenario="S1", budget=10e6, t_max=150, sauvegarder_deploiements_rl=False)
    env.reset()
    charger_ou_calculer(env)
