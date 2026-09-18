"""
diagnostiquer_distance_zero.py — pourquoi une antenne pile sur un point
ne le couvre pas.

Reproduit exactement _path_loss_p1812() de prepare_data.py, mais SANS
le filtre de sécurité (50 < pl < 280) qui rejette le résultat — pour
voir la vraie valeur calculée avant rejet, sur un point précis à
distance quasi nulle.

Lancer : python diagnostiquer_distance_zero.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datapreparation"))

from prepare_data import (
    covlib, _CrcSim, _P1812_CLUTTER, _TERR_ELEV_SRTM3, _TERRAIN_AVAILABLE,
    TERRAIN_DATA_DIR, MAT_HEIGHT_M, RECEIVER_HEIGHT_M,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")

POINT_ID_TEST = 164  # a échoué dans les deux runs précédents (3G et 4G)


def path_loss_brut(pt_lat, pt_lon, ant_lat, ant_lon, ant_height_agl_m, freq_mhz, vegetation):
    """Copie de _path_loss_p1812(), SANS le filtre 50<pl<280 — pour voir
    la vraie valeur avant qu'elle soit rejetée."""
    sim = _CrcSim()
    sim.SetTransmitterLocation(ant_lat, ant_lon)
    sim.SetTransmitterHeight(ant_height_agl_m)
    sim.SetTransmitterFrequency(freq_mhz)
    sim.SetReceiverHeightAboveGround(RECEIVER_HEIGHT_M)
    if _TERR_ELEV_SRTM3:
        try:
            sim.SetPrimaryTerrainElevDataSource(_TERR_ELEV_SRTM3)
            if _TERRAIN_AVAILABLE:
                sim.SetTerrainElevDataSourceDirectory(TERRAIN_DATA_DIR)
        except Exception:
            pass
    clutter = _P1812_CLUTTER.get(vegetation)
    if clutter is not None:
        try:
            sim.SetDefaultLandCoverClassMapping(clutter)
        except Exception:
            pass
    sim.SetPropagationModel(covlib.PropagationModel.ITU_R_P_1812)
    try:
        sim.SetITURP1812TimePercentage(50)
        sim.SetITURP1812LocationPercentage(50)
    except Exception:
        pass
    path_loss = sim.GenerateReceptionPointResult(pt_lat, pt_lon)
    return path_loss  # brut, aucun filtre


def main():
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    df_points = df_base[["point_id", "lat", "lon", "vegetation"]].drop_duplicates("point_id")
    point = df_points[df_points["point_id"] == POINT_ID_TEST].iloc[0]

    print(f"Point testé : point_id={POINT_ID_TEST}, lat={point['lat']}, lon={point['lon']}, "
          f"végétation={point['vegetation']}\n")

    for gen, freq, hauteur in [("3G", 900, MAT_HEIGHT_M["3G"]), ("4G", 700, MAT_HEIGHT_M["4G"])]:
        pl = path_loss_brut(
            point["lat"], point["lon"], point["lat"], point["lon"],
            hauteur, freq, point["vegetation"],
        )
        if pl is None:
            print(f"{gen}/{freq} : path_loss = None (crc-covlib lui-même a échoué, pas notre filtre)")
        else:
            rejete = pl <= 50.0 or pl > 280.0
            print(f"{gen}/{freq} : path_loss = {pl:.2f} dB "
                  f"{'-- REJETÉ par le filtre 50<pl<280' if rejete else '-- accepté normalement'}")

    # Même test, mais à quelques distances RÉALISTES pour le RL (le RL
    # place les antennes au centre d'une cellule de grille 250m, jamais
    # pile sur un point de trajet) — pour voir si le path_loss délirant
    # est spécifique à distance~0, ou si ça reste mauvais un peu plus loin.
    print("\n--- Même point, mais à des distances réalistes (pas distance=0) ---")
    import math as _math

    def decaler_gps(lat, lon, distance_m, direction_deg=45.0):
        """Décale un point de distance_m mètres dans une direction donnée."""
        rayon_terre_m = 6_371_000.0
        angle_rad = _math.radians(direction_deg)
        dlat = (distance_m * _math.cos(angle_rad)) / rayon_terre_m
        dlon = (distance_m * _math.sin(angle_rad)) / (rayon_terre_m * _math.cos(_math.radians(lat)))
        return lat + _math.degrees(dlat), lon + _math.degrees(dlon)

    for distance_m in [50, 100, 200, 500]:
        lat_ant, lon_ant = decaler_gps(point["lat"], point["lon"], distance_m)
        pl = path_loss_brut(
            point["lat"], point["lon"], lat_ant, lon_ant,
            MAT_HEIGHT_M["4G"], 700, point["vegetation"],
        )
        pl_str = f"{pl:.2f} dB" if pl is not None else "None"
        print(f"4G/700 à {distance_m}m : path_loss = {pl_str}")


if __name__ == "__main__":
    main()