"""
verifier_points_incouvrables.py — n_white=0 est-il même atteignable ?

Teste les 11 configurations (génération, bande) une par une, sur les
points déjà identifiés comme résistants au 3G/900 même en ciblage
individuel dédié (diagnostiquer_budget.py, run précédent). Si un point
résiste aux 11 combos, c'est un point structurellement incouvrable —
zone morte physique (terrain/NLOS), pas un problème de RL ni de budget.

Lancer : python verifier_points_incouvrables.py
"""

import math
import os

import pandas as pd

from simulateur_deploiement import simuler_deploiement
from prepare_data import TECH_PROFILES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")
DATASET_ENRICHI = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")

SCENARIO = "S1"
D_COV = 1.0
DISTANCE_ANTENNE_M = 100
DIRECTIONS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 directions, pas une seule

# Points qui ont échoué aux 11 technos EN UNE SEULE direction (45°) —
# on vérifie ici si c'est un vrai obstacle physique (échoue dans TOUTES
# les directions) ou juste cette direction précise qui était bloquée.
POINTS_A_TESTER = [301, 388, 391, 401, 413, 441, 455, 1195]


def decaler_gps(lat, lon, distance_m, direction_deg):
    rayon_terre_m = 6_371_000.0
    angle_rad = math.radians(direction_deg)
    dlat = (distance_m * math.cos(angle_rad)) / rayon_terre_m
    dlon = (distance_m * math.sin(angle_rad)) / (rayon_terre_m * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def main():
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    df_enrichi = pd.read_csv(DATASET_ENRICHI, low_memory=False)
    df_scenario = df_enrichi[df_enrichi["scenario_id"] == SCENARIO]

    df_points = df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")
    df_meteo = df_scenario[["point_id", "pluie_mm_h"]].drop_duplicates("point_id")

    combos = sorted(TECH_PROFILES.keys())
    print(f"Test de {len(POINTS_A_TESTER)} points x {len(DIRECTIONS_DEG)} directions x {len(combos)} configurations radio\n")

    points_totalement_incouvrables = []

    for pid in POINTS_A_TESTER:
        point = df_points[df_points["point_id"] == pid]
        if point.empty:
            print(f"point_id={pid} : introuvable dans le dataset, ignoré")
            continue
        point = point.iloc[0]

        meilleur_debit = 0.0
        meilleure_config = None
        for direction in DIRECTIONS_DEG:
            lat_ant, lon_ant = decaler_gps(point["lat"], point["lon"], DISTANCE_ANTENNE_M, direction)
            for gen, bande in combos:
                resultats = simuler_deploiement(
                    lat_ant, lon_ant, gen, bande, df_points, df_meteo,
                )
                cible = [r for r in resultats if r["point_id"] == pid]
                if cible:
                    debit = cible[0]["debit_adj_mbps"]
                    if debit > meilleur_debit:
                        meilleur_debit, meilleure_config = debit, (gen, bande, f"{direction}°")

        couvert = meilleur_debit >= D_COV
        statut = f"COUVERT par {meilleure_config} ({meilleur_debit:.2f} Mbps)" if couvert \
            else f"INCOUVRABLE dans les {len(DIRECTIONS_DEG)} directions testées (meilleur essai : {meilleur_debit:.2f} Mbps)"
        print(f"point_id={pid} : {statut}")

        if not couvert:
            points_totalement_incouvrables.append(pid)

    print(f"\n{'=' * 60}")
    if points_totalement_incouvrables:
        print(f"{len(points_totalement_incouvrables)}/{len(POINTS_A_TESTER)} points RÉSISTENT "
              f"aux 11 technologies, même en ciblage individuel dédié :")
        print(points_totalement_incouvrables)
        print("\n-> n_white=0 est probablement INATTEIGNABLE avec l'espace d'action actuel.")
        print("   Ce ne sont pas des points isolés testés au hasard : ce sont les points")
        print("   qui résistaient déjà au 3G/900 seul. S'ils résistent aussi aux 10 autres")
        print("   technologies, c'est une zone morte physique (terrain/NLOS structurel),")
        print("   pas un problème de politique RL ni de budget.")
    else:
        print("Tous les points testés sont couvrables par au moins une technologie.")
        print("n_white=0 reste théoriquement atteignable avec la bonne combinaison —")
        print("le problème est alors de trouver quelle techno utiliser où, ce qui est")
        print("exactement ce que le RL doit apprendre à faire.")


if __name__ == "__main__":
    main()