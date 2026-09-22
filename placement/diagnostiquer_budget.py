"""
heuristique_gloutonne_multitech.py — Baseline "couverture gloutonne" du §20

Contrairement à diagnostiquer_budget.py (une seule techno fixée par run),
choisit à CHAQUE étape la meilleure combinaison (position, génération,
bande) parmi toutes les options disponibles, selon le critère du §20 :
réduit le plus de zones blanches par euro dépensé.

Pour rester calculable (pas 156061 candidats testés à chaque étape),
les positions candidates à une étape donnée sont limitées aux points
encore blancs eux-mêmes, décalés de quelques mètres dans plusieurs
directions (même logique que verifier_points_incouvrables.py — jamais
pile sur le point, cas dégénéré P.1812). Ça reste un choix de conception
qui pourrait manquer une position légèrement meilleure ailleurs sur la
grille, mais donne un ordre de grandeur fiable et un test direct de
"zéro est-il atteignable en pratique, avec plusieurs technos ?".

Lancer : python heuristique_gloutonne_multitech.py
"""

import math
import os

import pandas as pd

from simulateur_deploiement import simuler_deploiement
from cost_model import get_total_cost
from prepare_data import TECH_PROFILES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")
DATASET_ENRICHI = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")

SCENARIO = "S1"
D_COV = 1.0
BUDGET_MAX = 10_000_000.0  # même budget que le RL, pour une comparaison équitable
DISTANCE_ANTENNE_M = 100
DIRECTIONS_DEG = [0, 180]  # réduit à 2 directions pour un premier test de vitesse — remets [0, 90, 180, 270] une fois la vitesse réelle connue
GENERATIONS_EXCLUES = {"2G"}  # même exclusion que l'espace d'action RL (§10, jamais couvrant)


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
    points_hors_tunnel = set(df_points[~df_points["in_tunnel"]]["point_id"])
    n_total = len(points_hors_tunnel)

    debit_reel = df_scenario[~df_scenario["in_tunnel"]].groupby("point_id")["debit_adj_mbps"].max()
    debit_courant = debit_reel.reindex(list(points_hors_tunnel)).fillna(0.0)

    combos = sorted((g, b) for g, b in TECH_PROFILES.keys() if g not in GENERATIONS_EXCLUES)

    sites_deja_presents = set()  # simplifié : traite tout comme un nouveau site (coût pessimiste, cohérent avec diagnostiquer_budget.py)
    budget_restant = BUDGET_MAX
    cout_total = 0.0
    n_sites = 0

    print(f"Départ : n_white={int((debit_courant < D_COV).sum())} / {n_total}, budget={BUDGET_MAX:.0f}€\n", flush=True)

    while True:
        points_blancs = df_points[
            df_points["point_id"].isin(points_hors_tunnel)
            & (df_points["point_id"].map(debit_courant) < D_COV)
        ]
        n_white = len(points_blancs)
        if n_white == 0:
            print(f"\n>>> ZÉRO ZONE BLANCHE ATTEINTE : {n_sites} sites, {cout_total:.0f}€ dépensés <<<", flush=True)
            break

        print(f"  (évaluation de {n_white} points blancs x {len(DIRECTIONS_DEG)} directions x "
              f"{len(combos)} technos = {n_white * len(DIRECTIONS_DEG) * len(combos)} combinaisons "
              f"pour ce site...)", flush=True)

        meilleur_ratio = 0.0
        meilleure_action = None  # (lat, lon, gen, bande, cout, points_couverts_set)

        for i, point in enumerate(points_blancs.itertuples()):
            if i % 20 == 0:  # heartbeat toutes les 20 points, pour prouver que ça calcule
                print(f"    ... point {i}/{n_white} en cours d'évaluation", flush=True)
            for direction in DIRECTIONS_DEG:
                lat_ant, lon_ant = decaler_gps(point.lat, point.lon, DISTANCE_ANTENNE_M, direction)
                for gen, bande in combos:
                    cout = get_total_cost((round(lat_ant, 4), round(lon_ant, 4)), gen, bande, sites_deja_presents)
                    if cout > budget_restant:
                        continue
                    resultats = simuler_deploiement(lat_ant, lon_ant, gen, bande, df_points, df_meteo)
                    nouveaux_couverts = {
                        r["point_id"] for r in resultats
                        if r["point_id"] in points_hors_tunnel
                        and r["debit_adj_mbps"] >= D_COV
                        and debit_courant.get(r["point_id"], 0.0) < D_COV
                    }
                    if not nouveaux_couverts:
                        continue
                    ratio = len(nouveaux_couverts) / cout
                    if ratio > meilleur_ratio:
                        meilleur_ratio = ratio
                        meilleure_action = (lat_ant, lon_ant, gen, bande, cout, nouveaux_couverts)

        if meilleure_action is None:
            print(f"\n>>> BLOQUÉ : aucune action améliorante trouvée dans le budget restant "
                  f"({budget_restant:.0f}€). n_white={n_white} restant. <<<", flush=True)
            break

        lat_ant, lon_ant, gen, bande, cout, nouveaux_couverts = meilleure_action
        for pid in nouveaux_couverts:
            debit_courant[pid] = D_COV  # marqué couvert (valeur exacte non nécessaire pour ce diagnostic)
        sites_deja_presents.add((round(lat_ant, 4), round(lon_ant, 4)))
        budget_restant -= cout
        cout_total += cout
        n_sites += 1

        print(f"  site {n_sites} : {gen}/{bande} -> {len(nouveaux_couverts)} points couverts, "
              f"{cout:.0f}€, budget restant={budget_restant:.0f}€, n_white={n_white - len(nouveaux_couverts)}", flush=True)

        if budget_restant <= 0:
            print(f"\n>>> BUDGET ÉPUISÉ : {n_sites} sites, n_white={n_white - len(nouveaux_couverts)} restant <<<", flush=True)
            break


if __name__ == "__main__":
    main()