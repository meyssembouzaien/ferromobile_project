"""
heuristique_gloutonne_multitech.py — Baseline "couverture gloutonne" du §20

Choisit à CHAQUE étape la meilleure combinaison (position, génération,
bande) parmi toutes les options disponibles, selon le critère du §20 :
réduit le plus de zones blanches par euro dépensé.

CORRIGÉ (bug trouvé après le premier run) : les candidats sont maintenant
des cellules de la grille RL (ix,iy), pas des coordonnées GPS arbitraires
décalées de 100m. Deux raisons :
  1. cost_model.get_total_cost() attend un identifiant de cellule de
     grille pour reconnaître un site déjà existant (voir son docstring),
     pas des lat/lon arrondis — l'ancienne version violait ce contrat,
     donc TOUS les déploiements étaient comptés comme "nouveau site"
     (145 000€ systématiquement), même les 29 sites réels ignorés.
  2. cellule_vers_gps() place toujours au CENTRE d'une cellule de 250m,
     jamais pile sur un point de trajet — élimine aussi le besoin du
     décalage par direction (plus de cas dégénéré P.1812 possible).

Candidats limités aux cellules contenant un point encore blanc (pas les
15606 cellules du corridor à chaque étape) — reste calculable, cohérent
avec le principe d'un glouton orienté couverture.

Lancer : python heuristique_gloutonne_multitech.py
"""

import os

import pandas as pd

from grid_config import construire_grille_depuis_dataset
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
GENERATIONS_EXCLUES = {"2G"}  # même exclusion que l'espace d'action RL (§10, jamais couvrant)


def main():
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    df_enrichi = pd.read_csv(DATASET_ENRICHI, low_memory=False)
    df_scenario = df_enrichi[df_enrichi["scenario_id"] == SCENARIO]

    grille = construire_grille_depuis_dataset(df_base)

    df_points = df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")
    df_meteo = df_scenario[["point_id", "pluie_mm_h"]].drop_duplicates("point_id")
    points_hors_tunnel = set(df_points[~df_points["in_tunnel"]]["point_id"])
    n_total = len(points_hors_tunnel)

    debit_reel = df_scenario[~df_scenario["in_tunnel"]].groupby("point_id")["debit_adj_mbps"].max()
    debit_courant = debit_reel.reindex(list(points_hors_tunnel)).fillna(0.0)

    combos = sorted((g, b) for g, b in TECH_PROFILES.keys() if g not in GENERATIONS_EXCLUES)

    # Sites réels, initialisés EXACTEMENT comme environment_rl.py.reset()
    # — sinon l'heuristique ignore les 29 sites existants et paie C_site
    # inutilement en repassant par ces mêmes emplacements.
    df_reel = df_scenario[(~df_scenario["in_tunnel"]) & (df_scenario["ant_id"].notna())]
    antennes_uniques = df_reel[["ant_id", "ant_lat", "ant_lon"]].drop_duplicates("ant_id")
    sites_deja_presents = set()
    for row in antennes_uniques.itertuples():
        c = grille.gps_vers_cellule(row.ant_lat, row.ant_lon)
        if c is not None:
            sites_deja_presents.add(c)

    budget_restant = BUDGET_MAX
    cout_total = 0.0
    n_sites = 0

    print(f"Départ : n_white={int((debit_courant < D_COV).sum())} / {n_total}, "
          f"budget={BUDGET_MAX:.0f}€, {len(sites_deja_presents)} sites réels connus\n", flush=True)

    while True:
        points_blancs = df_points[
            df_points["point_id"].isin(points_hors_tunnel)
            & (df_points["point_id"].map(debit_courant) < D_COV)
        ]
        n_white = len(points_blancs)
        if n_white == 0:
            print(f"\n>>> ZÉRO ZONE BLANCHE ATTEINTE : {n_sites} sites, {cout_total:.0f}€ dépensés <<<", flush=True)
            break

        # Cellules candidates : celles du corridor contenant au moins un
        # point encore blanc — dédoublonnées (plusieurs points blancs
        # peuvent partager la même cellule de 250m).
        candidats_cellules = set()
        for point in points_blancs.itertuples():
            c = grille.gps_vers_cellule(point.lat, point.lon)
            if c is not None and grille.corridor_mask[c[0], c[1]]:
                candidats_cellules.add(c)

        print(f"  ({n_white} points blancs -> {len(candidats_cellules)} cellules candidates x "
              f"{len(combos)} technos = {len(candidats_cellules) * len(combos)} combinaisons "
              f"pour ce site...)", flush=True)

        meilleur_ratio = 0.0
        meilleure_action = None  # (ix, iy, gen, bande, cout, points_couverts_set)

        for i, (ix, iy) in enumerate(candidats_cellules):
            if i % 20 == 0:
                print(f"    ... cellule {i}/{len(candidats_cellules)} en cours d'évaluation", flush=True)
            lat_ant, lon_ant = grille.cellule_vers_gps(ix, iy)
            for gen, bande in combos:
                cout = get_total_cost((ix, iy), gen, bande, sites_deja_presents)
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
                    meilleure_action = (ix, iy, gen, bande, cout, nouveaux_couverts)

        if meilleure_action is None:
            print(f"\n>>> BLOQUÉ : aucune action améliorante trouvée dans le budget restant "
                  f"({budget_restant:.0f}€). n_white={n_white} restant. <<<", flush=True)
            break

        ix, iy, gen, bande, cout, nouveaux_couverts = meilleure_action
        for pid in nouveaux_couverts:
            debit_courant[pid] = D_COV  # marqué couvert (valeur exacte non nécessaire pour ce diagnostic)
        sites_deja_presents.add((ix, iy))
        budget_restant -= cout
        cout_total += cout
        n_sites += 1

        print(f"  site {n_sites} : cellule ({ix},{iy}), {gen}/{bande} -> {len(nouveaux_couverts)} points couverts, "
              f"{cout:.0f}€, budget restant={budget_restant:.0f}€, n_white={n_white - len(nouveaux_couverts)}", flush=True)

        if budget_restant <= 0:
            print(f"\n>>> BUDGET ÉPUISÉ : {n_sites} sites, n_white={n_white - len(nouveaux_couverts)} restant <<<", flush=True)
            break


if __name__ == "__main__":
    main()