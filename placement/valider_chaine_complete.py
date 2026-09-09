"""
valider_chaine_complete.py — Validation complète de la chaîne d'évaluation
===========================================================================
Teste la chaîne que le futur RL utilisera :

    (x, y, g, f)
        ↓
    simulation de l'infrastructure hypothétique
        ↓
    ajout aux cellules réelles
        ↓
    trajectory_dp
        ↓
    métriques de trajet
        ↓
    U(I)
        ↓
    coût du déploiement
        ↓
    r_fin

Le script compare l'état du réseau AVANT et APRÈS un déploiement
hypothétique fixé à la main.

Il valide ainsi, sur données réelles, les interfaces entre :
    - simulateur_deploiement.py
    - trajectory_dp.py
    - cost_model.py
    - utility.py

Le coût de l'infrastructure réelle initiale n'est pas comptabilisé :
on mesure uniquement le coût engagé par le déploiement testé, comme le
fera le futur environnement RL (qui ne décide que des ajouts).

cost_model n'entre JAMAIS dans U(I) ni dans r_t : uniquement dans le
coût utilisé par r_fin (§14 du cahier des charges).

Lancer :
    python valider_chaine_complete.py
"""

import os

import pandas as pd

from trajectory_dp import trajectory_dp
from simulateur_deploiement import simuler_deploiement
from cost_model import get_total_cost
from utility import (
    WEIGHTS_REFERENCE,
    normaliser_metriques,
    compute_U,
    compute_r_t,
    compute_r_fin,
)
from grid_config import construire_grille_depuis_dataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")
DATASET_ENRICHI = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")

SCENARIO = "S1"

# Pénalités utilisées par la DP (λ_H, λ_O, λ_T, §10). Ce ne sont PAS les
# poids de U(I) (w_H, w_O, w_T, §14) — deux jeux de poids distincts par
# construction (§11).
LAMBDA_H = 0.05
LAMBDA_O = 0.05
LAMBDA_T = 0.05

D_COV = 1.0

# Paramètres de la récompense terminale pour CETTE validation. Ce sont
# des paramètres de test, pas des valeurs imposées par le cahier des
# charges (§14 : testés par sensibilité, jamais figés a priori).
Q_CRITIQUE = 0.60
R_COUVERTURE = 10.0
R_QUALITE = 5.0
R_SUCCES = 10.0
R_EFF = 5.0

# Budget de test FIXE et réaliste, indépendant du coût du déploiement
# testé. Le fixer égal au coût de l'action rendrait Cost(I_T)/B = 1
# systématiquement, donc le terme d'efficacité de r_fin (R_eff*(1 -
# Cost(I_T)/B)) resterait toujours nul quel que soit R_eff — ça ne
# testerait jamais réellement cette partie de la formule.
BUDGET_TEST_EUR = 500_000

# Déploiement test.
GENERATION_TEST = "4G"
BANDE_TEST = 1800


def construire_cells_per_point(df):
    """Regroupe le réseau réel par point_id : les points avec au moins
    un candidat radio, ceux que la DP évalue (N_eval, §11). Le site est
    représenté par (ant_lat, ant_lon) arrondi, comme dans trajectory_dp.py
    — voir cost_model.py pour la normalisation séparée en cellule de
    grille, utilisée uniquement pour le calcul du coût."""
    points = sorted(df["point_id"].unique())
    cells_per_point = []
    for pid in points:
        lignes = df[df["point_id"] == pid]
        candidats = [
            {
                "cellule": ((round(row.ant_lat, 6), round(row.ant_lon, 6)),
                            row.operateur, row.generation, row.bande_mhz),
                "qos": row.qos,
            }
            for row in lignes.itertuples()
        ]
        cells_per_point.append(candidats)
    return points, cells_per_point


def ajouter_cellules_hypothetiques(points, cells_per_point, resultats_simulateur):
    """Fusionne les cellules hypothétiques dans cells_per_point. Un point
    qui était une zone blanche (absent de `points`) et qui reçoit une
    cellule hypothétique est AJOUTÉ à la séquence, pas seulement complété."""
    cellules_hypo_par_point = {r["point_id"]: r for r in resultats_simulateur}
    cellules_reelles_par_point = dict(zip(points, cells_per_point))
    tous_les_pid = sorted(set(points) | set(cellules_hypo_par_point.keys()))

    nouveaux_points, nouveau_cells_per_point = [], []
    for pid in tous_les_pid:
        candidats = []
        if pid in cellules_reelles_par_point:
            candidats.extend(cellules_reelles_par_point[pid])
        if pid in cellules_hypo_par_point:
            r = cellules_hypo_par_point[pid]
            candidats.append({"cellule": r["cellule"], "qos": r["qos"]})
        if candidats:
            nouveaux_points.append(pid)
            nouveau_cells_per_point.append(candidats)
    return nouveaux_points, nouveau_cells_per_point


def compter_zones_blanches(meilleur_debit_par_point, points_hors_tunnel, D_cov=D_COV):
    """Nombre de points hors tunnel où AUCUN candidat n'atteint D_cov Mbps."""
    en_zone_blanche = meilleur_debit_par_point.reindex(list(points_hors_tunnel))
    return int((en_zone_blanche.fillna(0) < D_cov).sum())


def calculer_metriques_et_utility(resultat_dp, n_eval, n_white, n_total):
    """Construit les métriques normalisées (§11) puis U(I) (§14)."""
    metriques = normaliser_metriques(
        resultat_dp=resultat_dp, n_eval=n_eval, n_white=n_white, n_total=n_total)
    poids = WEIGHTS_REFERENCE
    U = compute_U(metriques, poids["w_Q"], poids["w_W"], poids["w_H"], poids["w_O"], poids["w_T"])
    return metriques, U


def afficher_etat(titre, resultat, metriques, U, n_eval, n_white, n_total):
    print(f"\n{'=' * 60}\n{titre}\n{'=' * 60}")
    print(f"J*                : {resultat['J_star']:.2f}")
    print(f"QoS moyenne       : {resultat['qos_mean']:.4f}")
    print(f"QoS pire cas      : {resultat['qos_min']:.4f}")
    print(f"N_H               : {resultat['N_H']}")
    print(f"N_O               : {resultat['N_O']}")
    print(f"N_T               : {resultat['N_T']}")
    print(f"N_eval            : {n_eval}")
    print(f"N_total           : {n_total}")
    print(f"N_white           : {n_white}")
    print("\nMétriques normalisées :")
    print(f"Q_hat             : {metriques['Q']:.4f}")
    print(f"W_hat             : {metriques['W']:.4f}")
    print(f"H_hat             : {metriques['H']:.4f}")
    print(f"O_hat             : {metriques['O']:.4f}")
    print(f"T_hat             : {metriques['T']:.4f}")
    print(f"\nU(I)              : {U:.4f}")


def construire_sites_existants(df_reel, grille):
    """Ensemble des sites existants, sous forme de cellules de grille —
    la représentation attendue par cost_model.py (§13 : même cellule de
    grille = même site, pas égalité de coordonnées GPS exactes)."""
    sites = set()
    for row in df_reel[["ant_lat", "ant_lon"]].drop_duplicates().itertuples(index=False):
        cellule = grille.gps_vers_cellule(row.ant_lat, row.ant_lon)
        if cellule is None:
            continue  # site réel hors grille : ne devrait jamais arriver
            # (vérifié 0/29 dans grid_config.py), mais on ne l'ajoute pas
            # à sites_existants plutôt que de planter silencieusement.
        sites.add(cellule)
    return sites


def main():
    # ================================================================
    # 1. CHARGEMENT
    # ================================================================
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    df_enrichi = pd.read_csv(DATASET_ENRICHI, low_memory=False)
    df_enrichi = df_enrichi[df_enrichi["scenario_id"] == SCENARIO]

    df_points = df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")
    points_hors_tunnel = set(df_points[~df_points["in_tunnel"]]["point_id"])
    n_total = len(points_hors_tunnel)

    # ================================================================
    # 2. GRILLE — construite depuis dataset_base.csv, PAS GrilleCorridor()
    #    à vide (le constructeur exige points_trajet_lat_lon).
    # ================================================================
    grille = construire_grille_depuis_dataset(df_base)

    # ================================================================
    # 3. RÉSEAU RÉEL INITIAL
    # ================================================================
    df_reel = df_enrichi[(~df_enrichi["in_tunnel"]) & (df_enrichi["ant_id"].notna())]
    points_avant, cells_avant = construire_cells_per_point(df_reel)
    debit_reel_par_point = df_enrichi[~df_enrichi["in_tunnel"]].groupby("point_id")["debit_adj_mbps"].max()

    # ================================================================
    # 4. ÉTAT AVANT
    # ================================================================
    resultat_avant = trajectory_dp(points_avant, cells_avant, LAMBDA_H, LAMBDA_O, LAMBDA_T)
    n_white_avant = compter_zones_blanches(debit_reel_par_point, points_hors_tunnel, D_COV)
    n_eval_avant = len(points_avant)
    metriques_avant, U_avant = calculer_metriques_et_utility(
        resultat_avant, n_eval_avant, n_white_avant, n_total)
    afficher_etat("ÉTAT INITIAL (réseau réel)", resultat_avant, metriques_avant,
                  U_avant, n_eval_avant, n_white_avant, n_total)

    # ================================================================
    # 5. CHOIX D'UNE ZONE BLANCHE POUR LE TEST
    # ================================================================
    en_zone_blanche = debit_reel_par_point.reindex(list(points_hors_tunnel)).fillna(0) < D_COV
    zones_blanches = sorted(en_zone_blanche[en_zone_blanche].index)
    if not zones_blanches:
        raise RuntimeError("Aucune zone blanche trouvée, impossible de choisir une cible pertinente.")

    point_cible = df_points[df_points["point_id"] == zones_blanches[len(zones_blanches) // 2]].iloc[0]
    lat_test, lon_test = point_cible["lat"], point_cible["lon"]

    # ================================================================
    # 6. SIMULATION DU DÉPLOIEMENT
    # ================================================================
    resultats_simulateur = simuler_deploiement(lat_test, lon_test, GENERATION_TEST, BANDE_TEST, df_points)
    # simuler_deploiement() exclut déjà les tunnels en interne (in_tunnel
    # -> continue) ; ce filtre est donc redondant mais laissé par sécurité,
    # au cas où l'implémentation du simulateur changerait plus tard.
    resultats_simulateur = [r for r in resultats_simulateur if r["point_id"] in points_hors_tunnel]

    print(f"\n{'=' * 60}\nDÉPLOIEMENT TEST\n{'=' * 60}")
    print(f"Site              : ({lat_test:.4f}, {lon_test:.4f})")
    print(f"Génération        : {GENERATION_TEST}")
    print(f"Bande             : {BANDE_TEST} MHz")
    print(f"Points évalués    : {len(resultats_simulateur)}")

    # ================================================================
    # 7. COÛT DU DÉPLOIEMENT (§12) — UNIQUEMENT pour r_fin, jamais U(I)
    # ================================================================
    cellule_site_test = grille.gps_vers_cellule(lat_test, lon_test)
    if cellule_site_test is None:
        raise RuntimeError("Le point testé tombe hors de la grille — ne devrait pas arriver "
                            "pour un point réel du trajet (vérifié 0/1923 dans grid_config.py).")

    sites_existants = construire_sites_existants(df_reel, grille)
    cout_action = get_total_cost(
        site=cellule_site_test, generation=GENERATION_TEST, bande_mhz=BANDE_TEST,
        sites_deja_presents=sites_existants)
    nouveau_site = cellule_site_test not in sites_existants

    print("\nCoût du déploiement :")
    print(f"Cellule de grille : {cellule_site_test}")
    print(f"Site existant     : {'oui' if not nouveau_site else 'non'}")
    print(f"Coût engagé       : {cout_action:,.0f} €")

    # ================================================================
    # 8. ÉTAT APRÈS
    # ================================================================
    points_apres, cells_apres = ajouter_cellules_hypothetiques(points_avant, cells_avant, resultats_simulateur)
    resultat_apres = trajectory_dp(points_apres, cells_apres, LAMBDA_H, LAMBDA_O, LAMBDA_T)

    debit_hypo_par_point = pd.Series({r["point_id"]: r["debit_adj_mbps"] for r in resultats_simulateur})
    debit_apres_par_point = pd.concat([debit_reel_par_point, debit_hypo_par_point], axis=1).max(axis=1)
    n_white_apres = compter_zones_blanches(debit_apres_par_point, points_hors_tunnel, D_COV)
    n_eval_apres = len(points_apres)

    metriques_apres, U_apres = calculer_metriques_et_utility(
        resultat_apres, n_eval_apres, n_white_apres, n_total)
    afficher_etat("ÉTAT APRÈS AJOUT", resultat_apres, metriques_apres,
                  U_apres, n_eval_apres, n_white_apres, n_total)

    # ================================================================
    # 9. r_t = U(I_{t+1}) - U(I_t) (§14) — via compute_r_t(), pas de
    #    soustraction manuelle : une seule source de vérité par formule.
    # ================================================================
    r_t = compute_r_t(U_apres, U_avant)

    # ================================================================
    # 10. r_fin (§14) — budget FIXE, indépendant du coût de l'action
    #     testée (voir BUDGET_TEST_EUR ci-dessus pour la justification).
    # ================================================================
    r_fin = compute_r_fin(
        n_white=n_white_apres, qos_min=resultat_apres["qos_min"],
        cost_total=cout_action, budget=BUDGET_TEST_EUR,
        Q_critique=Q_CRITIQUE, R_couverture=R_COUVERTURE, R_qualite=R_QUALITE,
        R_succes=R_SUCCES, R_eff=R_EFF)

    # ================================================================
    # 11. IMPACT FINAL
    # ================================================================
    print(f"\n{'=' * 60}\nIMPACT DU DÉPLOIEMENT\n{'=' * 60}")
    print(f"ΔJ*               : {resultat_apres['J_star'] - resultat_avant['J_star']:+.2f}")
    print(f"ΔQoS moyenne      : {resultat_apres['qos_mean'] - resultat_avant['qos_mean']:+.4f}")
    print(f"ΔQoS pire cas     : {resultat_apres['qos_min'] - resultat_avant['qos_min']:+.4f}")
    print(f"ΔN_H              : {resultat_apres['N_H'] - resultat_avant['N_H']:+d}")
    print(f"ΔN_O              : {resultat_apres['N_O'] - resultat_avant['N_O']:+d}")
    print(f"ΔN_T              : {resultat_apres['N_T'] - resultat_avant['N_T']:+d}")
    print(f"ΔN_white          : {n_white_apres - n_white_avant:+d}")
    print(f"ΔU = r_t          : {r_t:+.4f}")
    print(f"Budget de test    : {BUDGET_TEST_EUR:,.0f} €")
    print(f"r_fin             : {r_fin:+.4f}")
    print(f"Retour total test : {r_t + r_fin:+.4f}")


if __name__ == "__main__":
    main()