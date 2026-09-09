"""
valider_dp_sur_donnees_reelles.py — Valide trajectory_dp() sur le vrai corridor
==================================================================================
Après les tests synthétiques (test_trajectory_dp.py), cette étape fait
tourner la DP sur les vraies données de la ligne Courpière-Ambert, un
scénario météo à la fois.

Une cellule réelle est identifiée par ((ant_lat, ant_lon), operateur,
generation, bande_mhz) : le site est représenté par ses coordonnées,
pas par ant_id (vérifié sur le dataset réel : 204 ant_id pour
seulement 29 coordonnées de site uniques, donc ant_id n'est pas un
site). Cette représentation est la même que celle utilisée par
simulateur_deploiement.py pour une cellule hypothétique, ce qui
permettra un jour de détecter qu'un ajout se fait au même site qu'une
cellule réelle (§13, upgrade vs nouveau site).

Lancer : python valider_dp_sur_donnees_reelles.py
"""

import os

import pandas as pd
from trajectory_dp import trajectory_dp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
DATASET    = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")
SCENARIO = "S1"
LAMBDA_H, LAMBDA_O, LAMBDA_T = 0.05, 0.05, 0.05


def construire_cells_per_point(df):
    """Regroupe le dataset (déjà filtré sur un scénario) par point_id,
    et construit la liste attendue par trajectory_dp().

    Le site est représenté par (ant_lat, ant_lon) arrondis, exactement
    comme simulateur_deploiement.py le fait pour une cellule hypothétique.
    C'est nécessaire pour pouvoir un jour comparer "même site" entre une
    cellule réelle et une cellule hypothétique (§13, upgrade vs nouveau
    site) : ant_id seul ne suffirait pas, ce n'est pas une coordonnée.
    """
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


def main():
    print("=" * 60)
    print(f"Validation de la DP — scénario {SCENARIO}")
    print("=" * 60)

    df = pd.read_csv(DATASET, low_memory=False)
    df = df[df["scenario_id"] == SCENARIO]

    # Hors tunnel, et seulement les points avec au moins une cellule réelle.
    # Les points orphelins (aucun candidat radio valide) ont ant_id vide
    # dans prepare_data.py, mais ça devient NaN une fois relu par pandas.
    # notna() les exclut correctement ; "!= ''" ne le faisait PAS (NaN != ""
    # vaut True en pandas, donc ce filtre les laissait passer par erreur).
    #
    # Convention (cahier des charges §6, §11) : ces points sans candidat
    # sont des zones blanches, comptées séparément dans N_white, PAS dans
    # cette DP. Quand deux points évaluables sont séparés par une ou
    # plusieurs zones blanches, la transition entre eux utilise la
    # pénalité standard lambda_H, sans pénalité supplémentaire pour la
    # coupure : choix volontaire, pour ne pas pénaliser deux fois la
    # même zone blanche (une fois via N_white, une fois via une
    # pénalité de coupure séparée).
    df = df[(~df["in_tunnel"]) & (df["ant_id"].notna())]

    print(f"[1] {df['point_id'].nunique()} points avec au moins une cellule candidate")

    points, cells_per_point = construire_cells_per_point(df)
    print(f"[2] Trajet construit : {len(points)} points")

    resultat = trajectory_dp(points, cells_per_point, LAMBDA_H, LAMBDA_O, LAMBDA_T)

    print(f"[3] J* = {resultat['J_star']:.2f}")
    print(f"    QoS moyenne = {resultat['qos_mean']:.4f}")
    print(f"    QoS pire cas = {resultat['qos_min']:.4f}")
    print(f"    Changements de cellule (N_H) = {resultat['N_H']}")
    print(f"    Changements d'opérateur (N_O) = {resultat['N_O']}")
    print(f"    Changements de technologie (N_T) = {resultat['N_T']}")

    # Vérifications de bon sens, pas des tests formels : juste pour
    # repérer immédiatement un résultat qui n'a pas de sens physique.
    assert 0.0 <= resultat["qos_mean"] <= 1.0, "qos_mean hors de [0,1] : incohérent"
    assert 0.0 <= resultat["qos_min"] <= 1.0, "qos_min hors de [0,1] : incohérent"
    assert resultat["qos_min"] <= resultat["qos_mean"], "le pire cas ne peut pas dépasser la moyenne"
    assert resultat["N_H"] <= len(points) - 1, "plus de changements que de transitions possibles"
    print("\n[4] Vérifications de bon sens : OK")

    # Diagnostic optionnel : où est le pire point de la trajectoire, et
    # avait-il une meilleure option disponible ce jour-là ?
    for t, cellule in enumerate(resultat["trajectory"]):
        for candidat in cells_per_point[t]:
            if candidat["cellule"] == cellule and candidat["qos"] == resultat["qos_min"]:
                print(f"\n[5] Pire point : point_id={points[t]}, cellule choisie={cellule}")
                print("    Autres cellules disponibles à ce point :")
                for autre in cells_per_point[t]:
                    print(f"      {autre['cellule']} -> qos={autre['qos']:.4f}")
                break
        else:
            continue
        break


if __name__ == "__main__":
    main()