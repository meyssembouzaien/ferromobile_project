"""
valider_simulateur.py — Valide simuler_deploiement() sur les vraies données
==============================================================================
Deux vérifications :

1. Cohérence : re-simuler une antenne RÉELLE existante, avec exactement
   ses propres coordonnées/génération/bande, doit redonner (à peu près)
   les mêmes débit/RTT/QoS que ce que prepare_data.py a déjà calculé
   pour elle. Si ce n'est pas le cas, le simulateur ne réutilise pas
   correctement la même physique.

2. Nouveau site : simuler une antenne à une position qui n'existe pas
   dans le réseau actuel doit produire des résultats non vides et
   raisonnables (le test de faisabilité central du projet, §3 du
   cahier des charges : peut-on simuler une antenne hypothétique ?).

Lancer : python valider_simulateur.py
"""

import os

import pandas as pd

from simulateur_deploiement import simuler_deploiement, qos_nominal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")


def test_coherence_antenne_reelle(df_base):
    print("=" * 60)
    print("[1] Cohérence avec une antenne réelle existante")
    print("=" * 60)

    # On prend l'antenne réelle avec le plus de liens valides, pour
    # avoir plusieurs points de comparaison.
    reelles = df_base[df_base["ant_id"].notna() & (~df_base["in_tunnel"])]
    meilleur_ant = reelles["ant_id"].value_counts().idxmax()
    lignes_ant = reelles[reelles["ant_id"] == meilleur_ant]
    premiere = lignes_ant.iloc[0]

    print(f"Antenne testée : {meilleur_ant} "
          f"({premiere['generation']}, {premiere['bande_mhz']} MHz), "
          f"{len(lignes_ant)} liens dans le dataset original")

    df_points = df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")

    resimule = simuler_deploiement(
        premiere["ant_lat"], premiere["ant_lon"],
        premiere["generation"], premiere["bande_mhz"],
        df_points)

    print(f"Liens retrouvés par le simulateur : {len(resimule)}")

    # Compare les points communs aux deux
    resimule_par_point = {r["point_id"]: r for r in resimule}
    points_communs = set(lignes_ant["point_id"]) & set(resimule_par_point.keys())
    print(f"Points en commun : {len(points_communs)} / {len(lignes_ant)} liens originaux")

    if points_communs:
        ecarts_qos = []
        for pid in list(points_communs)[:5]:
            ligne_orig = lignes_ant[lignes_ant["point_id"] == pid].iloc[0]
            qos_original = qos_nominal(ligne_orig["debit_adjusted"], ligne_orig["rtt_ms"], ligne_orig["ber"], ligne_orig["generation"])
            qos_resimule = resimule_par_point[pid]["qos"]
            ecarts_qos.append(abs(qos_original - qos_resimule))
            print(f"  point {pid} : qos original={qos_original:.4f}  qos re-simulé={qos_resimule:.4f}")
        ecart_moyen = sum(ecarts_qos) / len(ecarts_qos)
        print(f"\nÉcart moyen sur l'échantillon : {ecart_moyen:.4f}")
        if ecart_moyen < 0.01:
            print("-> OK, le simulateur reproduit fidèlement la physique existante.")
        else:
            print("-> ÉCART SUSPECT, à investiguer avant de continuer.")
    else:
        print("-> ATTENTION : aucun point en commun, le simulateur ne retrouve pas l'antenne réelle.")


def test_nouveau_site(df_base):
    print("\n" + "=" * 60)
    print("[2] Nouveau site (position hypothétique, aucune antenne réelle ici)")
    print("=" * 60)

    # Un point du milieu du trajet, décalé de 2 km, comme position candidate.
    df_points = df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")
    point_milieu = df_points.iloc[len(df_points) // 2]
    lat_test = point_milieu["lat"] + 0.01   # ~1 km au nord
    lon_test = point_milieu["lon"]

    resultat = simuler_deploiement(lat_test, lon_test, "4G", 1800, df_points)

    print(f"Position testée : ({lat_test:.4f}, {lon_test:.4f})")
    print(f"Liens produits : {len(resultat)}")
    if resultat:
        qos_moy = sum(r["qos"] for r in resultat) / len(resultat)
        print(f"QoS moyenne sur ces liens : {qos_moy:.4f}")
        print("-> OK, une antenne hypothétique à une position nouvelle produit bien des résultats.")
    else:
        print("-> Aucun lien produit ici (peut-être normal si la position est trop isolée, à vérifier).")


if __name__ == "__main__":
    df_base = pd.read_csv(DATASET_BASE, low_memory=False)
    test_coherence_antenne_reelle(df_base)
    test_nouveau_site(df_base)
