"""
simulateur_deploiement.py — Simule l'ajout d'une infrastructure hypothétique
==============================================================================
Prend un déploiement décidé par le RL (x, y, generation, bande) et calcule
ce qu'il produirait réellement le long du trajet, en réutilisant EXACTEMENT
le même modèle physique que prepare_data.py (P.1812, LOS, débit empirique).
Rien n'est réimplémenté séparément pour rester cohérent avec le réseau réel.

Chaîne (cahier des charges §2, §13) :

    (x,y,g,f) -> antenne hypothétique -> P.1812 -> SNR -> débit/RTT
              -> QoS nominale -> cellule hypothétique c_RL=(site,None,g,f)

Cette cellule peut ensuite être ajoutée à cells_per_point et passée à
trajectory_dp() exactement comme une cellule réelle.

Le site est identifié par ses coordonnées (lat, lon) directement, jamais
par un ant_id : pour une amélioration de site existant, on passe les
coordonnées exactes du site déjà là ; pour un nouveau site, de nouvelles
coordonnées. C'est ce qui permet à la DP de savoir si deux déploiements
partagent le même site ou non (§9, §13).

Version actuelle : QoS nominale seulement, sans effet météo (comme le
réseau existant hors scénario pluie). L'intégration météo pourra
s'ajouter plus tard en réutilisant la même chaîne que enrich_data.py.
"""

import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAPREP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "datapreparation")
if DATAPREP_DIR not in sys.path:
    sys.path.insert(0, DATAPREP_DIR)

from prepare_data import (
    normalize_generation,
    get_profile,
    check_los_binaire,
    calc_metrics,
    haversine_km,
    MAT_HEIGHT_M,
    _get_elevation,
)

# Mêmes seuils et formules que enrich_data.py, pour que la QoS d'une
# cellule hypothétique reste comparable à celle des cellules réelles.
DEBIT_SEUIL_MIN, DEBIT_SEUIL_REF = 1.0, 300.0
RTT_SEUIL_MS, K_SIGMOID = 250.0, 0.02
BER_SEUIL_BON = {"2G": 1e-2, "3G": 1e-3, "4G": 5e-4, "5G": 1e-4}
BER_SEUIL_MAUVAIS = {g: 10 * v for g, v in BER_SEUIL_BON.items()}


def qos_nominal(debit_mbps, rtt_ms, ber, generation):
    """QoS nominale (sans météo) : moyenne de trois satisfactions
    normalisées (débit, latence, BER), même formule que enrich_data.py."""
    if debit_mbps is None or debit_mbps <= DEBIT_SEUIL_MIN:
        q_debit = 0.0
    elif debit_mbps >= DEBIT_SEUIL_REF:
        q_debit = 1.0
    else:
        q_debit = ((math.log(debit_mbps) - math.log(DEBIT_SEUIL_MIN))
                   / (math.log(DEBIT_SEUIL_REF) - math.log(DEBIT_SEUIL_MIN)))

    if rtt_ms is None:
        q_latence = 0.0
    else:
        q_latence = 1.0 / (1.0 + math.exp(K_SIGMOID * (rtt_ms - RTT_SEUIL_MS)))

    gen_norm = normalize_generation(generation)
    bon = BER_SEUIL_BON.get(gen_norm, 1e-3)
    mauvais = BER_SEUIL_MAUVAIS.get(gen_norm, 1e-2)
    if ber is None or ber <= 0:
        q_ber = 0.0
    elif ber <= bon:
        q_ber = 1.0
    elif ber >= mauvais:
        q_ber = 0.0
    else:
        q_ber = 1.0 - ((math.log10(ber) - math.log10(bon))
                       / (math.log10(mauvais) - math.log10(bon)))

    return (q_debit + q_latence + q_ber) / 3


def simuler_deploiement(lat_site, lon_site, generation, bande_mhz, df_points, vitesse_kmh=80.0):
    """Simule une antenne hypothétique et calcule ce qu'elle produirait
    à chaque point du trajet.

    lat_site, lon_site : coordonnées du site. Pour un site déjà existant,
        utiliser ses coordonnées exactes (§13, upgrade). Pour un nouveau
        site, de nouvelles coordonnées.
    generation, bande_mhz : la technologie déployée.
    df_points : DataFrame avec au moins point_id, lat, lon, altitude_m,
        vegetation, in_tunnel (un point par ligne, comme dataset_base.csv).
    vitesse_kmh : vitesse du train, pour la dégradation Doppler (comme
        prepare_data.py).

    Retourne une liste de dicts {"point_id", "cellule", "qos"}, un par
    point où le lien est valide (dans la portée technologique ET en
    LOS). Les tunnels et les points hors portée/NLOS n'apparaissent pas,
    exactement comme pour les antennes réelles dans prepare_data.py.
    """
    gen_norm = normalize_generation(generation)
    portee_km = get_profile(gen_norm, bande_mhz)[0]
    mat_m = MAT_HEIGHT_M.get(gen_norm, 35.0)

    alt_sol_site = _get_elevation(lat_site, lon_site, df_points)
    alt_abs_site = alt_sol_site + mat_m
    site_id = (round(lat_site, 6), round(lon_site, 6))

    resultats = []
    for pt in df_points.itertuples():
        if pt.in_tunnel:
            continue  # pas de lien radio en tunnel, comme le réseau existant

        dist_km = haversine_km(pt.lat, pt.lon, lat_site, lon_site)
        if dist_km > portee_km:
            continue  # hors de la portée technologique de cette bande

        los_ok = check_los_binaire(
            pt.lat, pt.lon, pt.altitude_m,
            lat_site, lon_site, alt_abs_site,
            df_points)
        if not los_ok:
            continue  # NLOS : lien non retenu

        (perte_db, snr_base, snr_adjusted, debit_adj, rtt, pkt_loss, ber,
         delta_ber, delta_vitesse, delta_snr, modele) = calc_metrics(
            dist_km, gen_norm, bande_mhz, pt.vegetation, 1, vitesse_kmh,
            pt_lat=pt.lat, pt_lon=pt.lon,
            ant_lat=lat_site, ant_lon=lon_site, ant_mat_m=mat_m)

        if modele == "P1812_FAILED":
            continue

        qos = qos_nominal(debit_adj, rtt, ber, gen_norm)

        resultats.append({
            "point_id": pt.point_id,
            "cellule": (site_id, None, gen_norm, bande_mhz),
            "qos": qos,
            "debit_adj_mbps": debit_adj,
            "rtt_ms": rtt,
            "ber": ber,
        })

    return resultats