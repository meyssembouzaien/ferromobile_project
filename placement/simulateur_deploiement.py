"""
simulateur_deploiement.py — Simule l'ajout d'une infrastructure hypothétique
==============================================================================
Prend un déploiement décidé par le RL (x, y, generation, bande) et calcule
ce qu'il produirait réellement le long du trajet, en réutilisant EXACTEMENT
le même modèle physique que prepare_data.py (P.1812, LOS, débit empirique)
ET la même chaîne météo/ARQ que enrich_data.py (pluie P.838, BER après
pluie, packet loss, débit/RTT effectifs sous retransmission).

Chaîne complète (cahier des charges §2, §13) :

    (x,y,g,f) -> antenne hypothétique -> P.1812 -> SNR nominal
              -> pluie (P.838-3) -> SNR après pluie -> BER après pluie
              -> packet loss (avec gain de codage) -> ARQ
              -> débit/RTT effectifs -> QoS -> cellule hypothétique
              c_RL=(site,None,g,f)

Cette cellule peut ensuite être ajoutée à cells_per_point et passée à
trajectory_dp() exactement comme une cellule réelle.

CORRECTION IMPORTANTE (post-review) : la version précédente de ce fichier
calculait la QoS directement depuis les métriques NOMINALES de calc_metrics()
(SNR/débit/RTT avant tout effet météo), sans jamais appliquer la pluie ni
l'ARQ — alors que enrich_data.py applique les deux au réseau réel avant de
calculer la QoS finale. Résultat : une cellule hypothétique semblait
toujours "meilleure" qu'une cellule réelle équivalente sous le même
scénario météo, uniquement parce qu'elle échappait à la dégradation pluie.
Ce biais aurait faussé U(I)/r_t/r_fin systématiquement en faveur du RL.
Corrigé en reproduisant ici exactement la même chaîne pluie -> BER ->
packet loss -> ARQ que enrich_data.py, en import direct des mêmes
fonctions/constantes (atten_pluie_p838, ber_physique, SNR_MAX_DB,
CODING_GAIN, BITS_PER_PKT, PATH_FRAC) — source unique de vérité, jamais
une deuxième implémentation.

Le site est identifié par ses coordonnées (lat, lon) directement, jamais
par un ant_id : pour une amélioration de site existant, on passe les
coordonnées exactes du site déjà là ; pour un nouveau site, de nouvelles
coordonnées. C'est ce qui permet à la DP de savoir si deux déploiements
partagent le même site ou non (§9, §13).

Non traité ici, volontairement, et distinct de ce fix : le shadowing de
calc_metrics() (_RNG_DELTA dans prepare_data.py) consomme un flux
aléatoire global à chaque appel, donc n'est pas reproductible lien par
lien. C'est un problème réel pour la reproductibilité de l'entraînement
RL, mais le corriger toucherait prepare_data.py et donc la génération du
réseau réel déjà verrouillée (dataset_base.csv/dataset_enrichi.csv) — à
traiter séparément, en connaissance de cause, pas comme un correctif
silencieux glissé ici.
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
    TECH_PROFILES,
)
from enrich_data import (
    atten_pluie_p838,
    ber_physique as ber_physique_pluie,  # nom distinct : calc_metrics() a déjà son propre ber_physique interne (nominal, pré-pluie)
    SNR_MAX_DB,
    CODING_GAIN,
    BITS_PER_PKT,
    PATH_FRAC,
)

# Mêmes seuils et formules que enrich_data.py, pour que la QoS d'une
# cellule hypothétique reste comparable à celle des cellules réelles.
DEBIT_SEUIL_MIN, DEBIT_SEUIL_REF = 1.0, 300.0
RTT_SEUIL_MS, K_SIGMOID = 250.0, 0.02
BER_SEUIL_BON = {"2G": 1e-2, "3G": 1e-3, "4G": 5e-4, "5G": 1e-4}
BER_SEUIL_MAUVAIS = {g: 10 * v for g, v in BER_SEUIL_BON.items()}


def qos_nominal(debit_mbps, rtt_ms, ber, generation):
    """Moyenne de trois satisfactions normalisées (débit, latence, BER),
    même formule que enrich_data.py. Le nom "nominal" est historique :
    malgré le nom, cette fonction est maintenant appelée avec des valeurs
    EFFECTIVES (post-pluie, post-ARQ), pas nominales — voir simuler_deploiement()."""
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


def simuler_deploiement(lat_site, lon_site, generation, bande_mhz, df_points,
                         df_meteo_scenario, vitesse_kmh=80.0):
    """Simule une antenne hypothétique et calcule ce qu'elle produirait
    à chaque point du trajet, SOUS LE MÊME SCÉNARIO MÉTÉO que le réseau
    réel comparé (cohérence avec enrich_data.py).

    lat_site, lon_site : coordonnées du site. Pour un site déjà existant,
        utiliser ses coordonnées exactes (§13, upgrade). Pour un nouveau
        site, de nouvelles coordonnées.
    generation, bande_mhz : la technologie déployée. DOIT être une clé de
        TECH_PROFILES (prepare_data.py) — lève ValueError sinon, plutôt
        que de retomber silencieusement sur DEFAULT_PROFILE comme le fait
        get_profile() pour le réseau réel (acceptable pour des données
        ANFR réelles imparfaites, pas pour une action RL qui doit être
        bien définie). L'espace d'action de environment_rl.py est déjà
        construit uniquement à partir de TECH_PROFILES, donc cette
        vérification ne devrait jamais échouer en usage normal — elle
        protège contre un appel direct erroné (tests, scripts).
    df_points : DataFrame avec au moins point_id, lat, lon, altitude_m,
        vegetation, in_tunnel (un point par ligne, comme dataset_base.csv).
    df_meteo_scenario : DataFrame avec au moins point_id, pluie_mm_h, DÉJÀ
        filtré sur le même scénario météo que le réseau réel comparé
        (même convention que build_state.construire_canaux_statiques).
    vitesse_kmh : vitesse du train, pour la dégradation Doppler (comme
        prepare_data.py).

    Retourne une liste de dicts {"point_id", "cellule", "qos",
    "debit_adj_mbps", "rtt_ms", "ber"}, un par point où le lien est valide
    (dans la portée technologique ET en LOS). debit_adj_mbps/rtt_ms/ber
    sont les valeurs EFFECTIVES (après pluie et ARQ), directement
    comparables aux colonnes du même nom dans dataset_enrichi.csv. Les
    tunnels et les points hors portée/NLOS n'apparaissent pas, exactement
    comme pour les antennes réelles dans prepare_data.py.
    """
    gen_norm = normalize_generation(generation)
    try:
        bande_int = int(float(bande_mhz))
    except (TypeError, ValueError):
        bande_int = None
    if (gen_norm, bande_int) not in TECH_PROFILES:
        raise ValueError(
            f"simuler_deploiement: combinaison (génération, bande) invalide : "
            f"({gen_norm}, {bande_mhz}). Doit être une clé de TECH_PROFILES "
            f"(prepare_data.py) — l'environnement RL ne devrait jamais proposer "
            f"autre chose (voir BANDES_PAR_GENERATION dans environment_rl.py)."
        )

    portee_km = get_profile(gen_norm, bande_mhz)[0]
    mat_m = MAT_HEIGHT_M.get(gen_norm, 35.0)

    alt_sol_site = _get_elevation(lat_site, lon_site, df_points)
    alt_abs_site = alt_sol_site + mat_m
    site_id = (round(lat_site, 6), round(lon_site, 6))

    pluie_par_point = df_meteo_scenario.set_index("point_id")["pluie_mm_h"]
    freq_mhz = float(bande_mhz)

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

        (perte_db, snr_base, snr_adjusted, debit_nominal, rtt_nominal, pkt_loss_nominal,
         ber_nominal, delta_ber, delta_vitesse, delta_snr, modele) = calc_metrics(
            dist_km, gen_norm, bande_mhz, pt.vegetation, 1, vitesse_kmh,
            pt_lat=pt.lat, pt_lon=pt.lon,
            ant_lat=lat_site, ant_lon=lon_site, ant_mat_m=mat_m)

        if modele == "P1812_FAILED":
            continue

        # --- Pluie (ITU-R P.838-3), même chaîne que enrich_data.py ---
        pluie_mm_h = float(pluie_par_point.get(pt.point_id, 0.0))
        atten_db_km = atten_pluie_p838(pluie_mm_h, freq_mhz)
        delta_snr_pluie = atten_db_km * dist_km * PATH_FRAC
        cap = SNR_MAX_DB.get(gen_norm, 35.0)
        snr_pluie = min(max(snr_adjusted - delta_snr_pluie, -10.0), cap)
        ber_pluie = ber_physique_pluie(snr_pluie, gen_norm)

        # --- Packet loss (avec gain de codage), puis ARQ — même chaîne ---
        snr_code = snr_pluie + CODING_GAIN.get(gen_norm, 0.0)
        ber_eff = ber_physique_pluie(snr_code, gen_norm)
        pc = math.exp(BITS_PER_PKT * math.log(max(1e-15, 1.0 - ber_eff)))
        packet_loss_pct = min(100.0, max(0.0, (1.0 - pc) * 100.0))

        p = min(max(packet_loss_pct / 100.0, 0.0), 0.95)
        debit_adj = round(debit_nominal * (1.0 - p), 4)
        rtt = round(rtt_nominal / (1.0 - p), 1)

        qos = qos_nominal(debit_adj, rtt, ber_pluie, gen_norm)

        resultats.append({
            "point_id": pt.point_id,
            "cellule": (site_id, None, gen_norm, bande_mhz),
            "qos": qos,
            "debit_adj_mbps": debit_adj,
            "rtt_ms": rtt,
            "ber": ber_pluie,
        })

    return resultats