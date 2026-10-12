"""
test_chemins_rapides.py : les accélérations de environment_rl.py ne changent aucun résultat.

Compare le masque, l'état et un épisode complet avec la version d'origine.
La propagation radio est remplacée par une simulation factice (pas besoin du DEM).

Lancer (depuis placement/) : python test_chemins_rapides.py
"""

import os
import zlib

import numpy as np
import pandas as pd

if "CRC_COVLIB_PATH" not in os.environ:
    os.environ["CRC_COVLIB_PATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "crc-covlib", "python-wrapper")

import cost_model
import environment_rl
from build_state_dynamic import construire_canaux_infrastructure, construire_canaux_performance
from prepare_data import haversine_km, get_profile


def simulation_factice(lat_site, lon_site, generation, bande_mhz, df_points, df_meteo_scenario, vitesse_kmh=80.0):
    """Déterministe : points hors tunnel à portée, valeurs tirées d'un hachage du lien."""
    portee = min(get_profile(generation, bande_mhz)[0], 4.0)
    site = (round(lat_site, 6), round(lon_site, 6))
    res = []
    for pt in df_points.itertuples():
        if pt.in_tunnel or haversine_km(pt.lat, pt.lon, lat_site, lon_site) > portee:
            continue
        h = zlib.crc32(f"{site}{pt.point_id}{generation}{bande_mhz}".encode()) / 2**32
        res.append({"point_id": pt.point_id, "cellule": (site, None, generation, bande_mhz),
                    "qos": 0.4 + 0.5 * h, "debit_adj_mbps": 0.5 + 10 * h, "rtt_ms": 50.0 + h, "ber": 1e-4 * h})
    return res


def etat_reference(env):
    """État construit exactement comme dans la version d'origine de _construire_etat."""
    canaux_infra = construire_canaux_infrastructure(env.grille, env.df_reel, cellules_hypothetiques=env._resultats_hypo)
    if env._resultats_hypo:
        debit_hypo = env._serie_max_par_point(env._resultats_hypo, "debit_adj_mbps")
        qos_hypo = env._serie_max_par_point(env._resultats_hypo, "qos")
        debit_combine = pd.concat([env.debit_reel_par_point, debit_hypo], axis=1).max(axis=1)
    else:
        debit_combine = env.debit_reel_par_point
    meilleur = pd.DataFrame({"point_id": debit_combine.index, "meilleur_debit": debit_combine.values})
    qos_reelle = env.df_reel.groupby("point_id")["qos"].max()
    meilleure_qos = pd.concat([qos_reelle, qos_hypo], axis=1).max(axis=1) if env._resultats_hypo else qos_reelle
    meilleur = meilleur.merge(meilleure_qos.rename("meilleur_qos"), left_on="point_id", right_index=True, how="left")
    canaux_perf = construire_canaux_performance(env.grille, env.df_points, meilleur)
    from build_state_dynamic import construire_canal_mats_rl
    mats = construire_canal_mats_rl(env.grille, env.infrastructure_deployee)
    return np.concatenate([env.canaux_statiques, canaux_infra, canaux_perf, mats], axis=0)


def jouer(env, graine, n_steps, verifier):
    rng = np.random.default_rng(graine)
    etat = env.reset()
    trace = []
    for _ in range(n_steps):
        masque = env._actions_faisables()
        if verifier:
            assert np.array_equal(masque, env._actions_faisables_reference()), "masque vectorisé différent"
            assert np.array_equal(etat["grille"], etat_reference(env)), "état différent de l'origine"
        # Actions proches des points blancs
        idx = int(rng.choice(np.nonzero(masque)[0][1:]))
        etat, r, done, info = env.step(env.actions[idx])
        trace.append((idx, r, env.U_courant, info.get("n_white"), etat["b_t"], etat["grille"].copy()))
        if done:
            break
    return trace


def test_equivalence():
    environment_rl.simuler_deploiement = simulation_factice
    env = environment_rl.FerroMobileEnv(scenario="S1", budget=1_000_000.0, t_max=12,
                                        sauvegarder_deploiements_rl=False)
    # Quelques centaines d'actions proches de la voie
    jouer(env, 0, 12, verifier=True)
    print("test_equivalence : OK (masque et état identiques à la version d'origine)")

    ancien = cost_model.C_SITE_EUR
    cost_model.C_SITE_EUR = 250_000           # variante de tg_variant.py : le masque suit le module
    try:
        jouer(env, 1, 6, verifier=True)
    finally:
        cost_model.C_SITE_EUR = ancien
    print("test_equivalence (C_site modifié) : OK")


def test_cache():
    environment_rl.simuler_deploiement = simulation_factice
    env = environment_rl.FerroMobileEnv(scenario="S1", budget=1_000_000.0, t_max=10,
                                        sauvegarder_deploiements_rl=False)
    environment_rl.vider_cache_simulation()
    sans_cache = jouer(env, 3, 10, verifier=False)          # remplit le cache
    avec_cache = jouer(env, 3, 10, verifier=False)          # lit le cache
    assert len(sans_cache) == len(avec_cache)
    for a, b in zip(sans_cache, avec_cache):
        assert a[:5] == b[:5] and np.array_equal(a[5], b[5]), "le cache change un résultat"
    print("test_cache : OK (même épisode avec et sans cache)")


def test_budget_reference():
    environment_rl.simuler_deploiement = simulation_factice
    env = environment_rl.FerroMobileEnv(scenario="S1", budget=12_000_000.0, budget_reference=10_000_000.0,
                                        t_max=5, sauvegarder_deploiements_rl=False)
    assert abs(env.reset()["b_t"] - 1.2) < 1e-12, "b_t doit valoir B / B_train (éq. 3.29)"
    print("test_budget_reference : OK (b_t = 1.2 pour B = 12 M€)")


def test_options_experiences():
    environment_rl.simuler_deploiement = simulation_factice
    base = environment_rl.FerroMobileEnv(scenario="S1", budget=1e6, t_max=5, sauvegarder_deploiements_rl=False)
    base.reset()
    sans_orange = environment_rl.FerroMobileEnv(scenario="S1", budget=1e6, t_max=5, sauvegarder_deploiements_rl=False,
                                                operateurs_exclus=["ORANGE"])
    sans_orange.reset()
    assert sans_orange._dernier_n_white >= base._dernier_n_white
    assert "ORANGE" not in set(sans_orange.df_reel["operateur"])
    print(f"test_options_experiences : sans Orange, {sans_orange._dernier_n_white} zones blanches "
          f"(réseau complet : {base._dernier_n_white})")

    points = sorted(base.points_hors_tunnel)[:500]
    bloc = environment_rl.FerroMobileEnv(scenario="S1", budget=1e6, t_max=5, sauvegarder_deploiements_rl=False,
                                         points_evalues=points, points_stats=points)
    bloc.reset()
    assert bloc.n_total == len(points) and bloc._dernier_n_white <= base._dernier_n_white
    autorisees = np.zeros(len(base.actions), dtype=bool)
    autorisees[[5, 17, 42]] = True
    petit = environment_rl.FerroMobileEnv(scenario="S1", budget=1e6, t_max=5, sauvegarder_deploiements_rl=False,
                                          actions_autorisees=autorisees)
    petit.reset()
    m = petit._actions_faisables()
    assert m[0] and set(np.nonzero(m)[0][1:]) <= {5, 17, 42}
    assert np.array_equal(m, petit._actions_faisables_reference())
    print("test_options_experiences : OK (opérateur retiré, bloc de points, actions restreintes)")


if __name__ == "__main__":
    test_budget_reference()
    test_options_experiences()
    test_cache()
    test_equivalence()
    print("\nTous les tests passent.")
