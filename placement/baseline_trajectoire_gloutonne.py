"""
baseline_trajectoire_gloutonne.py : baseline "trajectoire gloutonne" (§20).

À chaque étape, garde le déploiement qui augmente le plus U(I) par euro :
    a* = argmax (U(I + a) - U(I)) / Coût(a)
Candidats : cellules des points encore blancs, puis voisinage 3 x 3 (comme la couverture gloutonne).
Arrêt : plus aucun candidat n'améliore U (STOP), ou fin d'épisode de l'environnement.

Tout est calculé dans FerroMobileEnv : U, coût et r_fin sont ceux du RL.
Chaque site choisi est enregistré : relancer le script reprend là où il s'était arrêté.

Lancer : python baseline_trajectoire_gloutonne.py
Sorties : resultats/trajectoire_gloutonne_sites.csv, resultats/trajectoire_gloutonne_resume.csv
"""

import os
import time

import pandas as pd

from baseline_aleatoire import preparer_environnement, resume_episode, DOSSIER_RESULTATS

SCENARIO = "S1"
BUDGET = 10_000_000.0
T_MAX = 150
D_COV = 1.0
FICHIER_SITES = os.path.join(DOSSIER_RESULTATS, "trajectoire_gloutonne_sites.csv")
FICHIER_RESUME = os.path.join(DOSSIER_RESULTATS, "trajectoire_gloutonne_resume.csv")


def points_blancs(env):
    """point_id des points hors tunnel encore sous D_COV."""
    debit = env.debit_reel_par_point
    if env._resultats_hypo:
        debit = pd.concat([debit, env._serie_max_par_point(env._resultats_hypo, "debit_adj_mbps")],
                          axis=1).max(axis=1)
    debit = debit.reindex(list(env.points_hors_tunnel)).fillna(0)
    return set(debit[debit < D_COV].index)


def cellules_candidates(env, blancs, rayon):
    """Cellules du corridor contenant un point blanc (rayon 0) ou voisines (rayon 1)."""
    pts = env.df_points[env.df_points["point_id"].isin(blancs)]
    cellules = set()
    for r in pts.itertuples():
        c = env.grille.gps_vers_cellule(r.lat, r.lon)
        if c is None:
            continue
        for dx in range(-rayon, rayon + 1):
            for dy in range(-rayon, rayon + 1):
                ix, iy = c[0] + dx, c[1] + dy
                if 0 <= ix < env.grille.nx and 0 <= iy < env.grille.ny and env.grille.corridor_mask[ix, iy]:
                    cellules.add((ix, iy))
    return cellules


def simuler(env, action, cache):
    """Résultats radio d'une antenne seule. Ne dépend pas des autres antennes : mis en cache."""
    if action not in cache:
        from simulateur_deploiement import simuler_deploiement
        ix, iy, g, f = action
        lat, lon = env.grille.cellule_vers_gps(ix, iy)
        res = simuler_deploiement(lat, lon, g, f, env.df_points, env.df_meteo_scenario)
        cache[action] = [r for r in res if r["point_id"] in env.points_hors_tunnel]
    return cache[action]


def gain_utilite(env, action, cache):
    """U(I + a) - U(I), sans modifier l'environnement."""
    n_avant = len(env._resultats_hypo)
    env._resultats_hypo.extend(simuler(env, action, cache))
    resultat_dp, n_white, n_eval = env._evaluer_dp_courant()
    u_apres = env._calculer_U(resultat_dp, n_eval, n_white)
    del env._resultats_hypo[n_avant:]   # l'environnement redevient comme avant
    return u_apres - env.U_courant


def meilleure_action(env, configs, cache):
    """Meilleur ratio gain / coût, rayon 0 puis rayon 1. None si aucun gain positif."""
    from cost_model import get_total_cost
    blancs = points_blancs(env)
    for rayon in (0, 1):
        meilleur, meilleur_ratio = None, 0.0
        cellules = cellules_candidates(env, blancs, rayon)
        print(f"  rayon {rayon} : {len(cellules)} cellules x {len(configs)} configs", flush=True)
        for ix, iy in cellules:
            for g, f in configs:
                action = (ix, iy, g, f)
                if not env._est_faisable(action):
                    continue
                ratio = gain_utilite(env, action, cache) / get_total_cost((ix, iy), g, f, env.sites_deja_presents)
                if ratio > meilleur_ratio:
                    meilleur, meilleur_ratio = action, ratio
        if meilleur is not None:
            return meilleur
    return None


def lancer():
    """Boucle complète, avec reprise depuis FICHIER_SITES."""
    from environment_rl import FerroMobileEnv, BANDES_PAR_GENERATION
    from cost_model import get_total_cost

    env = FerroMobileEnv(scenario=SCENARIO, budget=BUDGET, t_max=T_MAX, sauvegarder_deploiements_rl=False)
    configs = [(g, f) for g in BANDES_PAR_GENERATION for f in BANDES_PAR_GENERATION[g]]
    cache = {}
    debut = time.time()
    env.reset()
    print(f"Départ : n_white = {env._dernier_n_white}, U = {env.U_courant:.4f}", flush=True)

    # Reprise : rejoue les sites déjà choisis
    sites = pd.read_csv(FICHIER_SITES).to_dict("records") if os.path.exists(FICHIER_SITES) else []
    reward_total, done, info = 0.0, False, {}
    for s in sites:
        _, reward, done, info = env.step((int(s["ix"]), int(s["iy"]), s["generation"], int(s["bande_mhz"])))
        reward_total += reward
    if sites:
        print(f"Reprise après {len(sites)} sites : n_white = {env._dernier_n_white}", flush=True)

    while not done:
        action = meilleure_action(env, configs, cache)
        if action is None:
            print("Aucun candidat faisable n'améliore U (ou budget insuffisant) : STOP", flush=True)
            _, reward, done, info = env.step("STOP")
            reward_total += reward
            break

        cout = get_total_cost((action[0], action[1]), action[2], action[3], env.sites_deja_presents)
        _, reward, done, info = env.step(action)
        reward_total += reward
        sites.append({"site": len(sites) + 1, "ix": action[0], "iy": action[1], "generation": action[2],
                      "bande_mhz": action[3], "cout": cout,
                      "n_white": info["n_white"], "U": round(env.U_courant, 5)})
        os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
        pd.DataFrame(sites).to_csv(FICHIER_SITES, index=False)
        print(f"site {len(sites)} : {action}, n_white = {info['n_white']}, U = {env.U_courant:.4f}, "
              f"{time.time() - debut:.0f} s", flush=True)

    resume = resume_episode(env, info, reward_total, len(sites), time.time() - debut)
    pd.DataFrame([resume]).to_csv(FICHIER_RESUME, index=False)
    print(f"\n>>> {len(sites)} sites, n_white = {resume['n_white']}, coût = {resume['cout']:.0f} €, "
          f"qos_min = {resume['qos_min']:.3f}, r_fin = {resume['r_fin']:.4f}", flush=True)
    return resume


if __name__ == "__main__":
    preparer_environnement()
    lancer()
