"""
baseline_couverture_gloutonne.py : Coverage-Greedy (tableau 3.14) joué dans l'environnement.

À chaque étape : le déploiement qui couvre le plus de nouveaux points blancs par euro.
Arrêt : couverture complète, aucun candidat utile, ou budget épuisé.

Lancer : python baseline_couverture_gloutonne.py   (S1, B = 10 M€)
Sortie : resultats/couverture_gloutonne_resume.csv
"""

import os
import time

import pandas as pd

from baseline_aleatoire import preparer_environnement, resume_episode, DOSSIER_RESULTATS
from baseline_trajectoire_gloutonne import points_blancs, cellules_candidates

D_COV = 1.0


def meilleure_action(env, configs):
    from cost_model import get_total_cost
    blancs = points_blancs(env)
    for rayon in (0, 1):
        meilleur, meilleur_ratio = None, 0.0
        for ix, iy in cellules_candidates(env, blancs, rayon):
            lat, lon = env.grille.cellule_vers_gps(ix, iy)
            for g, f in configs:
                action = (ix, iy, g, f)
                if not env._est_faisable(action):
                    continue
                nouveaux = {r["point_id"] for r in env._simuler(action, lat, lon)
                            if r["point_id"] in blancs and r["debit_adj_mbps"] >= D_COV}
                if not nouveaux:
                    continue
                ratio = len(nouveaux) / get_total_cost((ix, iy), g, f, env.sites_deja_presents)
                if ratio > meilleur_ratio:
                    meilleur, meilleur_ratio = action, ratio
        if meilleur is not None:
            return meilleur
    return None


def couverture_gloutonne(env, verbeux=True):
    """Joue Coverage-Greedy dans env (déjà construit). Renvoie (résumé, liste des actions)."""
    from environment_rl import BANDES_PAR_GENERATION
    configs = sorted((g, f) for g in BANDES_PAR_GENERATION for f in BANDES_PAR_GENERATION[g])
    debut = time.time()
    env.reset()
    done, reward_total, info, actions = False, 0.0, {"n_white": env._dernier_n_white}, []
    while not done:
        action = meilleure_action(env, configs) if env._dernier_n_white > 0 else None
        if action is None:
            _, reward, done, info = env.step("STOP")
            reward_total += reward
            break
        _, reward, done, info = env.step(action)
        reward_total += reward
        actions.append(action)
        if verbeux:
            print(f"  site {len(actions)} : {action}, n_white = {info['n_white']}, "
                  f"coût = {env.budget_total - env.budget_restant:.0f} €", flush=True)
    return resume_episode(env, info, reward_total, len(actions), time.time() - debut), actions


if __name__ == "__main__":
    preparer_environnement()
    from environment_rl import FerroMobileEnv
    env = FerroMobileEnv(scenario="S1", budget=10_000_000.0, t_max=150, sauvegarder_deploiements_rl=False)
    resume, actions = couverture_gloutonne(env)
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    pd.DataFrame([resume]).to_csv(os.path.join(DOSSIER_RESULTATS, "couverture_gloutonne_resume.csv"), index=False)
    print(resume)
