"""
baseline_aleatoire.py : politique aléatoire (baseline sans apprentissage).

À chaque step, une action faisable tirée uniformément, jusqu'à la fin de l'épisode.
Répétée sur plusieurs graines. Même environnement, même budget que le RL.

Lancer : python baseline_aleatoire.py
Sortie : resultats/baseline_aleatoire.csv
"""

import os
import time

import numpy as np
import pandas as pd

CE_FICHIER = os.path.dirname(os.path.abspath(__file__))
RACINE_PROJET = os.path.dirname(CE_FICHIER)
DOSSIER_RESULTATS = os.path.join(RACINE_PROJET, "resultats")

SCENARIO = "S1"
BUDGET = 10_000_000.0
T_MAX = 150
GRAINES = [0, 1, 2, 3, 4]


def preparer_environnement():
    """Chemin de crc-covlib, avant l'import qui le charge."""
    if "CRC_COVLIB_PATH" not in os.environ:
        os.environ["CRC_COVLIB_PATH"] = os.path.join(RACINE_PROJET, "crc-covlib", "python-wrapper")


def resume_episode(env, info, reward_total, n_steps, duree):
    """Mesures communes à toutes les baselines."""
    dp = env._dernier_resultat_dp
    return {"n_white": info["n_white"], "cout": info["cout_total"],
            "qos_mean": dp["qos_mean"], "qos_min": dp["qos_min"],
            "N_H": dp["N_H"], "N_O": dp["N_O"], "N_T": dp["N_T"],
            "r_fin": info["r_fin"], "reward_total": reward_total,
            "n_steps": n_steps, "duree_s": round(duree)}


def un_episode(env, rng):
    debut = time.time()
    env.reset()
    done, reward_total, n_steps, info = False, 0.0, 0, {}
    while not done:
        faisables = np.nonzero(env._actions_faisables())[0]
        action = env.actions[int(rng.choice(faisables))]
        _, reward, done, info = env.step(action)
        reward_total += reward
        n_steps += 1
    return resume_episode(env, info, reward_total, n_steps, time.time() - debut)


if __name__ == "__main__":
    preparer_environnement()
    from environment_rl import FerroMobileEnv

    env = FerroMobileEnv(scenario=SCENARIO, budget=BUDGET, t_max=T_MAX, sauvegarder_deploiements_rl=False)
    lignes = []
    for graine in GRAINES:
        resultat = {"graine": graine, **un_episode(env, np.random.default_rng(graine))}
        lignes.append(resultat)
        print(f"graine {graine} : n_white = {resultat['n_white']}, coût = {resultat['cout']:.0f} €, "
              f"qos_min = {resultat['qos_min']:.3f}, r_fin = {resultat['r_fin']:.4f}, "
              f"{resultat['n_steps']} steps, {resultat['duree_s']} s", flush=True)

    df = pd.DataFrame(lignes)
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    df.to_csv(os.path.join(DOSSIER_RESULTATS, "baseline_aleatoire.csv"), index=False)
    colonnes = ["n_white", "cout", "qos_mean", "qos_min", "r_fin", "reward_total"]
    print("\nMoyenne (écart-type) sur", len(GRAINES), "graines :")
    for c in colonnes:
        print(f"  {c:13s} : {df[c].mean():.4f} ({df[c].std():.4f})")
