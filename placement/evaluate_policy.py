"""
evaluate_policy.py : évaluation d'une politique entraînée (tableaux du chapitre 4).

    final       tableau 4.8 : S1, B = 10 M€
    budget      tableau 4.14 : B de 1 à 12 M€
    operateurs  tableau 4.12 : un opérateur retiré à la fois
    meteo       tableau 4.12 : scénarios S2, S3, S4
    pli         tableau 4.12 : bloc de test du pli --pli k, avec Coverage-Greedy
    petite      tableau 4.11 : petite instance (9 candidats)

Lancer (depuis placement/) :
    python evaluate_policy.py <checkpoint.pt> final budget operateurs meteo
    python evaluate_policy.py <checkpoint_pli2.pt> pli --pli 2
Sortie : resultats/evaluation_<run>.csv
"""

import argparse
import collections
import os
import time

import numpy as np
import pandas as pd
import torch

import train_dqn as T
from dqn_model import FerroMobileDuelingDQN, FerroMobileVanillaDQN

B_TRAIN = 10_000_000.0
BUDGETS_G = [1e6, 2e6, 3e6, 4e6, 6e6, 8e6, 10e6, 12e6]
OPERATEURS = ["ORANGE", "SFR", "BOUYGUES TELECOM", "FREE MOBILE"]
DOSSIER_RESULTATS = os.path.join(T.RACINE_PROJET, "resultats")


def charger_politique(chemin, device):
    etat = torch.load(chemin, map_location=device, weights_only=False)
    classe = {"FerroMobileDuelingDQN": FerroMobileDuelingDQN,
              "FerroMobileVanillaDQN": FerroMobileVanillaDQN}[etat["architecture"]]
    return etat, classe


def evaluer(env, etat_ckpt, classe, device, experience, configuration, extra=None):
    from environment_rl import BANDES_PAR_GENERATION
    n_configs = sum(len(v) for v in BANDES_PAR_GENERATION.values())
    modele = classe(n_configs, env.grille.corridor_mask).to(device)
    modele.load_state_dict(etat_ckpt["modele"])
    modele.eval()
    env.reset()
    n_avant = env._dernier_n_white
    debut = time.time()
    resume, _ = T.episode_glouton(env, modele, device)
    ligne = {"experience": experience, "configuration": configuration, "episode": etat_ckpt["episode"],
             "n_white_avant": n_avant, "rho": (n_avant - resume["n_white"]) / n_avant if n_avant else float("nan"),
             **resume, "duree_s": round(time.time() - debut, 1), **(extra or {})}
    print(f"{experience:10s} {configuration:22s} n_white {n_avant} -> {resume['n_white']}, "
          f"coût {resume['cout_eur'] / 1e6:.3f} M€, Q {resume['qos_mean']:.3f}, Q_min {resume['qos_min']:.3f}, "
          f"retour réf. {resume['return_ref']:.4f}", flush=True)
    return ligne


def petite_instance_actions(env):
    """Les 9 candidats de small_instance.py."""
    from baseline_trajectoire_gloutonne import points_blancs
    env.reset()
    pts = env.df_points[env.df_points.point_id.isin(points_blancs(env))]
    compte = collections.Counter(c for c in (env.grille.gps_vers_cellule(r.lat, r.lon) for r in pts.itertuples()) if c)
    cellules = [c for c, _ in compte.most_common(3)]
    candidats = [(c[0], c[1], g, f) for c in cellules for g, f in [("3G", 900), ("4G", 800), ("5G", 700)]]
    autorisees = np.zeros(len(env.actions), dtype=bool)
    autorisees[[env.action_index[a] for a in candidats]] = True
    return autorisees


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("experiences", nargs="+", choices=["final", "budget", "operateurs", "meteo", "pli", "petite"])
    p.add_argument("--pli", type=int, default=0)
    args = p.parse_args()

    T.preparer_environnement_colab()
    from environment_rl import FerroMobileEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    etat_ckpt, classe = charger_politique(args.checkpoint, device)
    nom_run = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    print(f"Politique : {args.checkpoint} ({etat_ckpt['architecture']}, épisode {etat_ckpt['episode']}, "
          f"poids d'entraînement {etat_ckpt.get('poids')})")

    def env_(**k):
        k.setdefault("scenario", "S1")
        k.setdefault("budget", B_TRAIN)
        k.setdefault("budget_reference", B_TRAIN)
        return FerroMobileEnv(t_max=T.T_MAX, sauvegarder_deploiements_rl=False, **k)

    lignes = []
    for exp in args.experiences:
        if exp == "final":
            lignes.append(evaluer(env_(), etat_ckpt, classe, device, "final", "S1, B = 10 M€"))
        elif exp == "budget":
            for b in BUDGETS_G:
                lignes.append(evaluer(env_(budget=b), etat_ckpt, classe, device, "budget", f"B = {b / 1e6:g} M€"))
        elif exp == "operateurs":
            for o in OPERATEURS:
                lignes.append(evaluer(env_(operateurs_exclus=[o]), etat_ckpt, classe, device, "operateurs", f"sans {o}"))
        elif exp == "meteo":
            for s in ["S2", "S3", "S4"]:
                lignes.append(evaluer(env_(scenario=s), etat_ckpt, classe, device, "meteo", s))
        elif exp == "pli":
            import cv_blocks
            from baseline_couverture_gloutonne import couverture_gloutonne
            assert args.pli, "--pli k requis"
            assert etat_ckpt.get("config", {}).get("pli") == args.pli, "checkpoint entraîné sur un autre pli"
            base = env_()
            base.reset()
            config = cv_blocks.preparer_pli(base, args.pli, B_TRAIN)
            del base
            lignes.append(evaluer(cv_blocks.env_test(config), etat_ckpt, classe, device, "pli",
                                  f"pli {args.pli}, bloc de test", {"budget_eur": config["budget_test"]}))
            env_cg = cv_blocks.env_test(config)
            env_cg.reset()
            n_avant = env_cg._dernier_n_white
            resume, _ = couverture_gloutonne(env_cg, verbeux=False)
            lignes.append({"experience": "pli", "configuration": f"pli {args.pli}, Coverage-Greedy",
                           "n_white_avant": n_avant, "rho": (n_avant - resume["n_white"]) / n_avant,
                           "cout_eur": resume["cout"], "budget_eur": config["budget_test"],
                           **{k: resume[k] for k in ("n_white", "qos_mean", "qos_min", "N_H", "N_O", "N_T", "r_fin")}})
            print(f"pli {args.pli} Coverage-Greedy : n_white {n_avant} -> {resume['n_white']}, "
                  f"coût {resume['cout'] / 1e6:.3f} M€")
        elif exp == "petite":
            ref = os.path.join(DOSSIER_RESULTATS, "petite_instance.csv")
            optimum = pd.read_csv(ref) if os.path.exists(ref) else None
            for b in (0.3e6, 0.5e6):
                base = env_(budget=b)
                env = env_(budget=b, actions_autorisees=petite_instance_actions(base))
                ligne = evaluer(env, etat_ckpt, classe, device, "petite", f"9 candidats, B = {b / 1e6:g} M€")
                if optimum is not None:
                    o = optimum[(optimum.method == "optimum") & (optimum.budget == b)]
                    if len(o):
                        U_opt = float(o.U.iloc[0])
                        ligne["U_optimum"] = U_opt
                        ligne["gap"] = (U_opt - ligne["U_final_ref"]) / abs(U_opt)
                        print(f"           écart à l'optimum : {100 * ligne['gap']:.2f} %")
                lignes.append(ligne)

    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    sortie = os.path.join(DOSSIER_RESULTATS, f"evaluation_{nom_run}.csv")
    df = pd.DataFrame(lignes)
    df.insert(0, "checkpoint", os.path.basename(args.checkpoint))
    if os.path.exists(sortie):
        df = pd.concat([pd.read_csv(sortie), df], ignore_index=True)
    df.to_csv(sortie, index=False)
    print(f"\nRésultats ajoutés à {sortie}")


if __name__ == "__main__":
    main()
