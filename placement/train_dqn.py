"""
train_dqn.py : entraînement du Dueling DQN (ou du Vanilla DQN) sur FerroMobileEnv.

Configuration du mémoire (section 3.8.3, tableau 4.1) : gamma = 1, Adam 1e-4, Huber,
batch 32, replay buffer de 2000 transitions, epsilon de 1 à 0,05 en 6000 steps,
exploration de Boltzmann (T = 1) après 1000 steps uniformes, cible Double DQN,
réseau cible mis à jour tous les 200 steps, t_max = 150, B = 10 M€, évaluation
gloutonne et checkpoint tous les 25 épisodes.

Réglages sans modifier le code (variables d'environnement) :
    FM_EPISODES=1000      nombre total d'épisodes (défaut 300)
    FM_W_W=5              poids w_W des zones blanches dans U (défaut 1, valeur de référence)
    FM_ARCH=vanilla       DQN sans décomposition V/A (ablation), défaut : dueling
    FM_EXPLORATION=uniforme   ablation de l'exploration, défaut : softmax (Boltzmann)
    FM_SEED=1             graine (défaut 0)
    FM_FOLD=1..4          validation croisée spatiale, pli k (Expérience C, niveau 1) ; défaut : ligne entière
    FM_SAVE_EVERY=25      checkpoint tous les N épisodes
    FM_MAX_HOURS=11       arrêt propre (checkpoint + sauvegarde) avant cette durée de session ;
                          défaut : 11 sur Kaggle, aucune limite ailleurs
    FM_SYNC=1             sauvegarde dans un dataset Kaggle privé (défaut : 1 sur Kaggle)
    FM_KAGGLE_DATASET     nom du dataset (défaut : fm-<nom du run>)
    FM_RUN=nom            nom du run et de son dossier checkpoints/<nom>
    FM_CHECKPOINTS=dossier  dossier racine des runs (défaut : <projet>/checkpoints)
Variantes de récompense NON utilisées dans le mémoire (perspectives, section 4.13) :
    FM_W_TRANS=2          multiplie w_H, w_O, w_T dans U pendant l'entraînement (défaut 1)
    FM_COST_STEP=0.2      r_t - FM_COST_STEP * coût du step / B (défaut 0)

Journaux (dans le dossier du run) :
    train_log.csv  : une ligne par épisode
    eval_log.csv   : une ligne par évaluation gloutonne ; return_ref = U_T - U_0 + r_fin
                     recalculé avec les poids de référence (tous à 1), comparable aux
                     autres méthodes du tableau 4.8 quel que soit w_W
    eval_trace_epN.csv : chaque action de l'évaluation N
    console.log    : toute la sortie console, toutes sessions confondues
    run_etat.json  : dernier épisode sauvegardé
Reprise exacte : poids, réseau cible, optimiseur, epsilon, compteur de steps, états des
générateurs aléatoires et replay buffer. À la reprise, les lignes des épisodes postérieurs
au checkpoint sont supprimées (ils sont rejoués), donc aucun épisode n'apparaît deux fois.
"""

import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from dqn_model import FerroMobileDuelingDQN, FerroMobileVanillaDQN, N_CANAUX_ETAT
from replay_buffer import ReplayBuffer

CE_FICHIER = os.path.dirname(os.path.abspath(__file__))
RACINE_PROJET = os.path.dirname(CE_FICHIER)
SUR_KAGGLE = os.path.isdir("/kaggle/working")

# Hyperparamètres (tableau 4.1)
N_EPISODES_TRAIN = int(os.environ.get("FM_EPISODES", 300))
T_MAX = 150
BUDGET = 10_000_000.0
SCENARIO = "S1"
BATCH_SIZE = 32
GAMMA = 1.0                      # épisodique : le retour vaut U_T - U_0 + r_fin
LR = 1e-4
EPSILON_DEBUT, EPSILON_FIN = 1.0, 0.05
N_STEPS_EXPLORATION = 6000       # epsilon atteint EPSILON_FIN après ce nombre de steps (éq. 3.37)
EPSILON_DECROISSANCE = EPSILON_FIN ** (1 / N_STEPS_EXPLORATION)
CAPACITE_BUFFER = 2000
N_CANAUX_FIXES = 25              # 9 statiques + 16 infrastructure : stockés une fois dans le buffer
MODE_EXPLORATION = os.environ.get("FM_EXPLORATION", "softmax")   # "uniforme" ou "softmax"
TEMPERATURE = 1.0                # tableau 3.13
N_STEPS_UNIFORME = 1000          # softmax : exploration uniforme tant que le réseau n'a rien appris
SEED = int(os.environ.get("FM_SEED", 0))
ARCHITECTURE = os.environ.get("FM_ARCH", "dueling")   # "dueling" ou "vanilla"
CLASSE_MODELE = {"dueling": FerroMobileDuelingDQN, "vanilla": FerroMobileVanillaDQN}[ARCHITECTURE]
MAJ_CIBLE_TOUS_LES_N_STEPS = 200
SAUVER_TOUS_LES_N_EPISODES = int(os.environ.get("FM_SAVE_EVERY", 25))
EVALUER_TOUS_LES_N_EPISODES = 25
PLI = int(os.environ.get("FM_FOLD", 0))
DUREE_MAX_S = 3600 * float(os.environ.get("FM_MAX_HOURS", 11.0 if SUR_KAGGLE else 0))
SYNCHRONISER = os.environ.get("FM_SYNC", "1" if SUR_KAGGLE else "0") == "1"

# Poids de U. w_W = 1 est la référence ; w_W = 5 est la configuration principale du
# chapitre 4 (section 4.6.3). Calcul exact (cellule idéale par segment blanc) : segments
# dont la couverture isolée fait baisser U, en 3G / 4G / 5G :
#   w_W = 1 : 20 / 16 / 11 segments sur 32 ;  w_W = 3 : 2 / 2 / 0 ;  w_W = 5 : 2 / 0 / 0.
POIDS_W = float(os.environ.get("FM_W_W", 1.0))
POIDS_TRANS = float(os.environ.get("FM_W_TRANS", 1.0))
PENALITE_COUT = float(os.environ.get("FM_COST_STEP", 0.0))
POIDS = {"w_Q": 1.0, "w_W": POIDS_W, "w_H": POIDS_TRANS, "w_O": POIDS_TRANS, "w_T": POIDS_TRANS}

_SUFFIXE = ((f"_wT{POIDS_TRANS:g}" if POIDS_TRANS != 1.0 else "")
            + (f"_cost{PENALITE_COUT:g}" if PENALITE_COUT else "")
            + (f"_fold{PLI}" if PLI else ""))
NOM_RUN = os.environ.get("FM_RUN", f"{ARCHITECTURE}_{MODE_EXPLORATION}_wW{POIDS_W:g}{_SUFFIXE}_seed{SEED}")
DOSSIER_CHECKPOINTS = os.path.join(os.environ.get("FM_CHECKPOINTS", os.path.join(RACINE_PROJET, "checkpoints")),
                                   NOM_RUN)
CHEMIN_DERNIER_CHECKPOINT = os.path.join(DOSSIER_CHECKPOINTS, "dernier.pt")
CHEMIN_BUFFER = os.path.join(DOSSIER_CHECKPOINTS, "buffer.npz")
CHEMIN_LOG_TRAIN = os.path.join(DOSSIER_CHECKPOINTS, "train_log.csv")
CHEMIN_LOG_EVAL = os.path.join(DOSSIER_CHECKPOINTS, "eval_log.csv")
COLONNES_LOG_TRAIN = ["episode", "steps", "reward", "n_white", "cout_eur", "nouveaux_sites",
                      "deploiements", "stop", "r_fin", "rewards_nuls", "perte", "epsilon",
                      "total_steps", "duree_s"]
COLONNES_LOG_EVAL = ["episode", "steps", "reward", "n_white", "cout_eur", "nouveaux_sites",
                     "deploiements", "stop", "r_fin", "qos_mean", "qos_min", "N_H", "N_O", "N_T",
                     "U_final_ref", "return_ref", "temps_decision_s"]

# Ce qui définit le problème d'apprentissage : un checkpoint d'une autre configuration est refusé
CONFIG_RUN = {"architecture": CLASSE_MODELE.__name__, "poids": POIDS, "exploration": MODE_EXPLORATION,
              "seed": SEED, "penalite_cout": PENALITE_COUT, "pli": PLI, "budget": BUDGET, "scenario": SCENARIO,
              "t_max": T_MAX, "batch": BATCH_SIZE, "lr": LR, "buffer": CAPACITE_BUFFER,
              "n_steps_exploration": N_STEPS_EXPLORATION, "maj_cible": MAJ_CIBLE_TOUS_LES_N_STEPS}


class Console:
    """Copie la sortie console dans console.log (toutes les sessions, à la suite)."""

    def __init__(self, chemin, flux):
        self.fichier = open(chemin, "a", buffering=1)
        self.flux = flux

    def write(self, texte):
        self.flux.write(texte)
        self.fichier.write(texte)

    def flush(self):
        self.flux.flush()
        self.fichier.flush()


def etat_vers_tenseurs(etat, device):
    """Un état (dict) -> tenseurs de batch 1."""
    grille = torch.as_tensor(etat["grille"], dtype=torch.float32, device=device).unsqueeze(0)
    b_t = torch.tensor([etat["b_t"]], dtype=torch.float32, device=device)
    progression = torch.tensor([etat["progression"]], dtype=torch.float32, device=device)
    return grille, b_t, progression


def probas_boltzmann(q, temperature):
    """P(a) proportionnelle à exp(z_a / T), avec z les Q-values centrées-réduites (éq. 3.38)."""
    z = (q - q.mean()) / (q.std() + 1e-8)
    e = np.exp((z - z.max()) / temperature)
    return e / e.sum()


def choisir_action(modele, etat, masque, epsilon, device, exploration="uniforme"):
    """Epsilon-greedy, toujours parmi les actions faisables (figure 3.21)."""
    faisables = np.nonzero(masque)[0]
    explorer = np.random.random() < epsilon
    if explorer and exploration == "uniforme":
        return int(np.random.choice(faisables))

    with torch.no_grad():
        q = modele(*etat_vers_tenseurs(etat, device)).squeeze(0).cpu().numpy()
    if explorer:
        return int(np.random.choice(faisables, p=probas_boltzmann(q[faisables], TEMPERATURE)))
    return int(np.argmax(np.where(masque, q, -np.inf)))


def calculer_cibles(modele, modele_cible, batch, gamma, device):
    """Cible Double DQN : y = r + gamma * Q_cible(s', argmax_a' Q(s', a')) (éq. 3.36)."""
    grilles = batch["next_grilles"].to(device)
    b_t = batch["next_b_t"].to(device)
    progression = batch["next_progression"].to(device)
    masque = batch["next_masks"].to(device)

    with torch.no_grad():
        q_en_ligne = modele(grilles, b_t, progression).masked_fill(~masque, float("-inf"))
        meilleure_action = q_en_ligne.argmax(dim=1, keepdim=True)
        q_suivant = modele_cible(grilles, b_t, progression).gather(1, meilleure_action).squeeze(1)

    dones = batch["dones"].to(device).float()
    return batch["rewards"].to(device) + gamma * q_suivant * (1.0 - dones)


def entrainer_un_batch(modele, modele_cible, optimiseur, buffer, device):
    batch = buffer.sample(BATCH_SIZE)
    q = modele(batch["grilles"].to(device), batch["b_t"].to(device), batch["progression"].to(device))
    q_action = q.gather(1, batch["actions"].to(device).unsqueeze(1)).squeeze(1)

    perte = nn.functional.smooth_l1_loss(q_action, calculer_cibles(modele, modele_cible, batch, GAMMA, device))
    optimiseur.zero_grad()
    perte.backward()
    optimiseur.step()
    return perte.item()


def verifier_ordre_actions(env, n_configs_radio):
    """Chaque cellule doit avoir les mêmes (g, f) dans le même ordre."""
    actions = env.actions[1:]
    assert len(actions) % n_configs_radio == 0
    reference = [(a[2], a[3]) for a in actions[:n_configs_radio]]
    for debut in range(0, len(actions), n_configs_radio):
        bloc = actions[debut:debut + n_configs_radio]
        assert [(a[2], a[3]) for a in bloc] == reference, f"ordre (g, f) incohérent au bloc {debut}"
        assert len({(a[0], a[1]) for a in bloc}) == 1, f"plusieurs cellules dans le bloc {debut}"
    print(f"verifier_ordre_actions : OK ({n_configs_radio} configs par cellule)")


def verifier_integration(env, modele, device):
    """Un reset et un forward, avant tout entraînement. Vérifie aussi que le masque
    vectorisé est identique à la boucle d'origine sur l'état initial."""
    assert modele.taille_sortie() == len(env.actions), \
        f"sortie du modèle {modele.taille_sortie()} != {len(env.actions)} actions"
    etat = env.reset()
    assert modele.tronc[0].in_channels == etat["grille"].shape[0], "nombre de canaux différent"
    with torch.no_grad():
        q = modele(*etat_vers_tenseurs(etat, device))
    assert q.shape == (1, len(env.actions)) and torch.isfinite(q).all()
    masque = env._actions_faisables()
    assert masque.shape == (len(env.actions),) and masque[0]
    assert np.array_equal(masque, env._actions_faisables_reference()), "masque vectorisé différent"
    print(f"verifier_integration : OK (état {etat['grille'].shape}, |A| = {len(env.actions)}, "
          f"{int(masque.sum())} actions faisables)")


def etats_aleatoires():
    etat = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        etat["cuda"] = torch.cuda.get_rng_state_all()
    return etat


def restaurer_etats_aleatoires(etat):
    random.setstate(etat["python"])
    np.random.set_state(etat["numpy"])
    torch.set_rng_state(etat["torch"].cpu())
    if "cuda" in etat and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in etat["cuda"]])


def sauver_checkpoint(modele, modele_cible, optimiseur, episode, epsilon, total_steps, n_actions, buffer):
    """Écritures atomiques : un arrêt pendant la sauvegarde laisse le checkpoint précédent intact."""
    from kaggle_sync import ecrire_etat
    os.makedirs(DOSSIER_CHECKPOINTS, exist_ok=True)
    etat = {"modele": modele.state_dict(), "modele_cible": modele_cible.state_dict(),
            "optimiseur": optimiseur.state_dict(),
            "episode": episode, "epsilon": epsilon, "total_steps": total_steps, "n_actions": n_actions,
            "architecture": type(modele).__name__, "poids": POIDS, "exploration": MODE_EXPLORATION,
            "seed": SEED, "config": CONFIG_RUN, "rng": etats_aleatoires()}
    buffer.sauver(CHEMIN_BUFFER)
    torch.save(etat, os.path.join(DOSSIER_CHECKPOINTS, f"checkpoint_ep{episode}.pt"))
    torch.save(etat, CHEMIN_DERNIER_CHECKPOINT + ".tmp")
    os.replace(CHEMIN_DERNIER_CHECKPOINT + ".tmp", CHEMIN_DERNIER_CHECKPOINT)
    ecrire_etat(DOSSIER_CHECKPOINTS, {"episode": episode, "total_steps": total_steps, "epsilon": epsilon,
                                      "episodes_prevus": N_EPISODES_TRAIN, "run": NOM_RUN,
                                      "termine": episode >= N_EPISODES_TRAIN, "config": CONFIG_RUN,
                                      "date": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"  checkpoint sauvegardé (épisode {episode}, buffer {len(buffer)} transitions)")


def charger_checkpoint(modele, modele_cible, optimiseur, device, n_actions, buffer):
    """Reprend un entraînement interrompu. Refuse un checkpoint d'une autre configuration."""
    if not os.path.exists(CHEMIN_DERNIER_CHECKPOINT):
        print("Aucun checkpoint : nouvel entraînement.")
        return 0, EPSILON_DEBUT, 0
    etat = torch.load(CHEMIN_DERNIER_CHECKPOINT, map_location=device, weights_only=False)
    if etat.get("n_actions") != n_actions or etat.get("architecture") != type(modele).__name__:
        raise RuntimeError("Checkpoint d'un autre modèle dans ce dossier. Changer FM_RUN.")
    config = etat.get("config")
    if config is not None and config != CONFIG_RUN:
        differences = {k: (config.get(k), v) for k, v in CONFIG_RUN.items() if config.get(k) != v}
        raise RuntimeError(f"Checkpoint d'une autre configuration {differences}. Changer FM_RUN.")
    if etat.get("poids", POIDS) != POIDS:
        raise RuntimeError(f"Checkpoint entraîné avec les poids {etat['poids']}, run actuel {POIDS}.")
    modele.load_state_dict(etat["modele"])
    modele_cible.load_state_dict(etat.get("modele_cible", etat["modele"]))
    optimiseur.load_state_dict(etat["optimiseur"])
    restaurer_etats_aleatoires(etat["rng"])
    if os.path.exists(CHEMIN_BUFFER):
        buffer.charger(CHEMIN_BUFFER)
        print(f"  replay buffer restauré ({len(buffer)} transitions)")
    else:
        print("  ATTENTION : pas de replay buffer sauvegardé, la reprise n'est pas exacte")
    print(f"Reprise à l'épisode {etat['episode']}, epsilon = {etat['epsilon']:.3f}")
    return etat["episode"], etat["epsilon"], etat["total_steps"]


def preparer_log(chemin, colonnes, dernier_episode):
    """Crée le CSV, ou supprime les épisodes > dernier_episode (rejoués après une reprise)."""
    lignes = []
    if os.path.exists(chemin):
        with open(chemin, newline="") as f:
            lignes = [l for l in csv.DictReader(f) if int(l["episode"]) <= dernier_episode]
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colonnes, extrasaction="ignore")
        w.writeheader()
        w.writerows(lignes)


def ajouter_log(chemin, colonnes, ligne):
    with open(chemin, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=colonnes).writerow(ligne)


def jouer_action(env, action):
    """env.step(), plus le comptage des nouveaux sites et la variante de coût (FM_COST_STEP)."""
    nouveau_site = action != "STOP" and (action[0], action[1]) not in env.sites_deja_presents
    budget_avant = env.budget_restant
    etat, reward, done, info = env.step(action)
    if PENALITE_COUT:
        reward -= PENALITE_COUT * (budget_avant - env.budget_restant) / env.budget_total
    return etat, reward, done, info, nouveau_site


def episode_glouton(env, modele, device):
    """Un épisode sans exploration (epsilon = 0). choisir_action tire tout de même un
    np.random.random() par step (comme la version d'origine) : l'évaluation a lieu aux mêmes
    épisodes dans un run continu et dans un run repris, donc la reprise reste exacte."""
    etat, done, t, n_deploiements, n_sites, reward_total, info = env.reset(), False, 0, 0, 0, 0.0, {}
    U0_ref = env.U_reference()
    trace, temps = [], []
    while not done and t < T_MAX:
        debut = time.perf_counter()
        idx = choisir_action(modele, etat, env._actions_faisables(), 0.0, device)
        temps.append(time.perf_counter() - debut)
        action = env.actions[idx]
        n_deploiements += action != "STOP"
        etat, reward, done, info, nouveau = jouer_action(env, action)
        n_sites += nouveau
        reward_total += reward
        t += 1
        trace.append({"t": t, "action": str(action), "nouveau_site": int(nouveau), "reward": reward,
                      "n_white": info.get("n_white"), "cout_eur": env.budget_total - env.budget_restant})
    dp = env._dernier_resultat_dp
    U_final_ref = env.U_reference()
    resume = {"steps": t, "reward": reward_total, "n_white": env._dernier_n_white,
              "cout_eur": env.budget_total - env.budget_restant, "nouveaux_sites": n_sites, "deploiements": n_deploiements,
              "stop": int(n_deploiements < t), "r_fin": info.get("r_fin"),
              "qos_mean": dp["qos_mean"], "qos_min": dp["qos_min"],
              "N_H": dp["N_H"], "N_O": dp["N_O"], "N_T": dp["N_T"],
              "U_final_ref": U_final_ref, "return_ref": U_final_ref - U0_ref + info.get("r_fin", 0.0),
              "temps_decision_s": float(np.mean(temps)) if temps else float("nan")}
    return resume, trace


def evaluer_politique_greedy(env, modele, device, episode):
    """Évaluation gloutonne tous les 25 épisodes (section 4.3.2). N'alimente pas le buffer."""
    resume, trace = episode_glouton(env, modele, device)
    ajouter_log(CHEMIN_LOG_EVAL, COLONNES_LOG_EVAL, {"episode": episode, **resume})
    with open(os.path.join(DOSSIER_CHECKPOINTS, f"eval_trace_ep{episode}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0]))
        w.writeheader()
        w.writerows(trace)
    print(f"  === GREEDY ép. {episode} : {resume['steps']} steps, {resume['deploiements']} déploiements "
          f"({resume['nouveaux_sites']} nouveaux sites), n_white = {resume['n_white']}, "
          f"coût = {resume['cout_eur']}, r_fin = {resume['r_fin']}, Q = {resume['qos_mean']:.4f}, "
          f"Q_min = {resume['qos_min']:.4f}, retour (poids de référence) = {resume['return_ref']:.4f} ===")
    return resume


def boucle_entrainement(env, modele, modele_cible, buffer, optimiseur, device,
                        episode_depart, epsilon, total_steps, sauvegarde=None, debut_session=None):
    """Renvoie l'épisode atteint. S'arrête proprement (checkpoint) si la durée de session
    restante ne suffit plus pour un épisode (FM_MAX_HOURS)."""
    debut_session = debut_session or time.time()
    durees = []
    n_actions = len(env.actions)

    def checkpoint(ep, message):
        sauver_checkpoint(modele, modele_cible, optimiseur, ep, epsilon, total_steps, n_actions, buffer)
        if sauvegarde is not None:
            sauvegarde.envoyer(message)

    for episode in range(episode_depart, N_EPISODES_TRAIN):
        if DUREE_MAX_S and durees:
            # Marge : la plus longue des 5 dernières durées, x 2 si une évaluation suit, + 15 min d'envoi
            prochain = max(durees[-5:]) * (2.0 if (episode + 1) % EVALUER_TOUS_LES_N_EPISODES == 0 else 1.0)
            if time.time() - debut_session + 1.5 * prochain + 900 > DUREE_MAX_S:
                print(f"\nDurée de session presque atteinte : arrêt propre après l'épisode {episode}. "
                      "Relancer le notebook pour continuer (reprise automatique).")
                checkpoint(episode, f"pause de session, épisode {episode}")
                return episode

        debut = time.time()
        etat, done, t, pertes, reward_total, n_nuls, info = env.reset(), False, 0, [], 0.0, 0, {}
        n_sites, n_deploiements = 0, 0

        while not done and t < T_MAX:
            masque = env._actions_faisables()
            mode = MODE_EXPLORATION if total_steps >= N_STEPS_UNIFORME else "uniforme"
            action_idx = choisir_action(modele, etat, masque, epsilon, device, mode)
            etat_suivant, reward, done, info, nouveau = jouer_action(env, env.actions[action_idx])
            n_sites += nouveau
            n_deploiements += action_idx != 0

            # Masque suivant : sans importance si done (pas de bootstrap), STOP seul par sécurité
            if done:
                masque_suivant = np.zeros(n_actions, dtype=bool)
                masque_suivant[0] = True
            else:
                masque_suivant = env._actions_faisables()

            buffer.push(etat["grille"], etat["b_t"], etat["progression"], action_idx, reward,
                        etat_suivant["grille"], etat_suivant["b_t"], etat_suivant["progression"],
                        done, masque_suivant)

            etat = etat_suivant
            reward_total += reward
            n_nuls += abs(reward) < 1e-6
            t += 1
            total_steps += 1
            epsilon = max(EPSILON_FIN, epsilon * EPSILON_DECROISSANCE)

            if len(buffer) >= BATCH_SIZE:
                pertes.append(entrainer_un_batch(modele, modele_cible, optimiseur, buffer, device))
            if total_steps % MAJ_CIBLE_TOUS_LES_N_STEPS == 0:
                modele_cible.load_state_dict(modele.state_dict())

        perte_moy = np.mean(pertes) if pertes else float("nan")
        duree = time.time() - debut
        durees.append(duree)
        print(f"épisode {episode + 1}/{N_EPISODES_TRAIN} : {t} steps, reward = {reward_total:.4f}, "
              f"rewards nuls = {n_nuls}/{t}, perte = {perte_moy:.5f}, epsilon = {epsilon:.3f}, "
              f"n_white = {info.get('n_white')}, coût = {info.get('cout_total')}, "
              f"sites = {n_sites}, déploiements = {n_deploiements}, r_fin = {info.get('r_fin')}, {duree:.0f}s")
        ajouter_log(CHEMIN_LOG_TRAIN, COLONNES_LOG_TRAIN, {
            "episode": episode + 1, "steps": t, "reward": reward_total, "n_white": info.get("n_white"),
            "cout_eur": info.get("cout_total"), "nouveaux_sites": n_sites, "deploiements": n_deploiements,
            "stop": int(n_deploiements < t), "r_fin": info.get("r_fin"), "rewards_nuls": n_nuls,
            "perte": perte_moy, "epsilon": epsilon, "total_steps": total_steps, "duree_s": round(duree, 1)})

        # Évaluation avant la sauvegarde : le checkpoint N contient déjà la ligne eval N
        if (episode + 1) % EVALUER_TOUS_LES_N_EPISODES == 0:
            evaluer_politique_greedy(env, modele, device, episode + 1)
        if (episode + 1) % SAUVER_TOUS_LES_N_EPISODES == 0 and episode + 1 < N_EPISODES_TRAIN:
            checkpoint(episode + 1, f"épisode {episode + 1}")

    checkpoint(N_EPISODES_TRAIN, f"run terminé, épisode {N_EPISODES_TRAIN}")
    return N_EPISODES_TRAIN


def preparer_environnement_colab():
    """Remet CRC_COVLIB_PATH et vérifie le DEM (utile après un redémarrage de session)."""
    if "CRC_COVLIB_PATH" not in os.environ:
        os.environ["CRC_COVLIB_PATH"] = os.path.join(RACINE_PROJET, "crc-covlib", "python-wrapper")
    print(f"CRC_COVLIB_PATH : {os.environ['CRC_COVLIB_PATH']}")
    dem = os.path.join(RACINE_PROJET, "data", "raw", "terrain", "eu_dem_courpiere_ambert.tif")
    if not os.path.exists(dem):
        raise RuntimeError(f"DEM introuvable : {dem}")
    print(f"DEM trouvé : {dem}")


def creer_environnement():
    """Ligne entière (défaut) ou blocs d'entraînement du pli FM_FOLD."""
    from environment_rl import FerroMobileEnv
    if not PLI:
        return FerroMobileEnv(scenario=SCENARIO, budget=BUDGET, t_max=T_MAX, poids=POIDS,
                              sauvegarder_deploiements_rl=False)
    import cv_blocks
    base = FerroMobileEnv(scenario=SCENARIO, budget=BUDGET, t_max=T_MAX, sauvegarder_deploiements_rl=False)
    base.reset()
    config = cv_blocks.preparer_pli(base, PLI, BUDGET)
    print(f"Pli {PLI} : entraînement sur {len(config['points_train'])} points "
          f"({config['n_blancs_train']} blancs), budget {config['budget_train'] / 1e6:.2f} M€, "
          f"{100 * config['part_masquee']:.0f} % des actions masquées (portée vers le bloc de test)")
    del base
    return cv_blocks.env_entrainement(config, scenario=SCENARIO, t_max=T_MAX, poids=POIDS)


def main():
    debut_session = time.time()
    os.makedirs(DOSSIER_CHECKPOINTS, exist_ok=True)

    sauvegarde = None
    if SYNCHRONISER:
        from kaggle_sync import SauvegardeKaggle
        sauvegarde = SauvegardeKaggle(DOSSIER_CHECKPOINTS, NOM_RUN, slug=os.environ.get("FM_KAGGLE_DATASET"))
        sauvegarde.restaurer()     # avant d'ouvrir console.log : il fait partie de la sauvegarde

    sys.stdout = Console(os.path.join(DOSSIER_CHECKPOINTS, "console.log"), sys.stdout)
    print(f"\n===== session du {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    preparer_environnement_colab()   # avant l'import qui charge crc-covlib

    from environment_rl import BANDES_PAR_GENERATION

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    print(f"Device : {device} | architecture : {ARCHITECTURE} | exploration : {MODE_EXPLORATION} | "
          f"seed : {SEED} | poids : {POIDS} | coût dans r_t : {PENALITE_COUT} | pli : {PLI or 'aucun'} | "
          f"épisodes : {N_EPISODES_TRAIN} | run : {DOSSIER_CHECKPOINTS} | "
          f"limite de session : {DUREE_MAX_S / 3600:.1f} h | sauvegarde Kaggle : "
          f"{'oui' if sauvegarde and sauvegarde.actif else 'non'}")
    with open(os.path.join(DOSSIER_CHECKPOINTS, "config_run.json"), "w") as f:
        json.dump({**CONFIG_RUN, "episodes": N_EPISODES_TRAIN, "run": NOM_RUN}, f, indent=1)

    env = creer_environnement()
    n_configs = sum(len(v) for v in BANDES_PAR_GENERATION.values())

    modele = CLASSE_MODELE(n_configs, env.grille.corridor_mask).to(device)
    modele_cible = CLASSE_MODELE(n_configs, env.grille.corridor_mask).to(device)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=LR)

    verifier_integration(env, modele, device)
    verifier_ordre_actions(env, n_configs)

    buffer = ReplayBuffer(capacity=CAPACITE_BUFFER, nx=env.grille.nx, ny=env.grille.ny,
                          n_canaux_etat=N_CANAUX_ETAT, n_actions=len(env.actions), n_canaux_fixes=N_CANAUX_FIXES)
    modele_cible.load_state_dict(modele.state_dict())
    # Chargé après les vérifications : les états aléatoires restaurés ne sont plus perturbés
    episode_depart, epsilon, total_steps = charger_checkpoint(modele, modele_cible, optimiseur, device,
                                                              len(env.actions), buffer)
    if episode_depart >= N_EPISODES_TRAIN:
        print(f"Run déjà terminé ({episode_depart} épisodes). Augmenter FM_EPISODES pour continuer.")
        return
    preparer_log(CHEMIN_LOG_TRAIN, COLONNES_LOG_TRAIN, episode_depart)
    preparer_log(CHEMIN_LOG_EVAL, COLONNES_LOG_EVAL, episode_depart)
    try:
        atteint = boucle_entrainement(env, modele, modele_cible, buffer, optimiseur, device,
                                      episode_depart, epsilon, total_steps, sauvegarde, debut_session)
    except BaseException as e:
        # Le dernier checkpoint reste valable ; on envoie au moins les journaux
        print(f"\nInterruption : {type(e).__name__}: {e}")
        if sauvegarde is not None:
            sauvegarde.envoyer("interruption, journaux seulement")
        raise
    if atteint >= N_EPISODES_TRAIN:
        print(f"\nRun terminé : {N_EPISODES_TRAIN} épisodes. Journaux : {DOSSIER_CHECKPOINTS}")


if __name__ == "__main__":
    main()
