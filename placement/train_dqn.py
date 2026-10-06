"""
train_dqn.py : entraînement du Dueling DQN sur FerroMobileEnv.

Cible Double DQN : le réseau en ligne choisit a', le réseau cible l'évalue.
Avec 156 061 actions, un max simple sur le réseau cible surestime fortement les Q-values.

Exploration (MODE_EXPLORATION) : avec une probabilité epsilon, l'action est tirée
    "uniforme" : uniformément parmi les actions faisables ;
    "softmax"  : selon une loi de Boltzmann sur les Q-values centrées-réduites,
                 après N_STEPS_UNIFORME steps d'exploration uniforme.

Lancer (Colab ou Kaggle, avec dataset et crc-covlib) : python train_dqn.py

Réglages sans modifier le code (variables d'environnement) :
    FM_EPISODES=1000      nombre total d'épisodes (défaut 300)
    FM_W_W=5              poids w_W des zones blanches dans U (défaut 1, valeur de référence)
    FM_EXPLORATION=uniforme, FM_SEED=1, FM_SAVE_EVERY=5, FM_SAVE_BUFFER=1
    FM_RUN=nom            sous-dossier de checkpoints/ (un run = un dossier)

Journaux (dans le dossier du run) :
    train_log.csv : une ligne par épisode (coût, sites, déploiements inclus)
    eval_log.csv  : une ligne par évaluation gloutonne
    eval_trace_epN.csv : n_white et coût après chaque action de l'évaluation N
À la reprise, les lignes des épisodes postérieurs au checkpoint sont supprimées
(ils vont être rejoués), donc aucun épisode n'apparaît deux fois.
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from dqn_model import FerroMobileDuelingDQN, N_CANAUX_ETAT
from replay_buffer import ReplayBuffer

CE_FICHIER = os.path.dirname(os.path.abspath(__file__))
RACINE_PROJET = os.path.dirname(CE_FICHIER)

# Hyperparamètres
N_EPISODES_TRAIN = int(os.environ.get("FM_EPISODES", 300))
T_MAX = 150
BUDGET = 10_000_000.0
BATCH_SIZE = 32
GAMMA = 1.0                      # épisodique : le retour vaut U_T - U_0 + r_fin
LR = 1e-4
EPSILON_DEBUT, EPSILON_FIN = 1.0, 0.05
N_STEPS_EXPLORATION = 6000       # epsilon atteint EPSILON_FIN après ce nombre de steps
EPSILON_DECROISSANCE = EPSILON_FIN ** (1 / N_STEPS_EXPLORATION)
CAPACITE_BUFFER = 2000           # environ 5,9 Go de RAM avec 28 canaux
MODE_EXPLORATION = os.environ.get("FM_EXPLORATION", "softmax")   # "uniforme" ou "softmax"
TEMPERATURE = 1.0                # softmax : 3 à 9 % de l'exploration sur le 1 % d'actions les mieux notées
N_STEPS_UNIFORME = 1000          # softmax : exploration uniforme tant que le réseau n'a rien appris
SEED = int(os.environ.get("FM_SEED", 0))
MAJ_CIBLE_TOUS_LES_N_STEPS = 200
SAUVER_TOUS_LES_N_EPISODES = int(os.environ.get("FM_SAVE_EVERY", 25))
EVALUER_TOUS_LES_N_EPISODES = 25
SAUVER_BUFFER = os.environ.get("FM_SAVE_BUFFER", "0") == "1"   # ~6 Go par sauvegarde

# Poids de U. w_W = 1 est la référence. Avec w_W = 1, couvrir seul un segment blanc
# de moins de 4 à 6 points fait baisser U (les changements de cellule ajoutés pèsent
# autant que les points couverts) : seule r_fin récompense les derniers points.
# Calcul exact sur les données (cellule idéale par segment blanc) : segments dont la
# couverture isolée fait baisser U, en 3G / 4G / 5G :
#   w_W = 1 : 20 / 16 / 11 segments sur 32 ;  w_W = 3 : 2 / 2 / 0 ;  w_W = 5 : 2 / 0 / 0.
# w_W > 1 est une variante de récompense, à déclarer comme telle (analyse de sensibilité).
POIDS_W = float(os.environ.get("FM_W_W", 1.0))
POIDS = {"w_Q": 1.0, "w_W": POIDS_W, "w_H": 1.0, "w_O": 1.0, "w_T": 1.0}

NOM_RUN = os.environ.get("FM_RUN", f"{MODE_EXPLORATION}_wW{POIDS_W:g}_seed{SEED}")
DOSSIER_CHECKPOINTS = os.path.join(RACINE_PROJET, "checkpoints", NOM_RUN)
CHEMIN_DERNIER_CHECKPOINT = os.path.join(DOSSIER_CHECKPOINTS, "dernier.pt")
CHEMIN_ANCIEN_CHECKPOINT = os.path.join(RACINE_PROJET, "checkpoints", "dernier.pt")
CHEMIN_BUFFER = os.path.join(DOSSIER_CHECKPOINTS, "buffer.pt")
CHEMIN_LOG_TRAIN = os.path.join(DOSSIER_CHECKPOINTS, "train_log.csv")
CHEMIN_LOG_EVAL = os.path.join(DOSSIER_CHECKPOINTS, "eval_log.csv")
COLONNES_LOG_TRAIN = ["episode", "steps", "reward", "n_white", "cout_eur", "nouveaux_sites",
                      "deploiements", "stop", "r_fin", "rewards_nuls", "perte", "epsilon",
                      "total_steps", "duree_s"]
COLONNES_LOG_EVAL = ["episode", "steps", "reward", "n_white", "cout_eur", "nouveaux_sites",
                     "deploiements", "stop", "r_fin", "qos_mean", "qos_min", "N_H", "N_O", "N_T"]


def etat_vers_tenseurs(etat, device):
    """Un état (dict) -> tenseurs de batch 1."""
    grille = torch.as_tensor(etat["grille"], dtype=torch.float32, device=device).unsqueeze(0)
    b_t = torch.tensor([etat["b_t"]], dtype=torch.float32, device=device)
    progression = torch.tensor([etat["progression"]], dtype=torch.float32, device=device)
    return grille, b_t, progression


def probas_boltzmann(q, temperature):
    """P(a) proportionnelle à exp(z_a / T), avec z les Q-values centrées-réduites."""
    z = (q - q.mean()) / (q.std() + 1e-8)
    e = np.exp((z - z.max()) / temperature)
    return e / e.sum()


def choisir_action(modele, etat, masque, epsilon, device, exploration="uniforme"):
    """Epsilon-greedy, toujours parmi les actions faisables."""
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
    """Cible Double DQN : y = r + gamma * Q_cible(s', argmax_a' Q(s', a'))."""
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
    """Un reset et un forward, avant tout entraînement."""
    assert modele.taille_sortie() == len(env.actions), \
        f"sortie du modèle {modele.taille_sortie()} != {len(env.actions)} actions"
    etat = env.reset()
    assert modele.tronc[0].in_channels == etat["grille"].shape[0], "nombre de canaux différent"
    with torch.no_grad():
        q = modele(*etat_vers_tenseurs(etat, device))
    assert q.shape == (1, len(env.actions)) and torch.isfinite(q).all()
    masque = env._actions_faisables()
    assert masque.shape == (len(env.actions),) and masque[0]
    print(f"verifier_integration : OK (état {etat['grille'].shape}, |A| = {len(env.actions)})")


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


def sauver_checkpoint(modele, modele_cible, optimiseur, episode, epsilon, total_steps, n_actions,
                      buffer=None):
    os.makedirs(DOSSIER_CHECKPOINTS, exist_ok=True)
    etat = {"modele": modele.state_dict(), "modele_cible": modele_cible.state_dict(),
            "optimiseur": optimiseur.state_dict(),
            "episode": episode, "epsilon": epsilon, "total_steps": total_steps, "n_actions": n_actions,
            "architecture": type(modele).__name__, "poids": POIDS, "exploration": MODE_EXPLORATION,
            "seed": SEED, "rng": etats_aleatoires()}
    torch.save(etat, os.path.join(DOSSIER_CHECKPOINTS, f"checkpoint_ep{episode}.pt"))
    torch.save(etat, CHEMIN_DERNIER_CHECKPOINT)
    if buffer is not None and SAUVER_BUFFER:
        torch.save(buffer.state_dict(), CHEMIN_BUFFER)
    print(f"  checkpoint sauvegardé (épisode {episode})")


def charger_checkpoint(modele, modele_cible, optimiseur, device, n_actions, buffer=None):
    """Reprend un entraînement interrompu : poids, réseau cible, optimiseur, epsilon,
    états aléatoires et, si sauvegardé, replay buffer. Refuse un checkpoint d'un autre
    modèle ou d'une autre récompense (ce serait un autre problème d'apprentissage)."""
    chemin = CHEMIN_DERNIER_CHECKPOINT
    if not os.path.exists(chemin) and POIDS_W == 1.0 and os.path.exists(CHEMIN_ANCIEN_CHECKPOINT):
        chemin = CHEMIN_ANCIEN_CHECKPOINT   # run lancé avant les sous-dossiers par run (poids de référence)
        print(f"Reprise depuis l'ancien emplacement : {chemin}")
    if not os.path.exists(chemin):
        print("Aucun checkpoint : nouvel entraînement.")
        return 0, EPSILON_DEBUT, 0
    etat = torch.load(chemin, map_location=device, weights_only=False)
    if etat.get("n_actions") != n_actions or etat.get("architecture") != type(modele).__name__:
        print("Checkpoint d'un autre modèle : ignoré, nouvel entraînement.")
        return 0, EPSILON_DEBUT, 0
    if etat.get("poids", POIDS) != POIDS:
        raise RuntimeError(f"Checkpoint entraîné avec les poids {etat['poids']}, run actuel {POIDS}. "
                           "Changer FM_RUN pour démarrer un nouveau run.")
    modele.load_state_dict(etat["modele"])
    modele_cible.load_state_dict(etat.get("modele_cible", etat["modele"]))
    optimiseur.load_state_dict(etat["optimiseur"])
    if "rng" in etat:
        restaurer_etats_aleatoires(etat["rng"])
    else:
        print("  ancien checkpoint sans états aléatoires : graines non restaurées")
    if buffer is not None and os.path.exists(CHEMIN_BUFFER):
        buffer.load_state_dict(torch.load(CHEMIN_BUFFER, weights_only=False))
        print(f"  replay buffer restauré ({len(buffer)} transitions)")
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
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        w.writerows(lignes)


def ajouter_log(chemin, colonnes, ligne):
    with open(chemin, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=colonnes).writerow(ligne)


def jouer_action(env, action):
    """env.step(), plus le comptage des nouveaux sites (cellule sans site avant l'action)."""
    nouveau_site = action != "STOP" and (action[0], action[1]) not in env.sites_deja_presents
    etat, reward, done, info = env.step(action)
    return etat, reward, done, info, nouveau_site


def evaluer_politique_greedy(env, modele, device, episode):
    """Un épisode sans exploration (epsilon = 0). N'alimente pas le buffer.
    N'utilise pas les générateurs aléatoires : l'entraînement n'est pas perturbé."""
    etat, done, t, n_deploiements, n_sites, reward_total, info = env.reset(), False, 0, 0, 0, 0.0, {}
    trace = []
    while not done and t < T_MAX:
        action = env.actions[choisir_action(modele, etat, env._actions_faisables(), 0.0, device)]
        n_deploiements += action != "STOP"
        etat, reward, done, info, nouveau = jouer_action(env, action)
        n_sites += nouveau
        reward_total += reward
        t += 1
        trace.append({"t": t, "action": str(action), "nouveau_site": int(nouveau), "reward": reward,
                      "n_white": info.get("n_white"),
                      "cout_eur": env.budget_total - env.budget_restant})
    dp = env._dernier_resultat_dp
    ligne = {"episode": episode, "steps": t, "reward": reward_total, "n_white": info.get("n_white"),
             "cout_eur": info.get("cout_total"), "nouveaux_sites": n_sites, "deploiements": n_deploiements,
             "stop": int(n_deploiements < t), "r_fin": info.get("r_fin"),
             "qos_mean": dp["qos_mean"], "qos_min": dp["qos_min"],
             "N_H": dp["N_H"], "N_O": dp["N_O"], "N_T": dp["N_T"]}
    ajouter_log(CHEMIN_LOG_EVAL, COLONNES_LOG_EVAL, ligne)
    with open(os.path.join(DOSSIER_CHECKPOINTS, f"eval_trace_ep{episode}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0]))
        w.writeheader()
        w.writerows(trace)
    print(f"  === GREEDY : {t} steps, {n_deploiements} déploiements ({n_sites} nouveaux sites), "
          f"reward = {reward_total:.4f}, n_white = {info.get('n_white')}, coût = {info.get('cout_total')}, "
          f"r_fin = {info.get('r_fin')}, Q = {dp['qos_mean']:.4f}, Q_min = {dp['qos_min']:.4f} ===")
    return info


def boucle_entrainement(env, modele, modele_cible, buffer, optimiseur, device,
                        episode_depart, epsilon, total_steps):
    for episode in range(episode_depart, N_EPISODES_TRAIN):
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
                masque_suivant = np.zeros(len(env.actions), dtype=bool)
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
        if (episode + 1) % SAUVER_TOUS_LES_N_EPISODES == 0:
            sauver_checkpoint(modele, modele_cible, optimiseur, episode + 1, epsilon, total_steps,
                              len(env.actions), buffer)

    sauver_checkpoint(modele, modele_cible, optimiseur, N_EPISODES_TRAIN, epsilon, total_steps,
                      len(env.actions), buffer)


def preparer_environnement_colab():
    """Remet CRC_COVLIB_PATH et vérifie le DEM (utile après un redémarrage de session)."""
    if "CRC_COVLIB_PATH" not in os.environ:
        os.environ["CRC_COVLIB_PATH"] = os.path.join(RACINE_PROJET, "crc-covlib", "python-wrapper")
    print(f"CRC_COVLIB_PATH : {os.environ['CRC_COVLIB_PATH']}")
    dem = os.path.join(RACINE_PROJET, "data", "raw", "terrain", "eu_dem_courpiere_ambert.tif")
    if not os.path.exists(dem):
        raise RuntimeError(f"DEM introuvable : {dem}")
    print(f"DEM trouvé : {dem}")


if __name__ == "__main__":
    preparer_environnement_colab()   # avant l'import qui charge crc-covlib

    from environment_rl import FerroMobileEnv, BANDES_PAR_GENERATION

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    print(f"Device : {device} | exploration : {MODE_EXPLORATION} | seed : {SEED} | poids : {POIDS} "
          f"| épisodes : {N_EPISODES_TRAIN} | run : {DOSSIER_CHECKPOINTS}")

    env = FerroMobileEnv(scenario="S1", budget=BUDGET, t_max=T_MAX, poids=POIDS,
                         sauvegarder_deploiements_rl=False)
    n_configs = sum(len(v) for v in BANDES_PAR_GENERATION.values())

    modele = FerroMobileDuelingDQN(n_configs, env.grille.corridor_mask).to(device)
    modele_cible = FerroMobileDuelingDQN(n_configs, env.grille.corridor_mask).to(device)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=LR)

    verifier_integration(env, modele, device)
    verifier_ordre_actions(env, n_configs)

    buffer = ReplayBuffer(capacity=CAPACITE_BUFFER, nx=env.grille.nx, ny=env.grille.ny,
                          n_canaux_etat=N_CANAUX_ETAT, n_actions=len(env.actions))
    modele_cible.load_state_dict(modele.state_dict())
    # Chargé après les vérifications : les états aléatoires restaurés ne sont plus perturbés
    episode_depart, epsilon, total_steps = charger_checkpoint(modele, modele_cible, optimiseur, device,
                                                              len(env.actions), buffer)
    preparer_log(CHEMIN_LOG_TRAIN, COLONNES_LOG_TRAIN, episode_depart)
    preparer_log(CHEMIN_LOG_EVAL, COLONNES_LOG_EVAL, episode_depart)
    boucle_entrainement(env, modele, modele_cible, buffer, optimiseur, device,
                        episode_depart, epsilon, total_steps)