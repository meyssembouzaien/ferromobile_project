"""
train_dqn.py : entraînement du Dueling DQN sur FerroMobileEnv.

Cible Double DQN : le réseau en ligne choisit a', le réseau cible l'évalue.
Avec 156 061 actions, un max simple sur le réseau cible surestime fortement les Q-values.

Exploration (MODE_EXPLORATION) : avec une probabilité epsilon, l'action est tirée
    "uniforme" : uniformément parmi les actions faisables ;
    "softmax"  : selon une loi de Boltzmann sur les Q-values centrées-réduites,
                 après N_STEPS_UNIFORME steps d'exploration uniforme.

Lancer (Colab ou Kaggle, avec dataset et crc-covlib) : python train_dqn.py
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from dqn_model import FerroMobileDuelingDQN, N_CANAUX_ETAT
from replay_buffer import ReplayBuffer

CE_FICHIER = os.path.dirname(os.path.abspath(__file__))
RACINE_PROJET = os.path.dirname(CE_FICHIER)
DOSSIER_CHECKPOINTS = os.path.join(RACINE_PROJET, "checkpoints")
CHEMIN_DERNIER_CHECKPOINT = os.path.join(DOSSIER_CHECKPOINTS, "dernier.pt")

# Hyperparamètres
N_EPISODES_TRAIN = 300
T_MAX = 150
BUDGET = 10_000_000.0
BATCH_SIZE = 32
GAMMA = 1.0                      # épisodique : le retour vaut U_T - U_0 + r_fin
LR = 1e-4
EPSILON_DEBUT, EPSILON_FIN = 1.0, 0.05
N_STEPS_EXPLORATION = 6000       # epsilon atteint EPSILON_FIN après ce nombre de steps
EPSILON_DECROISSANCE = EPSILON_FIN ** (1 / N_STEPS_EXPLORATION)
CAPACITE_BUFFER = 2000           # environ 5,9 Go de RAM avec 28 canaux
MODE_EXPLORATION = "softmax"     # "uniforme" ou "softmax"
TEMPERATURE = 1.0                # softmax : 3 à 9 % de l'exploration sur le 1 % d'actions les mieux notées
N_STEPS_UNIFORME = 1000          # softmax : exploration uniforme tant que le réseau n'a rien appris
SEED = 0
MAJ_CIBLE_TOUS_LES_N_STEPS = 200
SAUVER_TOUS_LES_N_EPISODES = 25
EVALUER_TOUS_LES_N_EPISODES = 25


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


def sauver_checkpoint(modele, optimiseur, episode, epsilon, total_steps, n_actions):
    os.makedirs(DOSSIER_CHECKPOINTS, exist_ok=True)
    etat = {"modele": modele.state_dict(), "optimiseur": optimiseur.state_dict(),
            "episode": episode, "epsilon": epsilon, "total_steps": total_steps, "n_actions": n_actions,
            "architecture": type(modele).__name__}
    torch.save(etat, os.path.join(DOSSIER_CHECKPOINTS, f"checkpoint_ep{episode}.pt"))
    torch.save(etat, CHEMIN_DERNIER_CHECKPOINT)
    print(f"  checkpoint sauvegardé (épisode {episode})")


def charger_checkpoint(modele, optimiseur, device, n_actions):
    """Reprend un entraînement interrompu. Repart de zéro si le checkpoint vient d'un autre modèle."""
    if not os.path.exists(CHEMIN_DERNIER_CHECKPOINT):
        print("Aucun checkpoint : nouvel entraînement.")
        return 0, EPSILON_DEBUT, 0
    etat = torch.load(CHEMIN_DERNIER_CHECKPOINT, map_location=device)
    if etat.get("n_actions") != n_actions or etat.get("architecture") != type(modele).__name__:
        print("Checkpoint d'un autre modèle : ignoré, nouvel entraînement.")
        return 0, EPSILON_DEBUT, 0
    modele.load_state_dict(etat["modele"])
    optimiseur.load_state_dict(etat["optimiseur"])
    print(f"Reprise à l'épisode {etat['episode']}, epsilon = {etat['epsilon']:.3f}")
    return etat["episode"], etat["epsilon"], etat["total_steps"]


def evaluer_politique_greedy(env, modele, device):
    """Un épisode sans exploration (epsilon = 0). N'alimente pas le buffer."""
    etat, done, t, n_deploiements, reward_total, info = env.reset(), False, 0, 0, 0.0, {}
    while not done and t < T_MAX:
        action = env.actions[choisir_action(modele, etat, env._actions_faisables(), 0.0, device)]
        n_deploiements += action != "STOP"
        etat, reward, done, info = env.step(action)
        reward_total += reward
        t += 1
    print(f"  === GREEDY : {t} steps, {n_deploiements} déploiements, reward = {reward_total:.4f}, "
          f"n_white = {info.get('n_white')}, coût = {info.get('cout_total')}, r_fin = {info.get('r_fin')} ===")
    return info


def boucle_entrainement(env, modele, modele_cible, buffer, optimiseur, device,
                        episode_depart, epsilon, total_steps):
    for episode in range(episode_depart, N_EPISODES_TRAIN):
        debut = time.time()
        etat, done, t, pertes, reward_total, n_nuls, info = env.reset(), False, 0, [], 0.0, 0, {}

        while not done and t < T_MAX:
            masque = env._actions_faisables()
            mode = MODE_EXPLORATION if total_steps >= N_STEPS_UNIFORME else "uniforme"
            action_idx = choisir_action(modele, etat, masque, epsilon, device, mode)
            etat_suivant, reward, done, info = env.step(env.actions[action_idx])

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
        print(f"épisode {episode + 1}/{N_EPISODES_TRAIN} : {t} steps, reward = {reward_total:.4f}, "
              f"rewards nuls = {n_nuls}/{t}, perte = {perte_moy:.5f}, epsilon = {epsilon:.3f}, "
              f"n_white = {info.get('n_white')}, r_fin = {info.get('r_fin')}, {time.time() - debut:.0f}s")

        if (episode + 1) % SAUVER_TOUS_LES_N_EPISODES == 0:
            sauver_checkpoint(modele, optimiseur, episode + 1, epsilon, total_steps, len(env.actions))
        if (episode + 1) % EVALUER_TOUS_LES_N_EPISODES == 0:
            evaluer_politique_greedy(env, modele, device)

    sauver_checkpoint(modele, optimiseur, N_EPISODES_TRAIN, epsilon, total_steps, len(env.actions))


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
    print(f"Device : {device} | exploration : {MODE_EXPLORATION} | seed : {SEED}")

    env = FerroMobileEnv(scenario="S1", budget=BUDGET, t_max=T_MAX, sauvegarder_deploiements_rl=False)
    n_configs = sum(len(v) for v in BANDES_PAR_GENERATION.values())

    modele = FerroMobileDuelingDQN(n_configs, env.grille.corridor_mask).to(device)
    modele_cible = FerroMobileDuelingDQN(n_configs, env.grille.corridor_mask).to(device)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=LR)

    episode_depart, epsilon, total_steps = charger_checkpoint(modele, optimiseur, device, len(env.actions))
    modele_cible.load_state_dict(modele.state_dict())

    verifier_integration(env, modele, device)
    verifier_ordre_actions(env, n_configs)

    buffer = ReplayBuffer(capacity=CAPACITE_BUFFER, nx=env.grille.nx, ny=env.grille.ny,
                          n_canaux_etat=N_CANAUX_ETAT, n_actions=len(env.actions))
    boucle_entrainement(env, modele, modele_cible, buffer, optimiseur, device,
                        episode_depart, epsilon, total_steps)