"""
train_dqn.py — Entraînement DQN vanilla sur FerroMobileEnv.

Ordre d'exécution : verifier_integration() d'abord (léger, pas de
simulation P.1812), puis la boucle d'entraînement (coûteuse : chaque
action de déploiement déclenche une vraie simulation P.1812).

Lancer sur Colab (nécessite dataset réel + crc-covlib) :
    python train_dqn.py
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from dqn_model import FerroMobileDQN
from replay_buffer import ReplayBuffer

CE_FICHIER = os.path.dirname(os.path.abspath(__file__))    # .../placement
RACINE_PROJET = os.path.dirname(CE_FICHIER)                 # .../ferromobile_project
DOSSIER_CHECKPOINTS = os.path.join(RACINE_PROJET, "checkpoints")

# Hyperparamètres du smoke test — à ajuster une fois le pipeline validé.
T_MAX_SMOKE_TEST = 20
N_EPISODES_TRAIN = 250
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
EPSILON_DEBUT = 1.0
EPSILON_FIN = 0.05
EPSILON_DECROISSANCE = 0.9  # multiplié à chaque fin d'épisode
CAPACITE_BUFFER = 500
MAJ_CIBLE_TOUS_LES_N_STEPS = 20
SAUVER_TOUS_LES_N_EPISODES = 25  # checkpoint disque


def etat_vers_tenseurs(etat, device):
    """etat : dict {"grille", "b_t", "progression"} d'un seul état.
    Retourne des tenseurs batch=1, sur device."""
    grille = torch.as_tensor(etat["grille"], dtype=torch.float32, device=device).unsqueeze(0)
    b_t = torch.tensor([etat["b_t"]], dtype=torch.float32, device=device)
    progression = torch.tensor([etat["progression"]], dtype=torch.float32, device=device)
    return grille, b_t, progression


def choisir_action(modele, etat, masque, epsilon, device):
    """Epsilon-greedy. Exploration ET exploitation restent DANS le
    masque de faisabilité — sinon env.step() lève une erreur."""
    indices_faisables = np.nonzero(masque)[0]

    if np.random.random() < epsilon:
        return int(np.random.choice(indices_faisables))

    grille, b_t, progression = etat_vers_tenseurs(etat, device)
    with torch.no_grad():
        q = modele(grille, b_t, progression).squeeze(0).cpu().numpy()

    q_masquee = np.where(masque, q, -np.inf)
    return int(np.argmax(q_masquee))


def calculer_cibles(modele_cible, batch, gamma, device):
    """Cible TD vanilla : y = r + gamma * max Q_cible(s', a') masqué.
    Si done : y = r (pas de bootstrap)."""
    with torch.no_grad():
        q_next = modele_cible(
            batch["next_grilles"].to(device),
            batch["next_b_t"].to(device),
            batch["next_progression"].to(device),
        )
        masque = batch["next_masks"].to(device)
        q_next_masque = q_next.masked_fill(~masque, float("-inf"))
        max_q_next, _ = q_next_masque.max(dim=1)
        # STOP toujours faisable donc max toujours fini ; filet de
        # sécurité seulement si un état stocké n'a aucune action faisable.
        max_q_next = torch.nan_to_num(max_q_next, neginf=0.0)

    dones = batch["dones"].to(device).float()
    rewards = batch["rewards"].to(device)
    return rewards + gamma * max_q_next * (1.0 - dones)


def entrainer_un_batch(modele, modele_cible, optimiseur, buffer, batch_size, gamma, device):
    batch = buffer.sample(batch_size)

    q_pred = modele(
        batch["grilles"].to(device),
        batch["b_t"].to(device),
        batch["progression"].to(device),
    )
    q_pred_a = q_pred.gather(1, batch["actions"].to(device).unsqueeze(1)).squeeze(1)

    cibles = calculer_cibles(modele_cible, batch, gamma, device)

    perte = nn.functional.smooth_l1_loss(q_pred_a, cibles)
    optimiseur.zero_grad()
    perte.backward()
    optimiseur.step()
    return perte.item()


def verifier_ordre_actions(env, n_configs_radio):
    """Vérifie que chaque cellule utilise le même ordre (g,f) dans
    env.actions — la garantie dont dépend l'ordre des Q-values."""
    actions_deploiement = env.actions[1:]  # sans STOP
    assert len(actions_deploiement) % n_configs_radio == 0

    bloc_reference = None
    for debut in range(0, len(actions_deploiement), n_configs_radio):
        bloc = actions_deploiement[debut:debut + n_configs_radio]
        configs = [(a[2], a[3]) for a in bloc]

        assert len(set(configs)) == n_configs_radio, f"configs dupliquées, bloc {debut}"

        if bloc_reference is None:
            bloc_reference = configs
        else:
            assert configs == bloc_reference, f"ordre (g,f) incohérent au bloc {debut}"

    print(f"verifier_ordre_actions : OK ({n_configs_radio} configs radio par cellule)")


def verifier_integration(env, modele, device):
    """À lancer AVANT tout entraînement.
    Un reset() et un forward DQN, sans step() de déploiement."""
    attendu = modele.taille_sortie_attendue()
    reel = len(env.actions)
    assert attendu == reel, f"taille_sortie_attendue={attendu} != len(env.actions)={reel}"

    etat = env.reset()
    grille, b_t, progression = etat_vers_tenseurs(etat, device)

    with torch.no_grad():
        q = modele(grille, b_t, progression)

    assert q.shape == (1, reel), f"Q shape {tuple(q.shape)}, attendu (1, {reel})"
    assert torch.isfinite(q).all(), "Q-values non finies"

    masque = env._actions_faisables()
    assert masque.shape == (reel,), f"masque shape {masque.shape}, attendu ({reel},)"
    assert masque[0], "STOP doit toujours être faisable"

    print(f"verifier_integration : OK (|A|={reel}, Q shape={tuple(q.shape)})")


def sauver_checkpoint(modele, optimiseur, episode, epsilon, total_steps):
    os.makedirs(DOSSIER_CHECKPOINTS, exist_ok=True)
    chemin = os.path.join(DOSSIER_CHECKPOINTS, f"checkpoint_ep{episode}.pt")
    torch.save({
        "modele": modele.state_dict(),
        "optimiseur": optimiseur.state_dict(),
        "episode": episode,
        "epsilon": epsilon,
        "total_steps": total_steps,
    }, chemin)
    print(f"  checkpoint sauvegardé : {chemin}")


def boucle_entrainement(env, modele, modele_cible, buffer, optimiseur, device,
                          n_episodes, t_max, batch_size, gamma,
                          epsilon_debut, epsilon_fin, epsilon_decroissance,
                          maj_cible_tous_les_n_steps,
                          sauver_tous_les_n_episodes=SAUVER_TOUS_LES_N_EPISODES):
    epsilon = epsilon_debut
    total_steps = 0
    debut_total = time.time()

    for episode in range(n_episodes):
        debut_episode = time.time()
        etat = env.reset()
        done = False
        t = 0
        pertes = []
        reward_total = 0.0

        while not done and t < t_max:
            masque = env._actions_faisables()
            action_idx = choisir_action(modele, etat, masque, epsilon, device)
            action = env.actions[action_idx]

            etat_suivant, reward, done, info = env.step(action)

            # Masque suivant : réel si pas terminal, sinon un remplisseur
            # sûr (jamais utilisé dans la cible, voir calculer_cibles).
            if done:
                masque_suivant = np.zeros(len(env.actions), dtype=bool)
                masque_suivant[0] = True
            else:
                masque_suivant = env._actions_faisables()

            buffer.push(
                etat["grille"], etat["b_t"], etat["progression"],
                action_idx, reward,
                etat_suivant["grille"], etat_suivant["b_t"], etat_suivant["progression"],
                done, masque_suivant,
            )

            etat = etat_suivant
            reward_total += reward
            t += 1
            total_steps += 1

            if len(buffer) >= batch_size:
                perte = entrainer_un_batch(modele, modele_cible, optimiseur, buffer, batch_size, gamma, device)
                pertes.append(perte)

            if total_steps % maj_cible_tous_les_n_steps == 0:
                modele_cible.load_state_dict(modele.state_dict())

        epsilon = max(epsilon_fin, epsilon * epsilon_decroissance)
        perte_moy = sum(pertes) / len(pertes) if pertes else float("nan")
        duree_episode = time.time() - debut_episode

        # _finaliser() (le seul chemin qui met done=True) renvoie r_fin/
        # cout_total/n_white dans info — donc dès que done=True, ces clés
        # existent. Si l'épisode s'arrête sans done=True (t_max externe
        # atteint avant que l'env ne se termine lui-même), elles n'y sont pas.
        raison_fin = "env_done" if done else "t_max_externe"
        detail = ""
        if "r_fin" in info:
            detail = f", cout_total={info['cout_total']:.0f}€, n_white={info['n_white']}, r_fin={info['r_fin']:.4f}"

        print(f"épisode {episode+1}/{n_episodes} : {t} steps, "
              f"reward total={reward_total:.4f}, perte moy={perte_moy:.4f}, "
              f"epsilon={epsilon:.3f}, durée={duree_episode:.1f}s, "
              f"fin={raison_fin}{detail}")

        if (episode + 1) % sauver_tous_les_n_episodes == 0:
            sauver_checkpoint(modele, optimiseur, episode + 1, epsilon, total_steps)

    duree_totale = time.time() - debut_total
    print(f"\nDurée totale : {duree_totale:.1f}s ({duree_totale / n_episodes:.1f}s/épisode en moyenne)")
    sauver_checkpoint(modele, optimiseur, n_episodes, epsilon, total_steps)  # checkpoint final, toujours


def preparer_environnement_colab():
    """Un redémarrage de session Colab efface les variables d'env et les
    modules importés (mais pas les fichiers sur disque). Cette fonction
    remet CRC_COVLIB_PATH et vérifie le DEM, pour ne plus avoir à
    relancer les cellules de setup à la main à chaque redémarrage."""
    if "CRC_COVLIB_PATH" not in os.environ:
        os.environ["CRC_COVLIB_PATH"] = os.path.join(RACINE_PROJET, "crc-covlib", "python-wrapper")
    print(f"CRC_COVLIB_PATH : {os.environ['CRC_COVLIB_PATH']}")

    dem_path = os.path.join(RACINE_PROJET, "data", "raw", "terrain", "eu_dem_courpiere_ambert.tif")
    if not os.path.exists(dem_path):
        raise RuntimeError(
            f"DEM introuvable : {dem_path}\n"
            "Le symlink vers Drive a disparu (redémarrage complet du runtime, "
            "pas juste de la session) — relance la cellule mkdir+ln -s du notebook."
        )
    print(f"DEM trouvé : {dem_path}")


if __name__ == "__main__":
    preparer_environnement_colab()  # avant l'import qui charge crc_covlib

    from environment_rl import FerroMobileEnv, BANDES_PAR_GENERATION

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    env = FerroMobileEnv(
        scenario="S1", budget=500_000.0, t_max=T_MAX_SMOKE_TEST,
        sauvegarder_deploiements_rl=False,
    )

    n_configs_radio = sum(len(v) for v in BANDES_PAR_GENERATION.values())

    modele = FerroMobileDQN(n_configs_radio=n_configs_radio, corridor_mask=env.grille.corridor_mask).to(device)
    modele_cible = FerroMobileDQN(n_configs_radio=n_configs_radio, corridor_mask=env.grille.corridor_mask).to(device)
    modele_cible.load_state_dict(modele.state_dict())
    modele_cible.eval()

    verifier_integration(env, modele, device)
    verifier_ordre_actions(env, n_configs_radio)

    buffer = ReplayBuffer(
        capacity=CAPACITE_BUFFER,
        nx=env.grille.nx, ny=env.grille.ny,
        n_canaux_etat=FerroMobileDQN.N_CANAUX_ETAT,
        n_actions=len(env.actions),
    )

    optimiseur = torch.optim.Adam(modele.parameters(), lr=LR)

    boucle_entrainement(
        env, modele, modele_cible, buffer, optimiseur, device,
        n_episodes=N_EPISODES_TRAIN, t_max=T_MAX_SMOKE_TEST,
        batch_size=BATCH_SIZE, gamma=GAMMA,
        epsilon_debut=EPSILON_DEBUT, epsilon_fin=EPSILON_FIN,
        epsilon_decroissance=EPSILON_DECROISSANCE,
        maj_cible_tous_les_n_steps=MAJ_CIBLE_TOUS_LES_N_STEPS,
    )