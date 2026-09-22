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
CHEMIN_DERNIER_CHECKPOINT = os.path.join(DOSSIER_CHECKPOINTS, "dernier.pt")

# Hyperparamètres — à ajuster une fois le pipeline validé.
T_MAX_SMOKE_TEST = 150
N_EPISODES_TRAIN = 20  # test rapide de sécurité avec gamma=1.0, sur Colab, avant d'engager les 1000 sur Kaggle
BATCH_SIZE = 32
GAMMA = 1.0  # était 0.99 — épisodique, horizon fini (t_max), récompenses bornées : rien ne justifie gamma<1 ici. Avec gamma=1, le retour du DQN correspond exactement à U_T-U_0+r_fin (télescopage de r_t), cohérent avec le cahier des charges §14. Aucun risque numérique (épisodes toujours <100 steps en pratique, confirmé sur données réelles).
LR = 1e-4
EPSILON_DEBUT = 1.0
EPSILON_FIN = 0.05
# Décroissance PAR STEP, recalibrée pour 100 épisodes x ~65 steps ≈ 6500 steps
# total (Phase 1). Plancher visé vers 90% de ce budget (~5850 steps) :
# 0.05^(1/5850) ≈ 0.999488.
# ATTENTION avant la Phase 2 (1000 épisodes) : recalibrer cette constante
# pour le nouveau budget de steps, ET décider si on repart de zéro ou si
# on continue depuis le checkpoint de la Phase 1 (même piège que la fois
# précédente : reprendre un checkpoint où epsilon est déjà au plancher
# donne très peu d'exploration pour la suite).
EPSILON_DECROISSANCE = 0.999488
CAPACITE_BUFFER = 2000  # ~5.5 Go (2.73 Mo/transition x 2000) — RAM Kaggle réelle ~13 Go, marge gardée
MAJ_CIBLE_TOUS_LES_N_STEPS = 200  # était 20 — beaucoup trop fréquent, annulait l'effet stabilisateur du réseau cible (~3 mises à jour/épisode). 200 donne ~30 mises à jour sur la Phase 1 (6500 steps), plus cohérent avec la pratique standard DQN à cette échelle.
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
    etat = {
        "modele": modele.state_dict(),
        "optimiseur": optimiseur.state_dict(),
        "episode": episode,
        "epsilon": epsilon,
        "total_steps": total_steps,
    }
    chemin = os.path.join(DOSSIER_CHECKPOINTS, f"checkpoint_ep{episode}.pt")
    torch.save(etat, chemin)
    # Copie à nom fixe en plus du fichier numéroté : permet à
    # charger_checkpoint() de retrouver le dernier état sans avoir à
    # parser les noms de fichiers ni trier par date.
    torch.save(etat, CHEMIN_DERNIER_CHECKPOINT)
    print(f"  checkpoint sauvegardé : {chemin}")


def charger_checkpoint(modele, optimiseur, device):
    """Reprend un entraînement interrompu. Restaure modèle + optimiseur
    + epsilon + total_steps + épisode de départ. Le buffer de replay
    repart vide (pas sauvegardé) — acceptable : il se remplit vite
    relativement au nombre total d'épisodes visé (voir CAPACITE_BUFFER
    ci-dessus pour la valeur actuelle), et ne contient de toute façon que
    les dernières transitions, pas un historique qu'on chercherait à
    préserver.

    Retourne (episode_depart, epsilon, total_steps) — episode_depart=0
    si aucun checkpoint trouvé (premier lancement, pas une erreur)."""
    if not os.path.exists(CHEMIN_DERNIER_CHECKPOINT):
        print("Aucun checkpoint trouvé — nouvel entraînement depuis le début.")
        return 0, EPSILON_DEBUT, 0

    etat = torch.load(CHEMIN_DERNIER_CHECKPOINT, map_location=device)
    modele.load_state_dict(etat["modele"])
    optimiseur.load_state_dict(etat["optimiseur"])
    print(f"Checkpoint chargé : reprise à partir de l'épisode {etat['episode']}, "
          f"epsilon={etat['epsilon']:.3f}, total_steps={etat['total_steps']}")
    return etat["episode"], etat["epsilon"], etat["total_steps"]


def evaluer_politique_greedy(env, modele, device, t_max):
    """Évaluation SANS exploration (epsilon=0, argmax pur) — donne le vrai
    niveau actuel de la politique, sans le bruit de l'aléatoire. N'ajoute
    RIEN au buffer de replay et ne compte pas dans total_steps : c'est une
    mesure, pas un step d'entraînement."""
    etat = env.reset()
    done = False
    t = 0
    n_deploiements = 0
    reward_total = 0.0
    r_fin = None
    while not done and t < t_max:
        masque = env._actions_faisables()
        action_idx = choisir_action(modele, etat, masque, epsilon=0.0, device=device)
        action = env.actions[action_idx]
        if action != "STOP":
            n_deploiements += 1
        etat, reward, done, info = env.step(action)
        reward_total += reward
        t += 1
    n_white_final = info.get("n_white")
    cout_total = info.get("cout_total")
    r_fin = info.get("r_fin")

    detail = f", cout_total={cout_total:.0f}€" if cout_total is not None else " (t_max atteint sans fin naturelle)"
    detail_r_fin = f", r_fin={r_fin:.4f}" if r_fin is not None else ""
    print(f"  === ÉVALUATION GREEDY : {t} steps, {n_deploiements} déploiements, "
          f"reward_total={reward_total:.4f}{detail_r_fin}, n_white={n_white_final}{detail} ===")
    return {"n_white": n_white_final, "cout_total": cout_total, "steps": t,
            "n_deploiements": n_deploiements, "reward_total": reward_total, "r_fin": r_fin}


def boucle_entrainement(env, modele, modele_cible, buffer, optimiseur, device,
                          n_episodes, t_max, batch_size, gamma,
                          epsilon_debut, epsilon_fin, epsilon_decroissance,
                          maj_cible_tous_les_n_steps,
                          episode_depart=0, total_steps_depart=0,
                          sauver_tous_les_n_episodes=SAUVER_TOUS_LES_N_EPISODES,
                          evaluer_tous_les_n_episodes=25):
    """episode_depart/total_steps_depart : non-zéro si on reprend depuis
    un checkpoint (charger_checkpoint()). epsilon_debut doit alors déjà
    être la valeur restaurée, pas EPSILON_DEBUT."""
    epsilon = epsilon_debut
    total_steps = total_steps_depart
    debut_total = time.time()
    N_EPISODES_DEBUG = 3 if episode_depart == 0 else 0  # trace détaillée seulement au tout premier lancement

    for episode in range(episode_depart, n_episodes):
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

            if episode < N_EPISODES_DEBUG:
                r_fin_info = f", r_fin={info['r_fin']:.4f}" if "r_fin" in info else ""
                print(f"    step {t}: action={action if action == 'STOP' else action[:2]}..., "
                      f"reward={reward:.4f}{r_fin_info}, n_white={info.get('n_white')}")

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

            # Décroissance PAR STEP, pas par épisode (voir constante).
            epsilon = max(epsilon_fin, epsilon * epsilon_decroissance)

            if len(buffer) >= batch_size:
                perte = entrainer_un_batch(modele, modele_cible, optimiseur, buffer, batch_size, gamma, device)
                pertes.append(perte)

            if total_steps % maj_cible_tous_les_n_steps == 0:
                modele_cible.load_state_dict(modele.state_dict())

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

        if (episode + 1) % evaluer_tous_les_n_episodes == 0:
            evaluer_politique_greedy(env, modele, device, t_max)

    duree_totale = time.time() - debut_total
    episodes_cette_session = n_episodes - episode_depart
    if episodes_cette_session > 0:
        print(f"\nDurée totale : {duree_totale:.1f}s "
              f"({duree_totale / episodes_cette_session:.1f}s/épisode en moyenne, "
              f"{episodes_cette_session} épisodes cette session)")
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
        scenario="S1", budget=10_000_000.0, t_max=T_MAX_SMOKE_TEST,
        sauvegarder_deploiements_rl=False,
    )

    n_configs_radio = sum(len(v) for v in BANDES_PAR_GENERATION.values())

    modele = FerroMobileDQN(n_configs_radio=n_configs_radio, corridor_mask=env.grille.corridor_mask).to(device)
    modele_cible = FerroMobileDQN(n_configs_radio=n_configs_radio, corridor_mask=env.grille.corridor_mask).to(device)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=LR)

    # Reprise automatique si un checkpoint existe déjà (dernier.pt) —
    # sinon episode_depart=0, epsilon=EPSILON_DEBUT, total_steps=0,
    # comme un premier lancement normal.
    episode_depart, epsilon_reprise, total_steps_depart = charger_checkpoint(modele, optimiseur, device)
    modele_cible.load_state_dict(modele.state_dict())
    modele_cible.eval()

    verifier_integration(env, modele, device)
    verifier_ordre_actions(env, n_configs_radio)

    if episode_depart >= N_EPISODES_TRAIN:
        print(f"episode_depart={episode_depart} >= N_EPISODES_TRAIN={N_EPISODES_TRAIN} : "
              "rien à faire, augmente N_EPISODES_TRAIN pour continuer l'entraînement.")
    else:
        buffer = ReplayBuffer(
            capacity=CAPACITE_BUFFER,
            nx=env.grille.nx, ny=env.grille.ny,
            n_canaux_etat=FerroMobileDQN.N_CANAUX_ETAT,
            n_actions=len(env.actions),
        )

        boucle_entrainement(
            env, modele, modele_cible, buffer, optimiseur, device,
            n_episodes=N_EPISODES_TRAIN, t_max=T_MAX_SMOKE_TEST,
            batch_size=BATCH_SIZE, gamma=GAMMA,
            epsilon_debut=epsilon_reprise, epsilon_fin=EPSILON_FIN,
            epsilon_decroissance=EPSILON_DECROISSANCE,
            maj_cible_tous_les_n_steps=MAJ_CIBLE_TOUS_LES_N_STEPS,
            episode_depart=episode_depart, total_steps_depart=total_steps_depart,
        )