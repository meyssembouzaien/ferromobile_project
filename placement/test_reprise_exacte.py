"""
test_reprise_exacte.py : un run interrompu puis repris donne exactement le même résultat
qu'un run continu (section 4.2.2 du mémoire).

Environnement factice, petit et déterministe, avec la même interface que FerroMobileEnv
(28 canaux, dont 25 fixes, STOP en action 0). Le run "interrompu" s'arrête par la limite de
durée de session (FM_MAX_HOURS) après chaque épisode, sauvegarde, puis reprend avec des
objets neufs (modèle, optimiseur, buffer), comme une nouvelle session Kaggle.
Compare : poids finaux, journaux d'entraînement et d'évaluation.

Lancer : python test_reprise_exacte.py
"""

import csv
import os
import tempfile

import numpy as np
import torch

import train_dqn as T
from dqn_model import N_CANAUX_ETAT
from replay_buffer import ReplayBuffer


class Grille:
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.corridor_mask = np.ones((nx, ny), dtype=bool)
        self.corridor_mask[0, 0] = False


class EnvFactice:
    """Chaque déploiement couvre des points blancs selon sa cellule ; récompense = gain."""

    def __init__(self, nx=6, ny=5, n_configs=10, t_max=8, budget=1e6):
        self.grille = Grille(nx, ny)
        self.actions = ["STOP"] + [(ix, iy, "4G", 800 + k) for ix in range(nx) for iy in range(ny)
                                   if self.grille.corridor_mask[ix, iy] for k in range(n_configs)]
        self.t_max, self.budget_total = t_max, budget
        rng = np.random.default_rng(42)
        self.fixes = rng.random((25, nx, ny))
        self.valeur = rng.random(len(self.actions))
        self.n_total = 50

    def reset(self):
        self.t, self.budget_restant, self.fait = 0, self.budget_total, np.zeros(len(self.actions), bool)
        self.sites_deja_presents = set()
        self.couverture = np.zeros((self.grille.nx, self.grille.ny))
        self._maj()
        return self._etat()

    def _maj(self):
        self._dernier_n_white = int(self.n_total - min(self.n_total, 10 * self.couverture.sum()))
        self._dernier_n_eval = self.n_total
        self._dernier_resultat_dp = {"qos_mean": 0.8 + 0.01 * self.couverture.sum(), "qos_min": 0.5,
                                     "N_H": int(self.fait.sum()), "N_O": 0, "N_T": 0}

    def _etat(self):
        mats = np.zeros((1, self.grille.nx, self.grille.ny))
        for idx in np.nonzero(self.fait)[0]:
            mats[0, self.actions[idx][0], self.actions[idx][1]] = 1
        dyn = np.stack([self.couverture, self.couverture ** 2])
        return {"grille": np.concatenate([self.fixes, dyn, mats]), "b_t": self.budget_restant / self.budget_total,
                "progression": self.t / self.t_max}

    def _actions_faisables(self):
        m = ~self.fait & (self.budget_restant >= 50_000)
        m[0] = True
        return m

    _actions_faisables_reference = _actions_faisables

    def U_reference(self):
        from utility import normaliser_metriques, compute_U
        m = normaliser_metriques(self._dernier_resultat_dp, self._dernier_n_eval, self._dernier_n_white, self.n_total)
        return compute_U(m, 1.0, 1.0, 1.0, 1.0, 1.0)

    def step(self, action):
        self.t += 1
        if action == "STOP":
            return self._etat(), -0.1, True, {"n_white": self._dernier_n_white, "cout_total": 0.0, "r_fin": -0.1}
        idx = self.actions.index(action)
        self.fait[idx] = True
        self.sites_deja_presents.add((action[0], action[1]))
        self.budget_restant -= 50_000
        gain = 0.1 * self.valeur[idx]
        self.couverture[action[0], action[1]] += gain
        self._maj()
        done = self.t >= self.t_max
        info = {"n_white": self._dernier_n_white, "cout_total": self.budget_total - self.budget_restant}
        if done:
            info["r_fin"] = -0.5
        return self._etat(), gain - (0.5 if done else 0.0), done, info


def lancer(dossier, n_episodes, duree_max_s):
    """Une ou plusieurs sessions jusqu'à n_episodes. Renvoie les poids finaux."""
    T.N_EPISODES_TRAIN = n_episodes
    T.DOSSIER_CHECKPOINTS = dossier
    T.CHEMIN_DERNIER_CHECKPOINT = os.path.join(dossier, "dernier.pt")
    T.CHEMIN_BUFFER = os.path.join(dossier, "buffer.npz")
    T.CHEMIN_LOG_TRAIN = os.path.join(dossier, "train_log.csv")
    T.CHEMIN_LOG_EVAL = os.path.join(dossier, "eval_log.csv")
    T.EVALUER_TOUS_LES_N_EPISODES = 3
    T.SAUVER_TOUS_LES_N_EPISODES = 3
    T.N_STEPS_UNIFORME = 20          # passe en Boltzmann pendant le test
    T.N_STEPS_EXPLORATION = 60
    T.EPSILON_DECROISSANCE = T.EPSILON_FIN ** (1 / 60)
    T.DUREE_MAX_S = duree_max_s
    T.MAJ_CIBLE_TOUS_LES_N_STEPS = 10
    T.T_MAX = 8
    sessions = 0
    while True:
        sessions += 1
        np.random.seed(0)
        torch.manual_seed(0)
        import random
        random.seed(0)
        env = EnvFactice()
        modele = T.CLASSE_MODELE(10, env.grille.corridor_mask)
        cible = T.CLASSE_MODELE(10, env.grille.corridor_mask)
        opt = torch.optim.Adam(modele.parameters(), lr=T.LR)
        buffer = ReplayBuffer(capacity=40, nx=6, ny=5, n_canaux_etat=N_CANAUX_ETAT,
                              n_actions=len(env.actions), n_canaux_fixes=T.N_CANAUX_FIXES)
        cible.load_state_dict(modele.state_dict())
        ep, eps, steps = T.charger_checkpoint(modele, cible, opt, torch.device("cpu"), len(env.actions), buffer)
        T.preparer_log(T.CHEMIN_LOG_TRAIN, T.COLONNES_LOG_TRAIN, ep)
        T.preparer_log(T.CHEMIN_LOG_EVAL, T.COLONNES_LOG_EVAL, ep)
        atteint = T.boucle_entrainement(env, modele, cible, buffer, opt, torch.device("cpu"), ep, eps, steps)
        if atteint >= n_episodes:
            return modele.state_dict(), sessions


def lire(chemin, ignorer=("duree_s", "temps_decision_s")):
    with open(chemin) as f:
        return [{k: v for k, v in l.items() if k not in ignorer} for l in csv.DictReader(f)]


def test_reprise_exacte():
    with tempfile.TemporaryDirectory() as d:
        continu, n1 = lancer(os.path.join(d, "continu"), 12, 0)
        # 900 s de marge d'envoi : chaque session s'arrête après un épisode
        repris, n2 = lancer(os.path.join(d, "repris"), 12, 900 + 1e-6)
        assert n1 == 1 and n2 > 5, (n1, n2)
        for k in continu:
            assert torch.equal(continu[k], repris[k]), f"poids différents : {k}"
        for nom in ("train_log.csv", "eval_log.csv"):
            a, b = lire(os.path.join(d, "continu", nom)), lire(os.path.join(d, "repris", nom))
            assert a == b, f"{nom} différent"
        print(f"test_reprise_exacte : OK (run continu = run en {n2} sessions : poids et journaux identiques)")


if __name__ == "__main__":
    test_reprise_exacte()
