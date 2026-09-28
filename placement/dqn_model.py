"""
dqn_model.py : Dueling DQN pour FerroMobile.

Entrée : grille (28, nx, ny) + b_t + progression.
Sortie : Q-values dans le même ordre que env.actions (STOP en position 0).

Q(s, a) = V(s) + A(s, a) - moyenne des A(s, a')

Tronc : 5 convolutions 3 x 3 dilatées (1, 2, 4, 8, 16). Champ récepteur de 63 x 63 cellules
(environ 15,75 km) : chaque cellule voit les zones blanches à plusieurs kilomètres.

Lancer : python dqn_model.py (tests locaux, sans dataset)
"""

import numpy as np
import torch
import torch.nn as nn

N_CANAUX_ETAT = 28   # doit être égal à build_state_dynamic.N_CANAUX_ETAT
DILATATIONS = (1, 2, 4, 8, 16)


class FerroMobileDuelingDQN(nn.Module):

    def __init__(self, n_configs_radio, corridor_mask, n_canaux=N_CANAUX_ETAT, largeur=64):
        super().__init__()
        self.n_configs_radio = n_configs_radio

        # Tronc convolutif dilaté : même taille de grille en sortie (padding = dilatation)
        couches, entree = [], n_canaux
        for d in DILATATIONS:
            couches += [nn.Conv2d(entree, largeur, kernel_size=3, padding=d, dilation=d), nn.ReLU()]
            entree = largeur
        self.tronc = nn.Sequential(*couches)

        # V(s) et A(s, STOP) : caractéristiques moyennées sur toute la grille, + b_t et progression
        self.tete_valeur = nn.Sequential(nn.Linear(largeur + 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.tete_stop = nn.Sequential(nn.Linear(largeur + 2, 64), nn.ReLU(), nn.Linear(64, 1))

        # A(s, a) d'un déploiement : caractéristiques locales de sa cellule, + b_t et progression
        self.tete_avantage = nn.Conv2d(largeur + 2, n_configs_radio, kernel_size=1)

        # Cellules du corridor, ordre C (ix puis iy) = ordre de env.actions
        assert corridor_mask.dtype == bool
        indices = np.nonzero(corridor_mask.reshape(-1))[0]
        self.register_buffer("indices_actions", torch.as_tensor(indices, dtype=torch.long))

    def taille_sortie(self):
        """|A| = 1 (STOP) + cellules du corridor x configs radio."""
        return 1 + len(self.indices_actions) * self.n_configs_radio

    def forward(self, grille, b_t, progression):
        """grille : (B, 28, nx, ny) ; b_t, progression : (B,). Retourne Q : (B, |A|)."""
        x = self.tronc(grille)
        B, _, H, W = x.shape
        scalaires = torch.stack([b_t, progression], dim=1)

        globales = torch.cat([x.mean(dim=[2, 3]), scalaires], dim=1)
        v = self.tete_valeur(globales)                                    # (B, 1)
        a_stop = self.tete_stop(globales)                                 # (B, 1)

        cartes = scalaires.view(B, 2, 1, 1).expand(B, 2, H, W)
        a = self.tete_avantage(torch.cat([x, cartes], dim=1))             # (B, configs, H, W)
        a = a.permute(0, 2, 3, 1).reshape(B, H * W, self.n_configs_radio)
        a = a.index_select(1, self.indices_actions).reshape(B, -1)        # cellules du corridor

        avantages = torch.cat([a_stop, a], dim=1)
        return v + avantages - avantages.mean(dim=1, keepdim=True)


def _test_ordre():
    """L'ordre des cellules doit être celui de la boucle for ix: for iy:."""
    masque = np.zeros((5, 4), dtype=bool)
    masque[1, 2] = masque[1, 3] = masque[3, 0] = True
    attendu = [ix * 4 + iy for ix in range(5) for iy in range(4) if masque[ix, iy]]
    modele = FerroMobileDuelingDQN(n_configs_radio=10, corridor_mask=masque)
    assert modele.indices_actions.tolist() == attendu
    print(f"_test_ordre : OK {attendu}")


def _test_forme():
    masque = np.random.default_rng(0).random((20, 15)) < 0.6
    modele = FerroMobileDuelingDQN(n_configs_radio=10, corridor_mask=masque)
    q = modele(torch.randn(4, N_CANAUX_ETAT, 20, 15), torch.rand(4), torch.rand(4))
    assert q.shape == (4, modele.taille_sortie()) and torch.isfinite(q).all()
    print(f"_test_forme : OK {tuple(q.shape)}")


def _test_dueling():
    """La moyenne des avantages est nulle : la moyenne des Q vaut V(s)."""
    masque = np.ones((10, 10), dtype=bool)
    modele = FerroMobileDuelingDQN(n_configs_radio=3, corridor_mask=masque)
    g, b, p = torch.randn(2, N_CANAUX_ETAT, 10, 10), torch.rand(2), torch.rand(2)
    q = modele(g, b, p)
    x = modele.tronc(g)
    v = modele.tete_valeur(torch.cat([x.mean(dim=[2, 3]), torch.stack([b, p], dim=1)], dim=1)).squeeze(1)
    assert torch.allclose(q.mean(dim=1), v, atol=1e-5)
    print("_test_dueling : OK")


def _test_champ_recepteur():
    """Architecture seule (modèle non entraîné) : une perturbation à 30 cellules (7,5 km) peut influencer
    l'avantage de la cellule centrale, une perturbation à 40 cellules (10 km) ne le peut pas."""
    torch.manual_seed(0)
    masque = np.ones((101, 101), dtype=bool)
    modele = FerroMobileDuelingDQN(n_configs_radio=1, corridor_mask=masque)
    b, p = torch.zeros(1), torch.zeros(1)

    def avantage_centre(grille):
        x = modele.tronc(grille)
        cartes = torch.zeros(1, 2, 101, 101)
        return modele.tete_avantage(torch.cat([x, cartes], dim=1))[0, 0, 50, 50]

    base = torch.rand(1, N_CANAUX_ETAT, 101, 101)
    with torch.no_grad():
        a0 = avantage_centre(base)
        proche, loin = base.clone(), base.clone()
        proche[0, :, 80, 50] += 5.0     # 30 cellules
        loin[0, :, 90, 50] += 5.0       # 40 cellules
        assert abs(avantage_centre(proche) - a0) > 1e-6, "30 cellules : devrait être vu"
        assert abs(avantage_centre(loin) - a0) < 1e-6, "40 cellules : hors du champ"
    print("_test_champ_recepteur : OK (portée de l'architecture : 7,5 km oui, 10 km non)")


if __name__ == "__main__":
    _test_ordre()
    _test_forme()
    _test_dueling()
    _test_champ_recepteur()
    print("Tous les tests passent.")