"""
dqn_model.py : DQN vanilla pour FerroMobile.

Entrée : grille (n_canaux, nx, ny) + b_t + progression.
Sortie : Q-values dans le même ordre que env.actions (STOP en position 0).

Lancer : python dqn_model.py (tests locaux, sans dataset)
"""

import numpy as np
import torch
import torch.nn as nn


class FerroMobileDQN(nn.Module):
    """Tronc convolutif, une tête spatiale (Q par cellule x config radio) et une tête STOP."""

    def __init__(self, n_canaux, n_configs_radio, masque_actions, canaux_conv=(64, 128, 128)):
        super().__init__()
        c1, c2, c3 = canaux_conv
        self.n_configs_radio = n_configs_radio

        self.conv1 = nn.Conv2d(n_canaux, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # +2 : cartes b_t et progression
        self.tete_spatiale = nn.Conv2d(c3 + 2, n_configs_radio, kernel_size=1)
        self.tete_stop = nn.Sequential(nn.Linear(c3 + 2, 64), nn.ReLU(), nn.Linear(64, 1))

        # Cellules où une action existe, ordre C (ix puis iy) = ordre de env.actions
        assert masque_actions.dtype == bool
        self.nx, self.ny = masque_actions.shape
        indices = np.nonzero(masque_actions.reshape(-1))[0]
        self.register_buffer("indices_actions", torch.as_tensor(indices, dtype=torch.long))

    def taille_sortie(self):
        """|A| = 1 (STOP) + cellules autorisées x configs radio."""
        return 1 + len(self.indices_actions) * self.n_configs_radio

    def forward(self, grille, b_t, progression):
        """grille : (B, n_canaux, nx, ny) ; b_t, progression : (B,). Retourne (B, |A|)."""
        x = self.relu(self.conv1(grille))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        B, _, H, W = x.shape

        # Tête STOP : moyenne globale + b_t + progression
        globales = torch.cat([x.mean(dim=[2, 3]), torch.stack([b_t, progression], dim=1)], dim=1)
        q_stop = self.tete_stop(globales)

        # Tête spatiale : b_t et progression diffusés en cartes constantes
        cartes = torch.stack([b_t, progression], dim=1).view(B, 2, 1, 1).expand(B, 2, H, W)
        q = self.tete_spatiale(torch.cat([x, cartes], dim=1))           # (B, configs, H, W)
        q = q.permute(0, 2, 3, 1).reshape(B, H * W, self.n_configs_radio)
        q = q.index_select(1, self.indices_actions).reshape(B, -1)       # cellules autorisées seulement

        return torch.cat([q_stop, q], dim=1)


def _test_ordre():
    """L'ordre des cellules doit être celui de la boucle for ix: for iy:."""
    masque = np.zeros((5, 4), dtype=bool)
    masque[1, 2] = masque[1, 3] = masque[3, 0] = True
    attendu = [ix * 4 + iy for ix in range(5) for iy in range(4) if masque[ix, iy]]
    modele = FerroMobileDQN(n_canaux=31, n_configs_radio=10, masque_actions=masque)
    assert modele.indices_actions.tolist() == attendu
    print(f"_test_ordre : OK {attendu}")


def _test_forme():
    masque = np.random.default_rng(0).random((20, 15)) < 0.3
    modele = FerroMobileDQN(n_canaux=31, n_configs_radio=10, masque_actions=masque)
    q = modele(torch.randn(4, 31, 20, 15), torch.rand(4), torch.rand(4))
    assert q.shape == (4, modele.taille_sortie()) and torch.isfinite(q).all()
    print(f"_test_forme : OK {tuple(q.shape)}")


if __name__ == "__main__":
    _test_ordre()
    _test_forme()
    print("Tous les tests passent.")