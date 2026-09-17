"""
dqn_model.py — DQN vanilla pour FerroMobile.

Entrée : grille (27, nx, ny) + b_t + progression.
Sortie : Q-values, même ordre que env.actions (STOP en position 0).

Lancer : python dqn_model.py (test local, CPU, pas de dataset réel).
"""

import numpy as np
import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]

# Valeur par défaut UNIQUEMENT pour le test local ci-dessous.
DEFAULT_N_CONFIGS_RADIO_TEST = 11


class FerroMobileDQN(nn.Module):
    """DQN vanilla. Une tête spatiale (Q par cellule x config radio),
    une tête STOP (Q globale). Pas de V(s)/A(s,a) : c'est du Dueling,
    pas ce fichier.

    set_corridor_mask() doit être appelé avant forward().
    """

    N_CANAUX_ETAT = 27  # doit matcher build_state_dynamic.py

    def __init__(self, n_configs_radio, canaux_conv=(64, 128, 128), corridor_mask=None):
        super().__init__()
        c1, c2, c3 = canaux_conv
        self.n_configs_radio = n_configs_radio
        self.c3 = c3

        self.conv1 = nn.Conv2d(self.N_CANAUX_ETAT, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # +2 = cartes b_t et progression, concaténées avant.
        self.tete_spatiale = nn.Conv2d(c3 + 2, n_configs_radio, kernel_size=1)
        self.tete_stop = nn.Sequential(
            nn.Linear(c3 + 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Buffer créé une fois ici, rempli par set_corridor_mask().
        self.register_buffer("valid_indices", torch.empty(0, dtype=torch.long))
        self.n_cellules_valides = None
        if corridor_mask is not None:
            self.set_corridor_mask(corridor_mask)

    def set_corridor_mask(self, corridor_mask: np.ndarray):
        """corridor_mask : (nx, ny) booléen, ex. env.grille.corridor_mask.

        Ordre C (ix dehors, iy dedans) = même ordre que la boucle
        for ix: for iy: de environment_rl.py.
        """
        assert corridor_mask.dtype == bool
        plat = corridor_mask.reshape(-1)
        indices_valides = np.nonzero(plat)[0]
        self.valid_indices = torch.as_tensor(indices_valides, dtype=torch.long)
        self.n_cellules_valides = len(indices_valides)
        self._nx, self._ny = corridor_mask.shape

    def _corridor_mask_defini(self):
        return self.n_cellules_valides is not None

    def taille_sortie_attendue(self):
        """|A| = 1 (STOP) + n_cellules_valides * n_configs_radio."""
        if not self._corridor_mask_defini():
            raise RuntimeError("Appeler set_corridor_mask() d'abord")
        return 1 + self.n_cellules_valides * self.n_configs_radio

    def forward(self, grille, b_t, progression):
        """grille : (B, 27, nx, ny). b_t, progression : (B,).
        Retourne (B, |A|), même ordre que env.actions.
        """
        if not self._corridor_mask_defini():
            raise RuntimeError("Appeler set_corridor_mask() avant forward()")

        x = self.relu(self.conv1(grille))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))  # (B, c3, nx, ny)

        B, C, H, W = x.shape
        assert (H, W) == (self._nx, self._ny), "grille incohérente avec corridor_mask"

        # b_t / progression diffusés en cartes constantes pour la tête spatiale.
        b_t_carte = b_t.view(B, 1, 1, 1).expand(B, 1, H, W)
        progression_carte = progression.view(B, 1, 1, 1).expand(B, 1, H, W)
        x_augmente = torch.cat([x, b_t_carte, progression_carte], dim=1)

        # b_t / progression en scalaires pour la tête STOP.
        pool_global = x.mean(dim=[2, 3])
        scalaires = torch.stack([b_t, progression], dim=1)
        features_globales = torch.cat([pool_global, scalaires], dim=1)
        q_stop = self.tete_stop(features_globales)  # (B, 1)

        q_spatial = self.tete_spatiale(x_augmente)  # (B, n_configs_radio, H, W)

        # (B,C,H,W) -> (B,H,W,C) -> (B,H*W,C) : même ordre que corridor_mask.reshape(-1).
        q_spatial = q_spatial.permute(0, 2, 3, 1).reshape(B, H * W, self.n_configs_radio)
        q_spatial_valide = q_spatial.index_select(1, self.valid_indices)
        q_spatial_plat = q_spatial_valide.reshape(B, -1)

        return torch.cat([q_stop, q_spatial_plat], dim=1)  # (B, |A|)


def _test_forme_synthetique():
    """Test de forme, données synthétiques, CPU, pas de quota Colab."""
    nx, ny = 20, 15
    rng = np.random.default_rng(0)
    corridor_mask = rng.random((nx, ny)) < 0.6
    n_valides = int(corridor_mask.sum())
    n_configs = DEFAULT_N_CONFIGS_RADIO_TEST

    modele = FerroMobileDQN(n_configs_radio=n_configs, corridor_mask=corridor_mask)
    taille_attendue = 1 + n_valides * n_configs
    assert modele.taille_sortie_attendue() == taille_attendue

    B = 4
    grille = torch.randn(B, FerroMobileDQN.N_CANAUX_ETAT, nx, ny)
    b_t = torch.rand(B)
    progression = torch.rand(B)

    q = modele(grille, b_t, progression)
    assert q.shape == (B, taille_attendue)
    assert torch.isfinite(q).all()

    print(f"_test_forme_synthetique : OK ({n_valides} cellules, |A|={taille_attendue}, sortie {q.shape})")


def _test_coherence_ordre_masque():
    """Vérifie que l'ordre de valid_indices == ordre de la boucle for ix: for iy:."""
    nx, ny = 5, 4
    corridor_mask = np.zeros((nx, ny), dtype=bool)
    corridor_mask[1, 2] = True
    corridor_mask[1, 3] = True
    corridor_mask[3, 0] = True

    ordre_attendu = []
    for ix in range(nx):
        for iy in range(ny):
            if corridor_mask[ix, iy]:
                ordre_attendu.append(ix * ny + iy)

    modele = FerroMobileDQN(n_configs_radio=DEFAULT_N_CONFIGS_RADIO_TEST, corridor_mask=corridor_mask)
    ordre_obtenu = modele.valid_indices.tolist()

    assert ordre_obtenu == ordre_attendu
    print(f"_test_coherence_ordre_masque : OK ({ordre_obtenu})")


if __name__ == "__main__":
    _test_coherence_ordre_masque()
    _test_forme_synthetique()
    print("\nTous les tests passent.")
