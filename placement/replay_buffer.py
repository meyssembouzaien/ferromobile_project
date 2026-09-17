"""
replay_buffer.py — buffer d'expérience, CPU RAM, float16.

État et état suivant stockés séparément (simple, pas de partage
d'index entre transitions). Le masque de faisabilité de l'état
suivant est stocké en bits compressés (8x moins de RAM).

Lancer : python replay_buffer.py (test local, pas de dataset réel).
"""

import numpy as np
import torch


class ReplayBuffer:
    """Buffer circulaire en RAM (jamais sur GPU)."""

    def __init__(self, capacity, nx, ny, n_canaux_etat, n_actions):
        assert capacity > 0
        assert nx > 0 and ny > 0
        assert n_canaux_etat > 0
        assert n_actions > 0

        self.capacity = capacity
        self.n_actions = n_actions
        self.ptr = 0
        self.taille = 0

        self.grilles = torch.zeros(capacity, n_canaux_etat, nx, ny, dtype=torch.float16)
        self.next_grilles = torch.zeros(capacity, n_canaux_etat, nx, ny, dtype=torch.float16)
        self.b_t = torch.zeros(capacity, dtype=torch.float32)
        self.next_b_t = torch.zeros(capacity, dtype=torch.float32)
        self.progression = torch.zeros(capacity, dtype=torch.float32)
        self.next_progression = torch.zeros(capacity, dtype=torch.float32)

        self.actions = torch.zeros(capacity, dtype=torch.int64)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        # Masque compressé en bits, pas un bool par action.
        n_octets_masque = (n_actions + 7) // 8
        self.next_masks_bits = torch.zeros(capacity, n_octets_masque, dtype=torch.uint8)

        self._afficher_estimation_memoire()

    def _afficher_estimation_memoire(self):
        octets = (
            self.grilles.element_size() * self.grilles.nelement()
            + self.next_grilles.element_size() * self.next_grilles.nelement()
            + self.next_masks_bits.element_size() * self.next_masks_bits.nelement()
        )
        print(f"ReplayBuffer : capacité {self.capacity}, ~{octets / 1e9:.2f} Go RAM")

    def push(self, grille, b_t, progression, action, reward,
              next_grille, next_b_t, next_progression, done, next_mask):
        """next_mask : array booléen numpy, taille n_actions."""
        next_mask = np.asarray(next_mask, dtype=bool)
        assert next_mask.shape == (self.n_actions,), (
            f"next_mask attendu {(self.n_actions,)}, reçu {next_mask.shape}"
        )

        i = self.ptr
        self.grilles[i] = torch.as_tensor(grille, dtype=torch.float16)
        self.next_grilles[i] = torch.as_tensor(next_grille, dtype=torch.float16)
        self.b_t[i] = b_t
        self.next_b_t[i] = next_b_t
        self.progression[i] = progression
        self.next_progression[i] = next_progression
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.next_masks_bits[i] = torch.from_numpy(np.packbits(next_mask))

        self.ptr = (self.ptr + 1) % self.capacity
        self.taille = min(self.taille + 1, self.capacity)

    def sample(self, batch_size):
        """Batch encore sur CPU. Le passage GPU se fait dans train_dqn.py."""
        assert self.taille >= batch_size, "pas assez de transitions"
        idx = np.random.choice(self.taille, size=batch_size, replace=False)

        next_masks = np.unpackbits(self.next_masks_bits[idx].numpy(), axis=1)[:, :self.n_actions]

        return {
            "grilles": self.grilles[idx].float(),
            "b_t": self.b_t[idx],
            "progression": self.progression[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_grilles": self.next_grilles[idx].float(),
            "next_b_t": self.next_b_t[idx],
            "next_progression": self.next_progression[idx],
            "dones": self.dones[idx],
            "next_masks": torch.from_numpy(next_masks).bool(),
        }

    def __len__(self):
        return self.taille


def _test_push_et_sample():
    nx, ny, n_canaux, n_actions = 5, 4, 27, 50
    buf = ReplayBuffer(capacity=20, nx=nx, ny=ny, n_canaux_etat=n_canaux, n_actions=n_actions)

    for i in range(15):
        grille = np.random.randn(n_canaux, nx, ny).astype(np.float32)
        next_grille = np.random.randn(n_canaux, nx, ny).astype(np.float32)
        mask = np.random.rand(n_actions) < 0.3
        buf.push(grille, 0.8, 0.1, action=i % n_actions, reward=float(i),
                  next_grille=next_grille, next_b_t=0.7, next_progression=0.2,
                  done=(i == 14), next_mask=mask)

    assert len(buf) == 15
    batch = buf.sample(8)
    assert batch["grilles"].shape == (8, n_canaux, nx, ny)
    assert batch["next_masks"].shape == (8, n_actions)
    assert batch["grilles"].dtype == torch.float32
    print("_test_push_et_sample : OK")


def _test_depassement_capacite():
    """Le buffer plein doit écraser les plus vieilles transitions."""
    buf = ReplayBuffer(capacity=5, nx=3, ny=3, n_canaux_etat=4, n_actions=10)
    for i in range(8):
        grille = np.full((4, 3, 3), i, dtype=np.float32)
        mask = np.zeros(10, dtype=bool)
        buf.push(grille, 0.5, 0.5, action=0, reward=0.0, next_grille=grille,
                  next_b_t=0.5, next_progression=0.5, done=False, next_mask=mask)

    assert len(buf) == 5
    # après 8 push sur capacité 5, les 3 premières valeurs (0,1,2) sont écrasées
    valeurs_restantes = sorted(buf.grilles[:, 0, 0, 0].tolist())
    assert valeurs_restantes == [3.0, 4.0, 5.0, 6.0, 7.0]
    print("_test_depassement_capacite : OK")


if __name__ == "__main__":
    _test_push_et_sample()
    _test_depassement_capacite()
    print("\nTous les tests passent.")
