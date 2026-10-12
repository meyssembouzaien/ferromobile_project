"""
replay_buffer.py — buffer d'expérience, CPU RAM, float16.

État et état suivant stockés séparément (simple, pas de partage
d'index entre transitions). Le masque de faisabilité de l'état
suivant est stocké en bits compressés (8x moins de RAM).

Canaux fixes (n_canaux_fixes) : les 25 premiers canaux de l'état (9 statiques et
16 d'infrastructure) ne changent jamais pendant un run. Ils sont stockés une seule
fois, et seuls les 3 canaux dynamiques le sont par transition : environ 0,7 Go au
lieu de 6 Go pour 2000 transitions. sample() rend exactement les mêmes tenseurs
(même conversion float16, même tirage aléatoire). push() vérifie que les canaux
fixes ne changent pas.

sauver() / charger() : fichier .npz compressé (les canaux dynamiques valent
surtout -1 ou 0 hors de la voie), pour le checkpoint et la sauvegarde Kaggle.

Lancer : python replay_buffer.py (test local, pas de dataset réel).
"""

import os

import numpy as np
import torch


class ReplayBuffer:
    """Buffer circulaire en RAM (jamais sur GPU)."""

    def __init__(self, capacity, nx, ny, n_canaux_etat, n_actions, n_canaux_fixes=0):
        assert capacity > 0
        assert 0 <= n_canaux_fixes < n_canaux_etat
        assert nx > 0 and ny > 0
        assert n_canaux_etat > 0
        assert n_actions > 0

        self.capacity = capacity
        self.n_actions = n_actions
        self.ptr = 0
        self.taille = 0
        self.n_canaux_fixes = n_canaux_fixes
        self.fixes = None   # (n_canaux_fixes, nx, ny), float16, fixé au premier push

        n_stockes = n_canaux_etat - n_canaux_fixes
        self.grilles = torch.zeros(capacity, n_stockes, nx, ny, dtype=torch.float16)
        self.next_grilles = torch.zeros(capacity, n_stockes, nx, ny, dtype=torch.float16)
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
        grille = torch.as_tensor(grille, dtype=torch.float16)
        next_grille = torch.as_tensor(next_grille, dtype=torch.float16)
        k = self.n_canaux_fixes
        if k:
            if self.fixes is None:
                self.fixes = grille[:k].clone()
            if not (torch.equal(grille[:k], self.fixes) and torch.equal(next_grille[:k], self.fixes)):
                raise ValueError("ReplayBuffer : un canal déclaré fixe a changé")
        self.grilles[i] = grille[k:]
        self.next_grilles[i] = next_grille[k:]
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
            "grilles": self._grilles_completes(self.grilles[idx]),
            "b_t": self.b_t[idx],
            "progression": self.progression[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_grilles": self._grilles_completes(self.next_grilles[idx]),
            "next_b_t": self.next_b_t[idx],
            "next_progression": self.next_progression[idx],
            "dones": self.dones[idx],
            "next_masks": torch.from_numpy(next_masks).bool(),
        }

    def _grilles_completes(self, dynamiques):
        """Remet les canaux fixes devant les canaux stockés, en float32."""
        if not self.n_canaux_fixes:
            return dynamiques.float()
        fixes = self.fixes.unsqueeze(0).expand(dynamiques.shape[0], -1, -1, -1)
        return torch.cat([fixes, dynamiques], dim=1).float()

    def __len__(self):
        return self.taille

    _CHAMPS = ["grilles", "next_grilles", "b_t", "next_b_t", "progression", "next_progression",
               "actions", "rewards", "dones", "next_masks_bits"]

    def state_dict(self):
        """Contenu du buffer, pour reprendre un entraînement sans repartir d'un buffer vide."""
        etat = {c: getattr(self, c)[:self.taille].clone() for c in self._CHAMPS}
        etat.update(ptr=self.ptr, taille=self.taille, capacity=self.capacity, n_actions=self.n_actions,
                    n_canaux_fixes=self.n_canaux_fixes, fixes=self.fixes)
        return etat

    def load_state_dict(self, etat):
        assert etat["capacity"] == self.capacity and etat["n_actions"] == self.n_actions, \
            "buffer sauvegardé avec une autre capacité ou un autre nombre d'actions"
        assert int(etat.get("n_canaux_fixes", 0)) == self.n_canaux_fixes, \
            "buffer sauvegardé avec un autre nombre de canaux fixes"
        n = int(etat["taille"])
        for c in self._CHAMPS:
            getattr(self, c)[:n] = torch.as_tensor(etat[c])
        self.ptr, self.taille = int(etat["ptr"]), n
        fixes = etat.get("fixes")
        self.fixes = None if fixes is None else torch.as_tensor(fixes).clone()

    def sauver(self, chemin):
        """Écrit le buffer dans un .npz compressé (écriture atomique)."""
        etat = self.state_dict()
        tableaux = {c: etat[c].numpy() for c in self._CHAMPS}
        if etat["fixes"] is not None:
            tableaux["fixes"] = etat["fixes"].numpy()
        for c in ("ptr", "taille", "capacity", "n_actions", "n_canaux_fixes"):
            tableaux[c] = np.array(etat[c])
        temporaire = chemin + ".tmp.npz"
        np.savez_compressed(temporaire, **tableaux)
        os.replace(temporaire, chemin)

    def charger(self, chemin):
        with np.load(chemin) as f:
            etat = {c: f[c] for c in f.files}
        for c in ("ptr", "taille", "capacity", "n_actions", "n_canaux_fixes"):
            etat[c] = int(etat[c])
        self.load_state_dict(etat)


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


def _test_sauvegarde_restauration():
    buf = ReplayBuffer(capacity=6, nx=3, ny=3, n_canaux_etat=4, n_actions=10)
    for i in range(4):
        grille = np.full((4, 3, 3), i, dtype=np.float32)
        buf.push(grille, 0.5, 0.5, action=i, reward=float(i), next_grille=grille,
                  next_b_t=0.5, next_progression=0.5, done=False, next_mask=np.ones(10, dtype=bool))
    copie = ReplayBuffer(capacity=6, nx=3, ny=3, n_canaux_etat=4, n_actions=10)
    copie.load_state_dict(buf.state_dict())
    assert len(copie) == 4 and copie.ptr == buf.ptr
    assert torch.equal(copie.grilles, buf.grilles) and torch.equal(copie.rewards, buf.rewards)
    print("_test_sauvegarde_restauration : OK")


def _test_canaux_fixes():
    """Même échantillon avec et sans stockage séparé des canaux fixes, et après sauver/charger."""
    import tempfile
    rng = np.random.default_rng(0)
    fixes = rng.standard_normal((5, 4, 3)).astype(np.float32)
    complet = ReplayBuffer(capacity=10, nx=4, ny=3, n_canaux_etat=8, n_actions=30)
    compact = ReplayBuffer(capacity=10, nx=4, ny=3, n_canaux_etat=8, n_actions=30, n_canaux_fixes=5)
    for i in range(14):
        g = np.concatenate([fixes, rng.standard_normal((3, 4, 3)).astype(np.float32)])
        ng = np.concatenate([fixes, rng.standard_normal((3, 4, 3)).astype(np.float32)])
        m = rng.random(30) < 0.5
        for b in (complet, compact):
            b.push(g, 0.3, 0.1, i % 30, float(i), ng, 0.2, 0.2, i == 13, m)
    with tempfile.TemporaryDirectory() as d:
        compact.sauver(os.path.join(d, "buffer.npz"))
        relu = ReplayBuffer(capacity=10, nx=4, ny=3, n_canaux_etat=8, n_actions=30, n_canaux_fixes=5)
        relu.charger(os.path.join(d, "buffer.npz"))
    for b in (compact, relu):
        np.random.seed(7)
        x = b.sample(6)
        np.random.seed(7)
        y = complet.sample(6)
        for k in x:
            assert torch.equal(x[k], y[k]), k
    try:
        g[0, 0, 0] += 1.0
        compact.push(g, 0.3, 0.1, 0, 0.0, ng, 0.2, 0.2, False, m)
        raise AssertionError("un canal fixe modifié aurait dû être refusé")
    except ValueError:
        pass
    print("_test_canaux_fixes : OK")


if __name__ == "__main__":
    _test_canaux_fixes()
    _test_sauvegarde_restauration()
    _test_push_et_sample()
    _test_depassement_capacite()
    print("\nTous les tests passent.")
