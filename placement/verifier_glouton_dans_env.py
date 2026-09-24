"""
verifier_glouton_dans_env.py
Rejoue les 29 sites du greedy dans FerroMobileEnv.
Vérifie : QoS finale, r_fin, somme r_t = U_final - U0, effet de la RNG.
"""

from train_dqn import preparer_environnement_colab
preparer_environnement_colab()  # avant l'import qui charge crc_covlib

from environment_rl import FerroMobileEnv

# les 29 cellules du greedy, dans l'ordre
SITES = [(40,119),(47,110),(42,114),(42,117),(100,63),(64,105),(59,110),
         (41,126),(90,85),(45,111),(52,109),(67,101),(43,113),(101,74),
         (100,64),(84,85),(44,112),(41,125),(41,115),(62,105),(48,109),
         (40,121),(44,111),(66,102),(40,114),(40,117),(41,116),(88,84),(42,125)]

env = FerroMobileEnv(scenario="S1", budget=10_000_000.0, t_max=150,
                     sauvegarder_deploiements_rl=False)
env.reset()
U0 = env.U_courant
somme_rt = 0.0
done = False

for k, (ix, iy) in enumerate(SITES, start=1):
    etat, r, done, info = env.step((ix, iy, "3G", 900))
    r_t = r - info.get("r_fin", 0.0)   # on retire r_fin du dernier step
    somme_rt += r_t
    print(f"site {k} : n_white={info['n_white']}, r_t={r_t:.4f}")
    if done:
        break

# si n_white n'est pas tombé à 0, l'épisode n'est pas fini : STOP pour avoir r_fin
if not done:
    print("\nn_white > 0 après les 29 sites : STOP manuel")
    etat, r, done, info = env.step("STOP")

dp = env._dernier_resultat_dp
print(f"\nqos_mean={dp['qos_mean']:.3f}, qos_min={dp['qos_min']:.3f}")
print(f"U0={U0:.4f}, U_final={env.U_courant:.4f}")
print(f"somme r_t={somme_rt:.4f}, U_final-U0={env.U_courant - U0:.4f}")
print(f"r_fin={info.get('r_fin')}, cout={info.get('cout_total')}, n_white final={info.get('n_white')}")