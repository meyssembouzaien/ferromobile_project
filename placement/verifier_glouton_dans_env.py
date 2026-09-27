"""
verifier_glouton_dans_env.py
Rejoue les sites du greedy dans FerroMobileEnv, deux fois dans le même processus.
Mesure : QoS finale, branche de r_fin, stabilité de n_white face au shadowing aléatoire.
"""

from train_dqn import preparer_environnement_colab
preparer_environnement_colab()  # avant l'import qui charge crc_covlib

from environment_rl import FerroMobileEnv

# les 26 cellules du greedy, dans l'ordre
SITES = [(40,119),(47,110),(42,114),(42,117),(100,63),(64,105),(41,126),(59,110),(90,85),
         (52,109),(43,112),(67,101),(101,74),(45,111),(41,115),(84,85),(100,64),(40,121),
         (44,112),(48,109),(44,111),(66,102),(41,123),(42,127),(41,114),(59,105)]

# n_white après chaque site, relevé dans le log du greedy
N_WHITE_GLOUTON = [265,221,187,156,127,107,87,68,57,50,43,37,32,
                   27,23,19,16,14,13,12,8,5,3,2,1,0]


def rejouer(env):
    """Rejoue les sites, puis STOP si l'épisode n'est pas fini."""
    env.reset()
    n_whites = []
    done = False
    for ix, iy in SITES:
        etat, r, done, info = env.step((ix, iy, "3G", 900))
        n_whites.append(info["n_white"])
        if done:
            break
    if not done:
        etat, r, done, info = env.step("STOP")
    dp = env._dernier_resultat_dp
    return n_whites, {"qos_mean": dp["qos_mean"], "qos_min": dp["qos_min"],
                      "r_fin": info["r_fin"], "cout": info["cout_total"],
                      "n_white": info["n_white"]}


env = FerroMobileEnv(scenario="S1", budget=10_000_000.0, t_max=150,
                     sauvegarder_deploiements_rl=False)

# Deux replays dans le MÊME processus : la RNG du shadowing avance entre les deux.
# (Deux lancements séparés du script donneraient le même résultat, seed fixe 123.)
nw1, res1 = rejouer(env)
nw2, res2 = rejouer(env)

print("site | greedy | replay 1 | replay 2")
for k in range(len(SITES)):
    a = nw1[k] if k < len(nw1) else "-"
    b = nw2[k] if k < len(nw2) else "-"
    print(f"{k+1:4d} | {N_WHITE_GLOUTON[k]:6d} | {a!s:>8} | {b!s:>8}")

for nom, res in [("replay 1", res1), ("replay 2", res2)]:
    print(f"\n{nom} : n_white final={res['n_white']}, qos_mean={res['qos_mean']:.3f}, "
          f"qos_min={res['qos_min']:.3f}, r_fin={res['r_fin']:.4f}, cout={res['cout']:.0f}€")

print("\nLecture de r_fin : entre 0.2 et 0.4 = succès, -0.5 = QoS insuffisante, < -0.5 = zones blanches restantes")