"""
cv_blocks.py : validation croisée spatiale par blocs (Expérience C, niveau 1, tableau 3.16).

    - 4 blocs contigus de la voie (environ 9,6 km), jamais des points isolés ;
    - pli k : entraînement sur les trois autres blocs, test sur le bloc k ;
    - budget de chaque bloc proportionnel à ses zones blanches (part de B) ;
    - pendant l'entraînement, les déploiements dont la couverture pourrait atteindre un
      point du bloc de test sont masqués. Critère retenu : distance du centre de la
      cellule au point de test le plus proche <= portée nominale de (g, f). C'est un
      sur-ensemble de la vraie couverture (la ligne de vue n'est pas testée), donc
      aucune action d'entraînement ne sert le bloc de test ;
    - statistiques de normalisation des canaux calculées sur les blocs d'entraînement.

Lancer : python cv_blocks.py (affiche les 4 plis : points, zones blanches, budgets, actions masquées)
"""

import os

import numpy as np
from scipy.spatial import cKDTree

from grid_config import VERS_METRES, RESOLUTION_M

N_BLOCS = 4
D_COV = 1.0


def blocs_de_points(df_points, n_blocs=N_BLOCS):
    """Listes de point_id (tunnels compris), en blocs contigus dans l'ordre du trajet."""
    ids = df_points.sort_values("point_id")["point_id"].values
    return [list(b) for b in np.array_split(ids, n_blocs)]


def preparer_pli(env_base, pli, budget_total, n_blocs=N_BLOCS):
    """Configuration du pli `pli` (1 à n_blocs). env_base : FerroMobileEnv sur toute la ligne,
    après reset(). Renvoie un dict utilisable comme arguments de FerroMobileEnv."""
    from prepare_data import get_profile

    assert 1 <= pli <= n_blocs
    blocs = blocs_de_points(env_base.df_points, n_blocs)
    test = set(blocs[pli - 1])
    train = set().union(*(set(b) for i, b in enumerate(blocs) if i != pli - 1))

    debit = env_base.debit_reel_par_point.reindex(list(env_base.points_hors_tunnel)).fillna(0)
    blancs = set(debit[debit < D_COV].index)
    n_blancs_test, n_blancs_train = len(blancs & test), len(blancs & train)
    budget_test = budget_total * n_blancs_test / len(blancs)
    budget_train = budget_total * n_blancs_train / len(blancs)

    # Masque d'entraînement : aucune action ne doit pouvoir atteindre un point de test
    pts = env_base.df_points
    pts_test = pts[pts["point_id"].isin(test) & ~pts["in_tunnel"]]
    x, y = VERS_METRES.transform(pts_test["lon"].values, pts_test["lat"].values)
    arbre = cKDTree(np.column_stack([x, y]))
    actions = env_base.actions[1:]
    ix = np.array([a[0] for a in actions])
    iy = np.array([a[1] for a in actions])
    cx = env_base.grille.x_min + (ix + 0.5) * RESOLUTION_M
    cy = env_base.grille.y_min + (iy + 0.5) * RESOLUTION_M
    distance, _ = arbre.query(np.column_stack([cx, cy]))
    portee_m = np.array([get_profile(a[2], a[3])[0] * 1000.0 for a in actions])
    autorisees = np.ones(len(env_base.actions), dtype=bool)
    autorisees[1:] = distance > portee_m

    return {
        "pli": pli, "points_train": sorted(train), "points_test": sorted(test),
        "n_blancs_train": n_blancs_train, "n_blancs_test": n_blancs_test,
        "budget_train": budget_train, "budget_test": budget_test,
        "actions_autorisees_train": autorisees,
        "part_masquee": 1.0 - autorisees[1:].mean(),
    }


def env_entrainement(pli_config, scenario="S1", t_max=150, poids=None, **kwargs):
    from environment_rl import FerroMobileEnv
    c = pli_config
    return FerroMobileEnv(scenario=scenario, budget=c["budget_train"], budget_reference=c["budget_train"],
                          t_max=t_max, poids=poids, sauvegarder_deploiements_rl=False,
                          points_evalues=c["points_train"], points_stats=c["points_train"],
                          actions_autorisees=c["actions_autorisees_train"], **kwargs)


def env_test(pli_config, scenario="S1", t_max=150, poids=None, **kwargs):
    """Bloc de test : budget de ce bloc, b_t normalisé par le budget d'entraînement du pli,
    statistiques de normalisation des blocs d'entraînement. Toutes les actions sont permises."""
    from environment_rl import FerroMobileEnv
    c = pli_config
    return FerroMobileEnv(scenario=scenario, budget=c["budget_test"], budget_reference=c["budget_train"],
                          t_max=t_max, poids=poids, sauvegarder_deploiements_rl=False,
                          points_evalues=c["points_test"], points_stats=c["points_train"], **kwargs)


if __name__ == "__main__":
    if "CRC_COVLIB_PATH" not in os.environ:
        os.environ["CRC_COVLIB_PATH"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "crc-covlib", "python-wrapper")
    from environment_rl import FerroMobileEnv
    env = FerroMobileEnv(scenario="S1", budget=10e6, t_max=150, sauvegarder_deploiements_rl=False)
    env.reset()
    for k in range(1, N_BLOCS + 1):
        c = preparer_pli(env, k, 10e6)
        print(f"pli {k} : test {len(c['points_test'])} points, {c['n_blancs_test']} blancs, "
              f"budget test {c['budget_test'] / 1e6:.2f} M€ | entraînement {c['n_blancs_train']} blancs, "
              f"budget {c['budget_train'] / 1e6:.2f} M€ | actions masquées {100 * c['part_masquee']:.0f} %")
