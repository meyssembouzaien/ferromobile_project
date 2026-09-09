"""
test_trajectory_dp.py — Tests synthétiques de trajectory_dp

Six vérifications, comme demandé avant d'intégrer la DP au RL :
1. la DP choisit bien le maximum
2. N_H, N_O, N_T sont correctement comptés
3. plusieurs changements sur une même transition se cumulent
4. le premier point ne génère jamais de pénalité
5. une cellule hypothétique (operateur=None) ne génère pas de pénalité opérateur
6. le résultat est cohérent si on donne à la DP un tronçon isolé

Chaque cas a un résultat attendu calculé à la main dans le commentaire.
"""

from trajectory_dp import trajectory_dp


def test_1_maximum_choisi():
    """La DP doit choisir de rester sur la meilleure cellule plutôt que
    de switcher pour un petit gain, si le changement coûte plus cher."""
    points = [0, 1]
    cells_per_point = [
        [{"cellule": ("s1", "Orange", "4G", "700"), "qos": 0.5},
         {"cellule": ("s2", "SFR", "4G", "700"), "qos": 0.9}],
        [{"cellule": ("s1", "Orange", "4G", "700"), "qos": 0.5},
         {"cellule": ("s2", "SFR", "4G", "700"), "qos": 0.9}],
    ]
    # Rester sur s2 les deux points : J = 0.9 + 0.9 = 1.8, pas de pénalité.
    # Switcher s1->s2 : J = 0.5 + 0.9 - 0.1 = 1.3. Moins bon, donc rejeté.
    resultat = trajectory_dp(points, cells_per_point, lambda_H=0.1, lambda_O=0.2, lambda_T=0.3)
    assert resultat["trajectory"] == [("s2", "SFR", "4G", "700"), ("s2", "SFR", "4G", "700")]
    assert abs(resultat["J_star"] - 1.8) < 1e-9
    assert resultat["N_H"] == 0
    print("test_1_maximum_choisi : OK")


def test_2_cumul_des_penalites():
    """Un seul candidat par point : la DP est forcée de changer de site,
    d'opérateur ET de génération en même temps. Les trois pénalités
    doivent se cumuler sur cette unique transition."""
    points = [0, 1]
    cells_per_point = [
        [{"cellule": ("s1", "Orange", "4G", "700"), "qos": 0.5}],
        [{"cellule": ("s2", "SFR", "5G", "2600"), "qos": 0.9}],
    ]
    lambda_H, lambda_O, lambda_T = 0.1, 0.2, 0.3
    resultat = trajectory_dp(points, cells_per_point, lambda_H, lambda_O, lambda_T)
    # J = 0.5 + 0.9 - (0.1 + 0.2 + 0.3) = 0.8
    assert abs(resultat["J_star"] - 0.8) < 1e-9
    assert resultat["N_H"] == 1
    assert resultat["N_O"] == 1
    assert resultat["N_T"] == 1
    print("test_2_cumul_des_penalites : OK")


def test_3_premier_point_sans_penalite():
    """Un seul point : J_star doit être exactement la QoS de ce point,
    sans aucune pénalité de transition (il n'y a pas de point avant)."""
    points = [0]
    cells_per_point = [
        [{"cellule": ("s1", "Orange", "4G", "700"), "qos": 0.42}],
    ]
    resultat = trajectory_dp(points, cells_per_point, lambda_H=100, lambda_O=100, lambda_T=100)
    assert abs(resultat["J_star"] - 0.42) < 1e-9
    print("test_3_premier_point_sans_penalite : OK")


def test_4_operateur_hypothetique_non_penalise():
    """Une cellule hypothétique (operateur=None) suivie d'une cellule
    réelle : pas de pénalité d'opérateur, même si "None" et "Orange"
    sont des valeurs différentes."""
    points = [0, 1]
    cells_per_point = [
        [{"cellule": ("s1", None, "4G", "700"), "qos": 0.6}],
        [{"cellule": ("s1", "Orange", "4G", "700"), "qos": 0.7}],
    ]
    lambda_H, lambda_O, lambda_T = 0.1, 0.2, 0.3
    resultat = trajectory_dp(points, cells_per_point, lambda_H, lambda_O, lambda_T)
    # Cellule différente (opérateur diffère dans le tuple) -> N_H = 1
    # Même génération -> N_T = 0
    # Un des deux opérateurs est None -> pas de pénalité -> N_O = 0
    assert resultat["N_H"] == 1
    assert resultat["N_T"] == 0
    assert resultat["N_O"] == 0
    # J = 0.6 + 0.7 - lambda_H = 1.2
    assert abs(resultat["J_star"] - 1.2) < 1e-9
    print("test_4_operateur_hypothetique_non_penalise : OK")


def test_5_tronçon_isole():
    """La même fonction, appelée sur un sous-ensemble de points, doit
    fonctionner sans erreur et traiter le premier point du sous-ensemble
    comme un vrai départ (sans pénalité), même s'il ne l'était pas dans
    le trajet complet. C'est le comportement attendu, pas un bug."""
    points_complet = [0, 1, 2]
    cells_complet = [
        [{"cellule": ("s1", "Orange", "4G", "700"), "qos": 0.5}],
        [{"cellule": ("s2", "SFR", "5G", "2600"), "qos": 0.9}],
        [{"cellule": ("s2", "SFR", "5G", "2600"), "qos": 0.8}],
    ]
    lambda_H, lambda_O, lambda_T = 0.1, 0.2, 0.3
    resultat_complet = trajectory_dp(points_complet, cells_complet, lambda_H, lambda_O, lambda_T)

    # Même DP, mais seulement sur les points 1 et 2 (le tronçon isolé).
    points_troncon = [1, 2]
    cells_troncon = cells_complet[1:]
    resultat_troncon = trajectory_dp(points_troncon, cells_troncon, lambda_H, lambda_O, lambda_T)

    # Dans le trajet complet : J = 0.5+0.9-0.6 (3 pénalités) + 0.8 (même cellule, 0 pénalité)
    assert abs(resultat_complet["J_star"] - (0.5 + 0.9 - 0.6 + 0.8)) < 1e-9
    # Dans le tronçon isolé : le premier point (s2/SFR/5G) ne coûte plus
    # rien, donc J = 0.9 + 0.8, sans la pénalité de la transition
    # site1->site2 qui existait dans le trajet complet.
    assert abs(resultat_troncon["J_star"] - (0.9 + 0.8)) < 1e-9
    assert resultat_troncon["trajectory"] == resultat_complet["trajectory"][1:]
    print("test_5_tronçon_isole : OK")


def test_6_erreurs_bien_signalees():
    """Un trajet vide, ou un point sans aucune cellule, doit lever une
    erreur claire plutôt que planter silencieusement ou renvoyer n'importe quoi."""
    try:
        trajectory_dp([], [], 0.1, 0.2, 0.3)
        assert False, "aurait dû lever une erreur"
    except ValueError:
        pass

    try:
        trajectory_dp([0], [[]], 0.1, 0.2, 0.3)
        assert False, "aurait dû lever une erreur"
    except ValueError:
        pass
    print("test_6_erreurs_bien_signalees : OK")


if __name__ == "__main__":
    test_1_maximum_choisi()
    test_2_cumul_des_penalites()
    test_3_premier_point_sans_penalite()
    test_4_operateur_hypothetique_non_penalise()
    test_5_tronçon_isole()
    test_6_erreurs_bien_signalees()
    print("\nTous les tests passent.")
