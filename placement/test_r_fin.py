"""
test_r_fin.py — Validation de U(I) et r_fin sur 5 scénarios jouets (§26 point 3)

MODIFIÉ (2e correction reward) : _evaluer() passe maintenant n_total à
compute_r_fin() (déjà calculé localement : n_total = n_eval = len(points)).
Une seule valeur numérique attendue change : dans test_4, nw_E=1 sur
n_total=2 donne désormais Ŵ=0.5, donc r_fin_E = -50-50*0.5 = -75 (au
lieu de -100 fixe avant). Les autres scénarios ont tous n_white=0 ou 2
(=n_total), donc inchangés par construction (voir commentaires).
"""

from trajectory_dp import trajectory_dp
from cost_model import get_total_cost
from utility import normaliser_metriques, compute_U, compute_r_fin, WEIGHTS_REFERENCE

D_COV = 1.0
Q_CRITIQUE = 0.5
R_COUVERTURE = 100.0
R_QUALITE = 50.0
R_SUCCES = 20.0
R_EFF = 20.0
BUDGET = 500_000.0
POIDS = WEIGHTS_REFERENCE

R = ("site_R", "ORANGE", "2G", 900)


def _n_white(points, cells_avec_debit, D_cov=D_COV):
    return sum(
        1 for t in range(len(points))
        if max(c["debit"] for c in cells_avec_debit[t]) < D_cov
    )


def _evaluer(points, cells_par_point, candidats_choisis):
    resultat = trajectory_dp(points, cells_par_point, 0.05, 0.05, 0.05)
    nw = _n_white(points, cells_par_point)
    n_total = n_eval = len(points)
    metriques = normaliser_metriques(resultat, n_eval, nw, n_total)
    U = compute_U(metriques, POIDS["w_Q"], POIDS["w_W"], POIDS["w_H"], POIDS["w_O"], POIDS["w_T"])

    sites_deja = {"site_R"}
    cout_total = 0.0
    for site, gen, bande in candidats_choisis:
        cout_total += get_total_cost(site, gen, bande, sites_deja)
        sites_deja.add(site)

    r_fin = compute_r_fin(nw, n_total, resultat["qos_min"], cout_total, BUDGET,
                           Q_CRITIQUE, R_COUVERTURE, R_QUALITE, R_SUCCES, R_EFF)
    return resultat, nw, U, cout_total, r_fin


def test_1_couverture_seule():
    points = [0, 1]
    A = ("site_A", None, "5G", 3500)
    cells_sans = [[{"cellule": R, "qos": 0.3, "debit": 0.5}] for _ in points]
    cells_avec = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                   {"cellule": A, "qos": 0.9, "debit": 50.0}] for _ in points]

    _, nw_sans, _, cout_sans, r_fin_sans = _evaluer(points, cells_sans, [])
    _, nw_avec, _, cout_avec, r_fin_avec = _evaluer(points, cells_avec, [("site_A", "5G", 3500)])

    # nw_sans=2=n_total -> Ŵ=1 -> -R_couverture, INCHANGÉ (cas limite)
    assert nw_sans == 2 and r_fin_sans == -R_COUVERTURE
    assert nw_avec == 0 and cout_avec == 205_000
    assert abs(r_fin_avec - 31.8000) < 1e-3
    assert r_fin_avec > r_fin_sans
    print("test_1_couverture_seule : OK")


def test_2_conflit_couverture_qos():
    points = [0, 1]
    A = ("site_A2", None, "4G", 800)
    B = ("site_B2", None, "5G", 3500)
    cells_A = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": A, "qos": 0.4, "debit": 20.0}] for _ in points]
    cells_B = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": B, "qos": 0.7, "debit": 50.0}] for _ in points]

    _, nw_A, _, cout_A, r_fin_A = _evaluer(points, cells_A, [("site_A2", "4G", 800)])
    _, nw_B, _, cout_B, r_fin_B = _evaluer(points, cells_B, [("site_B2", "5G", 3500)])

    assert nw_A == 0 and r_fin_A == -R_QUALITE
    assert nw_B == 0 and cout_B > cout_A
    assert abs(r_fin_B - 31.8000) < 1e-3
    assert r_fin_B > r_fin_A
    print("test_2_conflit_couverture_qos : OK")


def test_3_deux_solutions_acceptables_couts_differents():
    points = [0, 1]
    C = ("site_R", "ORANGE", "3G", 2100)
    D = ("site_D3", None, "5G", 3500)
    cells_C = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": C, "qos": 0.7, "debit": 30.0}] for _ in points]
    cells_D = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": D, "qos": 0.75, "debit": 60.0}] for _ in points]

    _, nw_C, _, cout_C, r_fin_C = _evaluer(points, cells_C, [("site_R", "3G", 2100)])
    _, nw_D, _, cout_D, r_fin_D = _evaluer(points, cells_D, [("site_D3", "5G", 3500)])

    assert nw_C == 0 and nw_D == 0
    assert cout_C == 20_000
    assert cout_D == 205_000
    assert r_fin_C > r_fin_D
    print("test_3_deux_solutions_acceptables_couts_differents : OK")


def test_4_couverture_vs_prix():
    points = [0, 1]
    E = ("site_E4", None, "2G", 900)
    F = ("site_F4", None, "5G", 3500)
    cells_E = [
        [{"cellule": R, "qos": 0.3, "debit": 0.5}, {"cellule": E, "qos": 0.5, "debit": 15.0}],
        [{"cellule": R, "qos": 0.3, "debit": 0.5}],
    ]
    cells_F = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": F, "qos": 0.8, "debit": 60.0}] for _ in points]

    _, nw_E, _, cout_E, r_fin_E = _evaluer(points, cells_E, [("site_E4", "2G", 900)])
    _, nw_F, _, cout_F, r_fin_F = _evaluer(points, cells_F, [("site_F4", "5G", 3500)])

    # CHANGÉ : nw_E=1 sur n_total=2 -> Ŵ=0.5 -> -50-50*0.5=-75 (avant : -100 fixe)
    assert nw_E == 1
    assert abs(r_fin_E - (-75.0)) < 1e-9
    assert nw_F == 0 and cout_F > cout_E
    assert r_fin_F > r_fin_E  # toujours vrai, marge plus faible qu'avant mais préservée
    print("test_4_couverture_vs_prix : OK")


def test_5_gain_marginal_vs_surcout():
    points = [0, 1]
    G = ("site_G5", None, "4G", 1800)
    H = ("site_H5", None, "5G", 3500)
    cells_G = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": G, "qos": 0.7, "debit": 40.0}] for _ in points]
    cells_GH = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                 {"cellule": G, "qos": 0.7, "debit": 40.0},
                 {"cellule": H, "qos": 0.85, "debit": 80.0}] for _ in points]

    resultat_G, nw_G, U_G, cout_G, r_fin_G = _evaluer(points, cells_G, [("site_G5", "4G", 1800)])
    resultat_GH, nw_GH, U_GH, cout_GH, r_fin_GH = _evaluer(
        points, cells_GH, [("site_G5", "4G", 1800), ("site_H5", "5G", 3500)])

    assert nw_G == 0 and nw_GH == 0
    assert U_GH > U_G
    assert cout_GH > cout_G
    assert r_fin_GH < r_fin_G
    print("test_5_gain_marginal_vs_surcout : OK")


if __name__ == "__main__":
    test_1_couverture_seule()
    test_2_conflit_couverture_qos()
    test_3_deux_solutions_acceptables_couts_differents()
    test_4_couverture_vs_prix()
    test_5_gain_marginal_vs_surcout()
    print("\nTous les tests passent.")