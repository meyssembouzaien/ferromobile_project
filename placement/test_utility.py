"""
test_utility.py — Tests unitaires pour utility.py

Ces tests valident le MÉCANISME de U(I) et r_fin (formules, branches,
cas limites), avec des valeurs à la main. Ce ne sont PAS les 5 scénarios
jouets du §26 point 3 (couverture seule, conflit couverture/QoS, etc.) :
ceux-là nécessitent une recherche exhaustive sur un petit cas réel
(DP + simulateur + cost_model ensemble) et sont la prochaine étape,
une fois ce mécanisme de base confirmé correct.
"""

from utility import normaliser_metriques, compute_U, compute_r_t, compute_r_fin


def test_normaliser_metriques_valeurs_de_base():
    resultat_dp = {"qos_mean": 0.8, "N_H": 3, "N_O": 1, "N_T": 2}
    m = normaliser_metriques(resultat_dp, n_eval=5, n_white=10, n_total=100)
    assert m["Q"] == 0.8
    assert m["H"] == 3 / 4
    assert m["O"] == 1 / 4
    assert m["T"] == 2 / 4
    assert m["W"] == 10 / 100
    print("test_normaliser_metriques_valeurs_de_base : OK")


def test_normaliser_metriques_un_seul_point_evalue():
    resultat_dp = {"qos_mean": 0.5, "N_H": 0, "N_O": 0, "N_T": 0}
    m = normaliser_metriques(resultat_dp, n_eval=1, n_white=0, n_total=50)
    assert m["H"] == 0.0 and m["O"] == 0.0 and m["T"] == 0.0
    print("test_normaliser_metriques_un_seul_point_evalue : OK")


def test_compute_U_poids_egaux():
    m = {"Q": 1.0, "W": 0.0, "H": 0.0, "O": 0.0, "T": 0.0}
    u = compute_U(m, w_Q=1.0, w_W=1.0, w_H=1.0, w_O=1.0, w_T=1.0)
    assert u == 1.0
    print("test_compute_U_poids_egaux : OK")


def test_compute_U_penalites_soustraites():
    m = {"Q": 1.0, "W": 0.2, "H": 0.1, "O": 0.1, "T": 0.1}
    u = compute_U(m, w_Q=1.0, w_W=1.0, w_H=1.0, w_O=1.0, w_T=1.0)
    assert abs(u - (1.0 - 0.2 - 0.1 - 0.1 - 0.1)) < 1e-9
    print("test_compute_U_penalites_soustraites : OK")


def test_compute_r_t_difference_simple():
    r = compute_r_t(0.7, 0.5)
    assert abs(r - 0.2) < 1e-9
    print("test_compute_r_t_difference_simple : OK")


def test_r_fin_zone_blanche_domine():
    r = compute_r_fin(n_white=5, qos_min=0.99, cost_total=10_000, budget=1_000_000,
                       Q_critique=0.3, R_couverture=100, R_qualite=50,
                       R_succes=200, R_eff=50)
    assert r == -100
    print("test_r_fin_zone_blanche_domine : OK")


def test_r_fin_qualite_insuffisante():
    r = compute_r_fin(n_white=0, qos_min=0.2, cost_total=10_000, budget=1_000_000,
                       Q_critique=0.3, R_couverture=100, R_qualite=50,
                       R_succes=200, R_eff=50)
    assert r == -50
    print("test_r_fin_qualite_insuffisante : OK")


def test_r_fin_succes_avec_efficacite():
    r = compute_r_fin(n_white=0, qos_min=0.5, cost_total=400_000, budget=1_000_000,
                       Q_critique=0.3, R_couverture=100, R_qualite=50,
                       R_succes=200, R_eff=50)
    attendu = 200 + 50 * (1 - 400_000 / 1_000_000)
    assert abs(r - attendu) < 1e-9
    print("test_r_fin_succes_avec_efficacite : OK")


def test_r_fin_succes_cout_nul_bonus_maximal():
    r = compute_r_fin(n_white=0, qos_min=0.9, cost_total=0, budget=1_000_000,
                       Q_critique=0.3, R_couverture=100, R_qualite=50,
                       R_succes=200, R_eff=50)
    assert r == 200 + 50
    print("test_r_fin_succes_cout_nul_bonus_maximal : OK")


def test_r_fin_rejette_poids_incoherents():
    try:
        compute_r_fin(n_white=0, qos_min=0.9, cost_total=0, budget=1_000_000,
                      Q_critique=0.3, R_couverture=50, R_qualite=50,
                      R_succes=200, R_eff=50)
        assert False, "devrait lever ValueError"
    except ValueError:
        print("test_r_fin_rejette_poids_incoherents : OK")


if __name__ == "__main__":
    test_normaliser_metriques_valeurs_de_base()
    test_normaliser_metriques_un_seul_point_evalue()
    test_compute_U_poids_egaux()
    test_compute_U_penalites_soustraites()
    test_compute_r_t_difference_simple()
    test_r_fin_zone_blanche_domine()
    test_r_fin_qualite_insuffisante()
    test_r_fin_succes_avec_efficacite()
    test_r_fin_succes_cout_nul_bonus_maximal()
    test_r_fin_rejette_poids_incoherents()
    print("\nTous les tests passent.")
