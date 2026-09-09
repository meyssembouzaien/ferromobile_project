"""
test_r_fin.py — Validation de U(I) et r_fin sur 5 scénarios jouets (§26 point 3)
==================================================================================
Chaque scénario évalue explicitement les configurations pertinentes d'un
petit espace jouet, afin d'isoler un seul comportement précis de r_fin
à la fois — ce n'est PAS une recherche exhaustive au sens de
l'Expérience F (§23), qui énumérera un vrai espace de placements sur
un petit cas pour chercher l'optimum. Ici on compare 2 configurations
choisies à la main par scénario (pas de vraies données, pas de
simulateur_deploiement), comme fait pour trajectory_dp.py.

Convention commune à tous les scénarios : un "réseau réel" faible R
existe aux deux points du trajet jouet (qos=0.3, débit=0.5 Mbps, sous
D_cov=1.0), pour que trajectory_dp() ait toujours au moins un candidat
et ne lève jamais d'erreur, même quand aucun déploiement candidat n'est
choisi. Le site de R fait partie de sites_deja_presents pour tous les
calculs de coût — c'est le réseau initial, non facturé (cohérent avec
valider_chaine_complete.py : seul l'AJOUT décidé est facturé).

Le budget (500 000 €) est choisi assez large pour que TOUTES les
configurations testées, y compris la plus chère (G+H, 365 000 € au
scénario 5), restent Cost(I) <= B. C'est important : dans l'environnement
RL réel, Cost(I) <= B est une contrainte de faisabilité stricte (§13) —
une action qui dépasserait le budget restant serait masquée avant même
d'être exécutée, donc r_fin ne serait jamais évalué sur une
configuration hors budget. Tester r_fin sur un cas Cost(I) > B
reviendrait à évaluer un état que le RL ne peut jamais produire ; ce
fichier reste donc dans le domaine des configurations admissibles.

Paramètres de récompense utilisés dans CE fichier (choix délibéré, pas
une valeur imposée par le cahier des charges — voir §14 : "des
paramètres de modèle, testés par sensibilité") :
    Q_critique = 0.5, R_couverture = 100, R_qualité = 50,
    R_succès = 20, R_eff = 20, budget = 500 000 €

Ces valeurs sont choisies pour que R_couverture domine toujours R_qualité,
qui domine toujours le meilleur retour positif atteignable (R_succès +
R_eff = 40 max). Le §14 précise explicitement que cette hiérarchie
(couverture > qualité > tout le reste) n'est PAS garantie
mathématiquement par la seule condition R_couverture > R_qualité > 0 —
elle dépend aussi de R_succès, R_eff et du budget. Ces 5 scénarios
vérifient qu'avec CES valeurs précises, la hiérarchie voulue tient bien ;
un autre jeu de poids pourrait ne pas la préserver, d'où l'utilité de
l'analyse de sensibilité (§23, Expérience D) au-delà de cette simple
validation de conception.

Lancer : python test_r_fin.py
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

R = ("site_R", "ORANGE", "2G", 900)  # réseau réel faible, commun aux 5 scénarios


def _n_white(points, cells_avec_debit, D_cov=D_COV):
    """N_white jouet : chaque candidat porte un champ 'debit' en plus de
    'qos' (trajectory_dp ignore ce champ, il ne lit que 'cellule'/'qos')."""
    return sum(
        1 for t in range(len(points))
        if max(c["debit"] for c in cells_avec_debit[t]) < D_cov
    )


def _evaluer(points, cells_par_point, candidats_choisis):
    """candidats_choisis : liste de (site, generation, bande) à facturer
    en plus du réseau réel (site_R, déjà présent, non facturé)."""
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

    r_fin = compute_r_fin(nw, resultat["qos_min"], cout_total, BUDGET,
                           Q_CRITIQUE, R_COUVERTURE, R_QUALITE, R_SUCCES, R_EFF)
    return resultat, nw, U, cout_total, r_fin


def test_1_couverture_seule():
    """Sans déploiement, le réseau réel faible laisse les 2 points en
    zone blanche (débit 0.5 < D_cov=1.0). Un déploiement 5G nouveau site
    (A) couvre tout. {A} doit l'emporter très largement sur l'absence de
    déploiement : la pénalité de non-couverture (-100) domine tout le
    reste, quel que soit son coût (205 000 €)."""
    points = [0, 1]
    A = ("site_A", None, "5G", 3500)

    cells_sans = [[{"cellule": R, "qos": 0.3, "debit": 0.5}] for _ in points]
    cells_avec = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                   {"cellule": A, "qos": 0.9, "debit": 50.0}] for _ in points]

    _, nw_sans, _, cout_sans, r_fin_sans = _evaluer(points, cells_sans, [])
    _, nw_avec, _, cout_avec, r_fin_avec = _evaluer(points, cells_avec, [("site_A", "5G", 3500)])

    assert nw_sans == 2 and r_fin_sans == -R_COUVERTURE
    assert nw_avec == 0 and cout_avec == 205_000
    assert abs(r_fin_avec - 31.8000) < 1e-3
    assert r_fin_avec > r_fin_sans  # {A} l'emporte largement
    print("test_1_couverture_seule : OK")


def test_2_conflit_couverture_qos():
    """Deux déploiements couvrent tout (N_white=0) : A (moins cher,
    155 000 €) a une QoS insuffisante (0.4 < Q_critique=0.5) ; B (plus
    cher, 205 000 €) passe le seuil (0.7). B doit l'emporter malgré son
    coût plus élevé : le coût n'intervient qu'APRÈS que couverture et
    qualité soient satisfaites (§14)."""
    points = [0, 1]
    A = ("site_A2", None, "4G", 800)
    B = ("site_B2", None, "5G", 3500)

    cells_A = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": A, "qos": 0.4, "debit": 20.0}] for _ in points]
    cells_B = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": B, "qos": 0.7, "debit": 50.0}] for _ in points]

    _, nw_A, _, cout_A, r_fin_A = _evaluer(points, cells_A, [("site_A2", "4G", 800)])
    _, nw_B, _, cout_B, r_fin_B = _evaluer(points, cells_B, [("site_B2", "5G", 3500)])

    assert nw_A == 0 and r_fin_A == -R_QUALITE  # couvre mais qos insuffisante
    assert nw_B == 0 and cout_B > cout_A  # B plus cher que A
    assert abs(r_fin_B - 31.8000) < 1e-3
    assert r_fin_B > r_fin_A  # B l'emporte malgré son coût plus élevé
    print("test_2_conflit_couverture_qos : OK")


def test_3_deux_solutions_acceptables_couts_differents():
    """C (upgrade du site réel existant, 20 000 €, qos=0.70) et D
    (nouveau site 5G, 205 000 €, qos=0.75) satisfont tous les deux
    couverture et qualité. C doit l'emporter : entre deux solutions
    acceptables, r_fin favorise le coût le plus faible (§14), même si D
    a une QoS légèrement meilleure — Q_min n'intervient qu'en seuil
    binaire (Q_critique), pas en optimisation continue dans r_fin."""
    points = [0, 1]
    C = ("site_R", "ORANGE", "3G", 2100)  # même site que le réel : pas de C_site
    D = ("site_D3", None, "5G", 3500)

    cells_C = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": C, "qos": 0.7, "debit": 30.0}] for _ in points]
    cells_D = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": D, "qos": 0.75, "debit": 60.0}] for _ in points]

    _, nw_C, _, cout_C, r_fin_C = _evaluer(points, cells_C, [("site_R", "3G", 2100)])
    _, nw_D, _, cout_D, r_fin_D = _evaluer(points, cells_D, [("site_D3", "5G", 3500)])

    assert nw_C == 0 and nw_D == 0
    assert cout_C == 20_000  # pas de C_site, site déjà présent
    assert cout_D == 205_000
    assert r_fin_C > r_fin_D  # la solution la moins chère l'emporte
    print("test_3_deux_solutions_acceptables_couts_differents : OK")


def test_4_couverture_vs_prix():
    """E (135 000 €) ne couvre qu'un des deux points ; F (205 000 €)
    couvre tout. F doit l'emporter malgré un coût plus élevé : la
    pénalité de non-couverture (-100) domine toujours, avec les poids
    choisis dans ce fichier."""
    points = [0, 1]
    E = ("site_E4", None, "2G", 900)
    F = ("site_F4", None, "5G", 3500)

    cells_E = [
        [{"cellule": R, "qos": 0.3, "debit": 0.5}, {"cellule": E, "qos": 0.5, "debit": 15.0}],
        [{"cellule": R, "qos": 0.3, "debit": 0.5}],  # E ne couvre pas ce point
    ]
    cells_F = [[{"cellule": R, "qos": 0.3, "debit": 0.5},
                {"cellule": F, "qos": 0.8, "debit": 60.0}] for _ in points]

    _, nw_E, _, cout_E, r_fin_E = _evaluer(points, cells_E, [("site_E4", "2G", 900)])
    _, nw_F, _, cout_F, r_fin_F = _evaluer(points, cells_F, [("site_F4", "5G", 3500)])

    assert nw_E == 1 and r_fin_E == -R_COUVERTURE
    assert nw_F == 0 and cout_F > cout_E
    assert r_fin_F > r_fin_E  # F l'emporte malgré son coût plus élevé
    print("test_4_couverture_vs_prix : OK")


def test_5_gain_marginal_vs_surcout():
    """G seul (160 000 €) couvre tout et passe Q_critique (qos=0.70).
    Ajouter H (site nouveau, +205 000 €) améliore la QoS (0.70 -> 0.85)
    mais ne change ni la couverture ni le passage du seuil, déjà acquis
    avec G seul. r_fin doit préférer G seul : une fois les critères
    d'acceptabilité satisfaits, payer plus pour un gain de qualité qui
    ne change pas l'acceptabilité n'est pas récompensé par r_fin.

    Point important pour le mémoire : U(I) AUGMENTE bien avec G+H
    (0.70 -> 0.85, via Q_hat) — donc r_t récompenserait ce gain marginal
    pendant l'épisode. C'est r_fin, calculé une seule fois à la fin, qui
    corrige cet effet en tenant compte du coût. C'est exactement le rôle
    prévu pour r_fin par le §14 : r_t seul ne suffit pas à décourager un
    déploiement inutile, parce qu'il ne contient jamais le coût."""
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
    assert U_GH > U_G  # U(I) récompense le gain marginal de QoS...
    assert cout_GH > cout_G
    assert r_fin_GH < r_fin_G  # ...mais r_fin le pénalise à cause du coût
    print("test_5_gain_marginal_vs_surcout : OK")


if __name__ == "__main__":
    test_1_couverture_seule()
    test_2_conflit_couverture_qos()
    test_3_deux_solutions_acceptables_couts_differents()
    test_4_couverture_vs_prix()
    test_5_gain_marginal_vs_surcout()
    print("\nTous les tests passent.")
