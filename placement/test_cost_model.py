"""
test_cost_model.py — Tests unitaires pour cost_model.py
"""

from cost_model import get_radio_cost, get_total_cost, C_SITE_EUR


def test_radio_cost_par_generation():
    assert get_radio_cost("2G", 900) == 10_000
    assert get_radio_cost("3G", 2100) == 20_000
    assert get_radio_cost("5G", 3500) == 80_000
    print("test_radio_cost_par_generation : OK")


def test_radio_cost_4g_bande():
    assert get_radio_cost("4G", 700) == 30_000
    assert get_radio_cost("4G", 800) == 30_000
    assert get_radio_cost("4G", 1799) == 30_000
    assert get_radio_cost("4G", 1800) == 35_000
    assert get_radio_cost("4G", 2600) == 35_000
    print("test_radio_cost_4g_bande : OK")


def test_site_existant_pas_de_cout_site():
    sites = {("cell", 12, 34)}
    cout = get_total_cost(("cell", 12, 34), "5G", 3500, sites)
    assert cout == 80_000  # pas de C_SITE ajouté
    print("test_site_existant_pas_de_cout_site : OK")


def test_nouveau_site_ajoute_cout_site():
    sites = {("cell", 12, 34)}
    cout = get_total_cost(("cell", 56, 78), "2G", 900, sites)
    assert cout == 10_000 + C_SITE_EUR
    print("test_nouveau_site_ajoute_cout_site : OK")


def test_c_site_paye_une_seule_fois_par_site():
    # Deux ajouts successifs sur le MÊME nouveau site : le premier paie
    # C_SITE, le second ne doit pas le repayer si l'appelant a bien mis
    # à jour sites_deja_presents entre les deux (simule I qui évolue
    # pendant l'épisode).
    sites = set()
    site = ("cell", 90, 10)
    cout_1 = get_total_cost(site, "4G", 700, sites)
    assert cout_1 == 30_000 + C_SITE_EUR
    sites.add(site)
    cout_2 = get_total_cost(site, "5G", 3500, sites)
    assert cout_2 == 80_000
    print("test_c_site_paye_une_seule_fois_par_site : OK")


def test_generation_inconnue_leve_erreur():
    try:
        get_radio_cost("6G", 900)
        assert False, "devrait lever ValueError"
    except ValueError:
        print("test_generation_inconnue_leve_erreur : OK")


if __name__ == "__main__":
    test_radio_cost_par_generation()
    test_radio_cost_4g_bande()
    test_site_existant_pas_de_cout_site()
    test_nouveau_site_ajoute_cout_site()
    test_c_site_paye_une_seule_fois_par_site()
    test_generation_inconnue_leve_erreur()
    print("\nTous les tests passent.")
