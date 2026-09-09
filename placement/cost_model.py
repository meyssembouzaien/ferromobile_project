"""
cost_model.py — Modèle de coût d'infrastructure (cahier des charges §12)
==========================================================================
Source de vérité unique pour Cost(I). enrich_data.py peut importer
get_radio_cost() d'ici plutôt que de garder sa propre copie de
INFRA_COSTS — à faire dans un second temps, sans changer les valeurs.

Deux coûts séparés (§12), jamais mélangés :
  - C_radio(g,f) : coût de l'équipement radio. Dépend de la génération
    pour toutes les générations, et en plus de la bande seulement pour
    la 4G (bande basse/haute, seuil 1800 MHz) — reflète exactement ce
    que fait enrich_data.py.
  - C_site : coût de construction du site (mât, alimentation,
    raccordement), payé UNE SEULE FOIS par site, jamais une fois par
    génération ajoutée dessus par la suite.

Valeurs :
  C_radio reprend exactement les coûts déjà utilisés dans le pipeline
  existant (enrich_data.py, INFRA_COSTS : composante installation
  seulement, pas le coût mensuel ni la conso W, qui ne font pas partie
  de C_total(a)).
  C_site ≈ 125 000 € (cahier des charges §12) : estimation sourcée sur
  le coût moyen de construction d'un pylône en Europe de l'Ouest
  (dgtlinfra.com, 2026). Hypothèse de modélisation, pas une valeur
  mesurée sur ce projet ; fait l'objet d'une analyse de sensibilité
  (§23, Expérience D).

Le budget B n'apparaît PAS dans ce fichier : c'est un paramètre de
l'environnement RL (§13), pas du modèle de coût lui-même. cost_model.py
répond uniquement à "combien coûte cette action ?", jamais à "a-t-on le
budget pour la faire ?".

IMPORTANT — identification du site (§13) :
  Un site est "existant" s'il appartient à la MÊME CELLULE DE GRILLE
  qu'un site déjà présent dans I (réseau initial ou ajouts précédents
  de l'épisode), pas s'il a exactement les mêmes coordonnées GPS.
  site_existe_deja() et get_total_cost() ci-dessous supposent donc que
  l'appelant leur passe des sites déjà normalisés à la résolution de la
  grille (via grid_config.GrilleCorridor.gps_vers_cellule, typiquement
  un tuple d'indices (i,j)), PAS des lat/lon bruts arrondis comme le
  fait aujourd'hui trajectory_dp.py pour ses cellules. Cette conversion
  est la responsabilité de l'environnement RL, pas de ce fichier.
"""

C_SITE_EUR = 125_000

RADIO_COSTS_EUR = {
    "2G": 10_000,
    "3G": 20_000,
    "4G_LB": 30_000,  # bande < 1800 MHz
    "4G_HB": 35_000,  # bande >= 1800 MHz
    "5G": 80_000,
}

SEUIL_4G_HAUTE_BANDE_MHZ = 1800


def get_radio_cost(generation, bande_mhz):
    """C_radio(g,f), §12. Dépend de g pour toutes les générations, et
    en plus de f seulement pour la 4G (bande basse vs haute)."""
    g = str(generation).upper()
    if g == "5G":
        return RADIO_COSTS_EUR["5G"]
    if g == "4G":
        if float(bande_mhz) >= SEUIL_4G_HAUTE_BANDE_MHZ:
            return RADIO_COSTS_EUR["4G_HB"]
        return RADIO_COSTS_EUR["4G_LB"]
    if g == "3G":
        return RADIO_COSTS_EUR["3G"]
    if g == "2G":
        return RADIO_COSTS_EUR["2G"]
    raise ValueError(f"cost_model: génération inconnue : {generation!r}")


def site_existe_deja(site, sites_deja_presents):
    """site : identifiant de cellule de grille (voir note d'en-tête, PAS
    des lat/lon bruts). sites_deja_presents : ensemble des sites déjà
    dans l'infrastructure I à ce moment de l'épisode."""
    return site in sites_deja_presents


def get_total_cost(site, generation, bande_mhz, sites_deja_presents):
    """C_total(a), §12. Additionne C_site seulement si `site` n'est pas
    déjà dans sites_deja_presents."""
    cout = get_radio_cost(generation, bande_mhz)
    if not site_existe_deja(site, sites_deja_presents):
        cout += C_SITE_EUR
    return cout
