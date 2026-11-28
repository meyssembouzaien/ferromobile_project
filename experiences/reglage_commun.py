"""
reglage_commun.py : le seul réglage de toutes les expériences.

Les méthodes de référence et l'agent lisent ce fichier : même environnement, mêmes actions permises,
même budget, mêmes critères, mêmes graines, même limite de temps, même machine.
Changer une valeur ici la change partout.
"""

# Environnement (identique pour toutes les méthodes, à l'entraînement comme au test)
ENV = {
    "scenario": "S1",
    "budget": 10_000_000.0,     # euros
    "t_max": 150,               # actions maximum par épisode
    "masque_los": True,         # actions permises : celles qui couvriraient un point blanc (portée + LOS)
    "poids": {"w_Q": 1.0, "w_W": 5.0, "w_H": 1.0, "w_O": 1.0, "w_T": 1.0},   # poids de U (agent et Trajectory-Greedy)
}
D_COV_MBPS = 1.0                # point blanc : débit < 1 Mbit/s
Q_CRITIQUE = 0.5                # seuil de q_min dans r_fin (évaluation et entraînement)

# Répétitions
GRAINES = [0, 1, 2]             # politique aléatoire et agent : 3 graines chacun
TEMPS_MAX_S = 3600              # par méthode et par configuration (solveur ILP, recherche gloutonne)
K_PRESELECTION = 200            # Trajectory-Greedy : déploiements testés exactement à chaque étape

# Configurations de test (ligne 1). La ligne 2 utilise seulement "complet".
OPERATEURS = ["ORANGE", "SFR", "BOUYGUES TELECOM", "FREE MOBILE"]
CONFIGURATIONS = {"complet": {}}
CONFIGURATIONS.update({f"sans_secteur_{k}": {"secteur_retire": k} for k in range(4)})
CONFIGURATIONS.update({f"sans_{o.split()[0].lower()}": {"operateurs_exclus": [o]} for o in OPERATEURS})

# Entraînement de l'agent (train_dqn.py), depuis zéro, sans affinage
ENTRAINEMENT = {
    "FM_ARCH": "dueling",
    "FM_EXPLORATION": "softmax",
    "FM_EPISODES": "1000",      # checkpoint final fixé à l'avance : épisode 1000
    "FM_W_W": "5",
    "FM_LR": "5e-5",
    "FM_MAJ_CIBLE": "1000",
    "FM_COST_STEP": "0.2",      # coût de chaque action dans la récompense de l'étape
    "FM_LOS_MASK": "1",
    "FM_SITE_DROP": "0.3",      # réseaux existants variés
    "FM_CLUSTER_DROP": "1",     # pannes régionales
    "FM_CLUSTER_MAX": "0.5",
    "FM_CLUSTER_PROB": "0.5",
    "FM_MAX_HOURS": "11",
}
EPISODE_FINAL = int(ENTRAINEMENT["FM_EPISODES"])


def variables_entrainement(graine, secteur_protege=None):
    """Variables d'environnement d'un run : même recette, seule la graine (et le pli de CV) change."""
    v = dict(ENTRAINEMENT, FM_SEED=str(graine))
    nom = f"commun_seed{graine}"
    if secteur_protege is not None:
        v["FM_CV_SECTOR"] = str(secteur_protege)
        nom += f"_cv{secteur_protege}"
    v["FM_RUN"] = nom
    return v


def options_env(**config):
    """Arguments de FerroMobileEnv pour une configuration de test."""
    return dict(ENV, Q_critique=Q_CRITIQUE, budget_reference=ENV["budget"],
                sauvegarder_deploiements_rl=False, **config)
