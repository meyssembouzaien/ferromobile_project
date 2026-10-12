"""
tg_variant.py : Trajectory-Greedy avec un paramètre modifié (Expérience D, tableau 4.5).

Lancer (depuis la racine) :
    python experiences/tg_variant.py csite_half | csite_double | wW_double | wW5 | fifth_op
Sortie : resultats/trajectoire_gloutonne_*_<variante>.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "placement"))
variant = sys.argv[1]
import baseline_trajectoire_gloutonne as T
import cost_model, environment_rl
T.FICHIER_SITES = T.FICHIER_SITES.replace(".csv", f"_{variant}.csv")
T.FICHIER_RESUME = T.FICHIER_RESUME.replace(".csv", f"_{variant}.csv")
if variant == "csite_half":
    cost_model.C_SITE_EUR = 62_500
elif variant == "csite_double":
    cost_model.C_SITE_EUR = 250_000
elif variant in ("wW_double", "wW5"):
    # wW5 : même w_W que les runs DQN principaux
    w_W = 2.0 if variant == "wW_double" else 5.0
    Base = environment_rl.FerroMobileEnv
    class Env(Base):
        def __init__(self, *a, **k):
            k["poids"] = {"w_Q": 1.0, "w_W": w_W, "w_H": 1.0, "w_O": 1.0, "w_T": 1.0}
            super().__init__(*a, **k)
    environment_rl.FerroMobileEnv = Env
elif variant == "fifth_op":
    # Cellules ajoutées = cinquième opérateur (changement d'opérateur compté)
    import simulateur_deploiement
    _orig = simulateur_deploiement.simuler_deploiement
    def _sim(*a, **k):
        res = _orig(*a, **k)
        for r in res:
            site, _, g, f = r["cellule"]
            r["cellule"] = (site, "HYPO", g, f)
        return res
    simulateur_deploiement.simuler_deploiement = _sim
    environment_rl.simuler_deploiement = _sim
else:
    raise SystemExit(f"unknown variant {variant}")
T.preparer_environnement()
T.lancer()
