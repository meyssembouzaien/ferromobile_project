"""
tg_variant.py : Trajectory-Greedy under one planning-parameter variant (Experiment D).
Usage (from the repository root):
    python experiences/tg_variant.py csite_half | csite_double | wW_double | wW5 | fifth_op
Outputs: <repo>/resultats/trajectoire_gloutonne_{sites,resume}_<variant>.csv
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
    # wW5 : same white-zone weight as the main DQN runs, for a like-for-like comparison
    w_W = 2.0 if variant == "wW_double" else 5.0
    Base = environment_rl.FerroMobileEnv
    class Env(Base):
        def __init__(self, *a, **k):
            k["poids"] = {"w_Q": 1.0, "w_W": w_W, "w_H": 1.0, "w_O": 1.0, "w_T": 1.0}
            super().__init__(*a, **k)
    environment_rl.FerroMobileEnv = Env
elif variant == "fifth_op":
    # hypothetical cells belong to a fifth, neutral operator: entering or leaving them counts as an operator change
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
