"""
small_instance.py : exhaustive optimum on the small instance of Experiment F (§4.7.1).
Three grid cells with the most white points x (3G 900, 4G 800, 5G 700) = 9 candidates, 512 subsets.
Coverage-Greedy and Trajectory-Greedy are rerun on the same 9 candidates.
Usage (from the repository root): python experiences/small_instance.py
Output: <repo>/resultats/petite_instance.csv
"""
import itertools, os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "placement"))
import pandas as pd
from baseline_aleatoire import preparer_environnement, DOSSIER_RESULTATS
import baseline_trajectoire_gloutonne as T
from environment_rl import FerroMobileEnv
from cost_model import get_radio_cost, C_SITE_EUR

preparer_environnement()
env = FerroMobileEnv(scenario="S1", budget=10e6, t_max=150, sauvegarder_deploiements_rl=False)
env.reset()
blancs = T.points_blancs(env)
pts = env.df_points[env.df_points.point_id.isin(blancs)]
cnt = collections.Counter(c for c in (env.grille.gps_vers_cellule(r.lat, r.lon) for r in pts.itertuples()) if c)
cells = [c for c, _ in cnt.most_common(3)]
CFG = [("3G", 900), ("4G", 800), ("5G", 700)]
cands = [(c[0], c[1], g, f) for c in cells for g, f in CFG]
print("cells", [(c, cnt[c]) for c in cells], flush=True)
cache = {}
for a in cands:
    T.simuler(env, a, cache)
exist = set(env.sites_deja_presents)
U0, base = env.U_courant, list(env._resultats_hypo)

def cost(sub):
    return sum(get_radio_cost(a[2], a[3]) for a in sub) + C_SITE_EUR * len({(a[0], a[1]) for a in sub} - exist)

def evaluate(sub):
    env._resultats_hypo = base + [r for a in sub for r in cache[a]]
    res, nw, ne = env._evaluer_dp_courant()
    u = env._calculer_U(res, ne, nw)
    env._resultats_hypo = list(base)
    return u, nw

allU = {}
for k in range(len(cands) + 1):
    for sub in itertools.combinations(cands, k):
        allU[sub] = evaluate(sub)
print("evaluated", len(allU), flush=True)

def greedy(B, score):
    sub = ()
    while True:
        best, bs = None, 0.0
        for a in cands:
            if a in sub or cost(sub + (a,)) > B:
                continue
            s = score(sub, a)
            if s > bs:
                best, bs = a, s
        if best is None:
            return sub
        sub = tuple(sorted(sub + (best,), key=cands.index))

def cov_score(sub, a):
    return (allU[sub][1] - allU[tuple(sorted(sub + (a,), key=cands.index))][1]) / (cost(sub + (a,)) - cost(sub))
def tg_score(sub, a):
    return (allU[tuple(sorted(sub + (a,), key=cands.index))][0] - allU[sub][0]) / (cost(sub + (a,)) - cost(sub))

rows = []
key = lambda s: tuple(sorted(s, key=cands.index))
for B in (0.3e6, 0.5e6):
    feas = [s for s in allU if cost(s) <= B]
    opt = max(feas, key=lambda s: allU[s][0])
    Uo = allU[opt][0]
    for name, sub in [("optimum", opt), ("coverage_greedy", key(greedy(B, cov_score))), ("trajectory_greedy", key(greedy(B, tg_score)))]:
        u, nw = allU[sub]
        rows.append(dict(budget=B, method=name, U=round(u, 5), gap=round((Uo - u) / abs(Uo), 4), n_white=nw,
                         cost=cost(sub), deployments=";".join(f"{a[0]}-{a[1]}-{a[2]}{a[3]}" for a in sub), n_feasible=len(feas)))
        print(rows[-1], flush=True)
os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(DOSSIER_RESULTATS, "petite_instance.csv"), index=False)
print("U0", U0)
