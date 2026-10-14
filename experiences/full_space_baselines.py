"""
full_space_baselines.py : ILP set cover and Coverage-Greedy over the full action space (all corridor cells x 10 pairs).

The candidate coverage comes from a geometric pre-computation (range + line of sight), which is a superset of the
exact coverage (exact = range + LOS + debit_adj >= D_COV after rain and ARQ, from env._simuler).
  - ILP : solved on the superset, then the chosen deployments are checked exactly and their coverage replaced;
    repeated until every chosen deployment is exact. The result is then optimal for the exact coverage,
    since every other deployment can only cover fewer points than in the superset.
  - Coverage-Greedy : lazy greedy. Superset gains are upper bounds; a deployment is chosen only once its exact
    gain is at least the bound of every other one. Same rule as baseline_couverture_gloutonne.py, on all cells.
Both plans are then replayed in the environment (same metrics as the other baselines).

Usage (Kaggle, with DEM + crc-covlib) :
  python full_space_baselines.py <repo_root> --couv couverture_potentielle_S1.npz
Outputs : <repo_root>/resultats/full_space_*.csv
"""
import argparse, heapq, json, os, sys, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

ap = argparse.ArgumentParser()
ap.add_argument("racine")
ap.add_argument("--couv", required=True)        # couverture_potentielle_S1.npz
ap.add_argument("--temps_max_s", type=int, default=5400)
a = ap.parse_args()

sys.path.insert(0, os.path.join(a.racine, "placement"))
os.environ.setdefault("CRC_COVLIB_PATH", os.path.join(a.racine, "crc-covlib", "python-wrapper"))
from cost_model import get_radio_cost, get_total_cost, C_SITE_EUR
from environment_rl import FerroMobileEnv
from baseline_aleatoire import resume_episode

D_COV = 1.0
OUT = os.path.join(a.racine, "resultats"); os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f} s] {msg}", flush=True)


env = FerroMobileEnv(scenario="S1", budget=10_000_000.0, t_max=150, sauvegarder_deploiements_rl=False)
env.reset()
debit = env.debit_reel_par_point.reindex(list(env.points_hors_tunnel)).fillna(0)
blancs = set(debit[debit < D_COV].index)
sites0 = set(env.sites_deja_presents)
log(f"{len(blancs)} white points, {len(sites0)} existing sites")

# Superset coverage (couverture_potentielle.py): range + LOS for every corridor cell and initial white point
Z = np.load(a.couv)
from prepare_data import TECH_PROFILES
from environment_rl import BANDES_PAR_GENERATION
ny = env.grille.ny
pid = list(Z["points"])
couv = {}
for i, c in enumerate(Z["cellules"]):
    ix, iy = int(c) // ny, int(c) % ny
    for g in BANDES_PAR_GENERATION:
        los = Z[f"los_{g}"][i]
        for f in BANDES_PAR_GENERATION[g]:
            d = (ix, iy, g, int(f))
            if d in env.deploiements_reels:
                continue
            m = los & (Z["dist_km"][i] <= TECH_PROFILES[(g, int(f))][0])
            pts = {pid[j] for j in np.flatnonzero(m)} & blancs
            if pts:
                couv[d] = pts
manquants = blancs - set().union(*couv.values())
log(f"{len(Z['cellules'])} cells, {len(couv)} deployments cover at least one white point; "
    f"{len(manquants)} white points covered by none (superset); file has {len(pid)} white points")

exact = {}


def couverture_exacte(d):
    if d not in exact:
        lat, lon = env.grille.cellule_vers_gps(d[0], d[1])
        exact[d] = {r["point_id"] for r in env._simuler(d, lat, lon)
                    if r["point_id"] in blancs and r["debit_adj_mbps"] >= D_COV}
    return exact[d]


def cov(d):
    return exact[d] if d in exact else couv[d]


# Check the superset assumption on a random sample (exact coverage must be inside the geometric one)
rng = np.random.default_rng(0)
liste = list(couv)
echantillon = [liste[i] for i in rng.choice(len(liste), size=min(150, len(liste)), replace=False)]
violations = sum(1 for d in echantillon if not couverture_exacte(d) <= couv[d])
log(f"superset check: {violations} of {len(echantillon)} sampled deployments cover a point outside the superset")


# ---------------------------------------------------------------- ILP
def resoudre_ilp():
    acts = [d for d in couv if cov(d)]
    nouv = sorted({(d[0], d[1]) for d in acts} - sites0)
    iy = {c: i for i, c in enumerate(nouv)}
    pts = sorted(set().union(*[cov(d) for d in acts]))
    nA, nY = len(acts), len(nouv)
    cout = np.r_[[get_radio_cost(d[2], d[3]) for d in acts], [C_SITE_EUR] * nY]
    rows, cols, vals, lo, hi, r = [], [], [], [], [], 0
    for j, d in enumerate(acts):
        c = (d[0], d[1])
        if c in iy:
            rows += [r, r]; cols += [j, nA + iy[c]]; vals += [1, -1]; lo.append(-np.inf); hi.append(0); r += 1
    couvreurs = {p: [] for p in pts}
    for j, d in enumerate(acts):
        for p in cov(d):
            couvreurs[p].append(j)
    for p in pts:
        rows += [r] * len(couvreurs[p]); cols += couvreurs[p]; vals += [1] * len(couvreurs[p])
        lo.append(1); hi.append(np.inf); r += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nA + nY))
    res = milp(cout, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nA + nY), bounds=Bounds(0, 1),
               options={"time_limit": a.temps_max_s, "mip_rel_gap": 0.0})
    if res.x is None:
        raise RuntimeError(res.message)
    choisies = [d for j, d in enumerate(acts) if res.x[j] > 0.5]
    return choisies, float(cout @ np.round(res.x)), res.status == 0, len(pts), nA


it = 0
while True:
    it += 1
    choisies, cout_ilp, prouve, n_pts, nA = resoudre_ilp()
    a_verifier = [d for d in choisies if d not in exact]
    log(f"ILP iteration {it}: {len(choisies)} deployments, {cout_ilp:,.0f} EUR, proven={prouve}, "
        f"{n_pts} coverable points, {nA} candidates, {len(a_verifier)} to check")
    if not a_verifier:
        break
    for d in a_verifier:
        couverture_exacte(d)
ilp = {"cout_eur": cout_ilp, "prouve_optimal": prouve, "iterations": it, "deploiements": len(choisies),
       "nouveaux_sites": len({(d[0], d[1]) for d in choisies} - sites0),
       "technologies": json.dumps(pd.Series([f"{d[2]} {d[3]}" for d in choisies]).value_counts().to_dict()),
       "points_couvrables": n_pts, "verifs_exactes": len(exact)}
log(f"ILP done: {ilp}")
pd.DataFrame(choisies, columns=["ix", "iy", "generation", "bande_mhz"]).to_csv(
    os.path.join(OUT, "full_space_ilp_set_cover.csv"), index=False)


# ---------------------------------------------------------------- Coverage-Greedy (lazy)
def coverage_greedy():
    reste, sites, budget, plan = set(blancs), set(sites0), 10_000_000.0, []
    tas = []

    def cle(d):
        return len(cov(d) & reste) / get_total_cost((d[0], d[1]), d[2], d[3], sites)

    for d in couv:
        heapq.heappush(tas, (-cle(d), d))
    deja = set()
    while reste and tas:
        borne, d = heapq.heappop(tas)
        if d in deja:
            continue
        cout = get_total_cost((d[0], d[1]), d[2], d[3], sites)
        if cout > budget:
            continue
        if d not in exact:
            couverture_exacte(d)
        v = cle(d)
        if v <= 0:
            continue
        if tas and v < -tas[0][0] - 1e-12:
            heapq.heappush(tas, (-v, d))
            continue
        plan.append(d); deja.add(d); budget -= cout
        reste -= cov(d)
        if (d[0], d[1]) not in sites:   # cost drops for the other pairs of this cell: refresh their keys
            sites.add((d[0], d[1]))
            for g in couv:
                if (g[0], g[1]) == (d[0], d[1]) and g not in deja:
                    heapq.heappush(tas, (-cle(g), g))
        log(f"  CG site {len(plan)}: {d}, white left {len(reste)}, cost {10_000_000.0 - budget:,.0f} EUR")
    return plan


plan_cg = coverage_greedy()
pd.DataFrame(plan_cg, columns=["ix", "iy", "generation", "bande_mhz"]).to_csv(
    os.path.join(OUT, "full_space_coverage_greedy.csv"), index=False)


# ---------------------------------------------------------------- replay in the environment
def rejouer(plan, nom):
    debut = time.time()
    env.reset()
    total, info, n = 0.0, {"n_white": env._dernier_n_white}, 0
    done = False
    for d in plan:
        if done:
            break
        _, r, done, info = env.step(d)
        total += r; n += 1
    if not done:
        _, r, done, info = env.step("STOP")
        total += r
    res = resume_episode(env, info, total, n, time.time() - debut)
    res["methode"] = nom
    log(f"{nom}: {res}")
    return res


lignes = [dict(rejouer(choisies, "ILP full space"), **{k: v for k, v in ilp.items()}),
          rejouer(plan_cg, "Coverage-Greedy full space")]
pd.DataFrame(lignes).to_csv(os.path.join(OUT, "full_space_resume.csv"), index=False)
log("done")
