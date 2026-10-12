"""
ilp_reference.py : référence exacte (sur un ensemble de candidats) pour le chapitre 4.

  (P1) set cover : coût minimal pour n_white = 0 ;
  (P2) couverture maximale sous budget B (courbe coût -> zones blanches).

Valide parce que la couverture est séparable : un point est couvert si AU MOINS UNE antenne lui donne
debit_adj >= D_COV, et le résultat d'une antenne ne dépend pas des autres (simuler_deploiement).
Coût identique à cost_model.py : C_site payé une fois par cellule nouvelle, C_radio par (g, f).
Optimalité garantie SUR L'ENSEMBLE DE CANDIDATS (cellules à <= --rayon cellules d'un point blanc) :
c'est une borne supérieure du vrai optimum si l'on élargit le rayon ; vérifier la stabilité en rayon.

Usage (Kaggle, avec DEM + crc-covlib) : python ilp_reference.py <racine_repo> --rayon 2
Test de plomberie sans DEM :          python ilp_reference.py <racine_repo> --mock
Sorties : <racine_repo>/resultats/ilp_*.csv, cache de couverture ilp_couverture_cache.pkl (reprise).
"""
import argparse, os, pickle, sys, time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

ap = argparse.ArgumentParser()
ap.add_argument("racine"); ap.add_argument("--rayon", type=int, default=2)
ap.add_argument("--mock", action="store_true"); ap.add_argument("--scenario", default="S1")
ap.add_argument("--budgets", default="0.5,1,1.5,2,2.5,3,3.5,4,5")
ap.add_argument("--temps_max_s", type=int, default=3600)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.racine, "placement"))
from grid_config import construire_grille_depuis_dataset
from cost_model import get_radio_cost, C_SITE_EUR

D_COV = 1.0
CONFIGS = [("3G", 900), ("3G", 2100), ("4G", 700), ("4G", 800), ("4G", 1800), ("4G", 2100), ("4G", 2600),
           ("5G", 700), ("5G", 2100), ("5G", 3500)]   # = BANDES_PAR_GENERATION (2G exclu)
OUT = os.path.join(a.racine, "resultats"); os.makedirs(OUT, exist_ok=True)

b = pd.read_csv(os.path.join(a.racine, "data/processed/dataset_base.csv"), low_memory=False)
e = pd.read_csv(os.path.join(a.racine, "data/processed/dataset_enrichi.csv"), low_memory=False)
s = e[e.scenario_id == a.scenario]
grille = construire_grille_depuis_dataset(b)
df_points = b[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]].drop_duplicates("point_id")
ht = set(df_points[~df_points.in_tunnel].point_id)
debit = s[~s.in_tunnel].groupby("point_id").debit_adj_mbps.max().reindex(list(ht)).fillna(0)
blancs = set(debit[debit < D_COV].index)
df_meteo = s[["point_id", "pluie_mm_h", "temp_c"]].drop_duplicates("point_id")
reel = s[(~s.in_tunnel) & (s.ant_id.notna())].drop_duplicates("ant_id")
sites_existants = {c for c in (grille.gps_vers_cellule(r.ant_lat, r.ant_lon) for r in reel.itertuples()) if c}
deja = {(c[0], c[1], r.generation, int(r.bande_mhz)) for r in reel.itertuples()
        for c in [grille.gps_vers_cellule(r.ant_lat, r.ant_lon)] if c}
print(f"{len(blancs)} points blancs, {len(sites_existants)} sites réels sur la grille")

cellules = set()
for r in df_points[df_points.point_id.isin(blancs)].itertuples():
    c = grille.gps_vers_cellule(r.lat, r.lon)
    if c is None: continue
    for dx in range(-a.rayon, a.rayon + 1):
        for dy in range(-a.rayon, a.rayon + 1):
            ix, iy = c[0] + dx, c[1] + dy
            if 0 <= ix < grille.nx and 0 <= iy < grille.ny and grille.corridor_mask[ix, iy]:
                cellules.add((ix, iy))
actions = [(ix, iy, g, f) for ix, iy in sorted(cellules) for g, f in CONFIGS if (ix, iy, g, f) not in deja]
print(f"rayon {a.rayon} : {len(cellules)} cellules, {len(actions)} actions candidates")

cache_f = os.path.join(OUT, f"ilp_couverture_cache{'_mock' if a.mock else ''}.pkl")
cache = pickle.load(open(cache_f, "rb")) if os.path.exists(cache_f) else {}
if a.mock:
    PORTEE = {"3G": 0.6, "4G": 0.5, "5G": 0.3}       # NON PHYSIQUE : test de plomberie uniquement
    pts_b = df_points[df_points.point_id.isin(blancs)]
    def couvre(act):
        lat, lon = grille.cellule_vers_gps(act[0], act[1])
        d = 111.0 * np.hypot(pts_b.lat - lat, (pts_b.lon - lon) * np.cos(np.radians(lat)))
        return set(pts_b.point_id[d <= PORTEE[act[2]]])
else:
    from simulateur_deploiement import simuler_deploiement
    def couvre(act):
        lat, lon = grille.cellule_vers_gps(act[0], act[1])
        res = simuler_deploiement(lat, lon, act[2], act[3], df_points, df_meteo)
        return {r["point_id"] for r in res if r["point_id"] in blancs and r["debit_adj_mbps"] >= D_COV}
t0 = time.time()
for i, act in enumerate(actions):
    if act not in cache:
        cache[act] = couvre(act)
        if i % 200 == 0:
            pickle.dump(cache, open(cache_f, "wb")); print(f"  {i}/{len(actions)} simulées, {time.time() - t0:.0f} s", flush=True)
pickle.dump(cache, open(cache_f, "wb"))

couvrables = set().union(*[cache[x] for x in actions]) if actions else set()
non_couvrables = blancs - couvrables
print(f"{len(couvrables)} points couvrables par les candidats, {len(non_couvrables)} non couvrables")

def cout_radio(act): return get_radio_cost(act[2], act[3])

def modele(budget=None):
    """MILP résolu par HiGHS (scipy.optimize.milp). Variables : x (actions), y (nouveaux sites), z (points, P2)."""
    nouv = sorted({(act[0], act[1]) for act in actions} - sites_existants)
    iy = {c: i for i, c in enumerate(nouv)}
    pts = sorted(couvrables); ip = {p: i for i, p in enumerate(pts)}
    nA, nY, nZ = len(actions), len(nouv), (0 if budget is None else len(pts))
    n = nA + nY + nZ
    cout_vec = np.r_[[cout_radio(act) for act in actions], [C_SITE_EUR] * nY, [0.0] * nZ]
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for j, act in enumerate(actions):                       # x_a - y_c <= 0
        c = (act[0], act[1])
        if c in iy:
            rows += [r, r]; cols += [j, nA + iy[c]]; vals += [1, -1]; lo.append(-np.inf); hi.append(0); r += 1
    couvreurs = {p: [] for p in pts}
    for j, act in enumerate(actions):
        for p in cache[act]:
            couvreurs[p].append(j)
    for p in pts:
        if budget is None:                                  # sum x >= 1
            rows += [r] * len(couvreurs[p]); cols += couvreurs[p]; vals += [1] * len(couvreurs[p]); lo.append(1); hi.append(np.inf)
        else:                                               # z_p - sum x <= 0
            rows += [r] * (len(couvreurs[p]) + 1); cols += couvreurs[p] + [nA + nY + ip[p]]
            vals += [-1] * len(couvreurs[p]) + [1]; lo.append(-np.inf); hi.append(0)
        r += 1
    if budget is not None:                                  # coût <= B
        rows += [r] * (nA + nY); cols += list(range(nA + nY)); vals += list(cout_vec[:nA + nY]); lo.append(-np.inf); hi.append(budget); r += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n))
    obj = cout_vec if budget is None else np.r_[cout_vec[:nA + nY] * 1e-9, [-1.0] * nZ]   # départage : moins cher à couverture égale
    res = milp(obj, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": a.temps_max_s, "mip_rel_gap": 0.0})
    if res.x is None:
        return res.message, None, [], res
    choisies = [act for j, act in enumerate(actions) if res.x[j] > 0.5]
    statut = "Optimal" if res.status == 0 else f"Non prouvé ({res.message})"
    return statut, float(cout_vec @ np.round(res.x)), choisies, res

statut, cout, choisies, m = modele()
n_sites = len({(c[0], c[1]) for c in choisies} - sites_existants)
print(f"(P1) statut {statut} : coût minimal {cout:,.0f} €, {len(choisies)} déploiements, {n_sites} nouveaux sites, "
      f"zones blanches restantes = {len(non_couvrables)} (non couvrables par les candidats)")
pd.DataFrame(choisies, columns=["ix", "iy", "generation", "bande_mhz"]).to_csv(os.path.join(OUT, f"ilp_set_cover{'_mock' if a.mock else ''}.csv"), index=False)
lignes = []
for B in [float(v) * 1e6 for v in a.budgets.split(",")]:
    st, c, ch, mm = modele(B)
    nb = len(set().union(*[cache[x] for x in ch])) if ch else 0
    lignes.append({"budget_eur": B, "statut": st, "cout_eur": c, "points_couverts": nb,
                   "n_white": len(blancs) - nb, "deploiements": len(ch)})
    print(lignes[-1], flush=True)
pd.DataFrame(lignes).to_csv(os.path.join(OUT, f"ilp_couverture_budget{'_mock' if a.mock else ''}.csv"), index=False)
print("Statut 'Optimal' = optimum prouvé sur les candidats (écart relatif nul) ; sinon meilleure solution trouvée, pas un optimum.")
