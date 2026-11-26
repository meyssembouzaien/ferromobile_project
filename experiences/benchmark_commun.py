"""
benchmark_commun.py : toutes les méthodes, sur les mêmes configurations, avec le même réglage (reglage_commun.py).

Pour chaque configuration (réseau complet, un secteur retiré, un opérateur retiré) :
  Existing network   aucun déploiement
  Random             3 graines, actions tirées parmi les actions permises
  Coverage-Greedy    plus de points blancs couverts par euro (couverture exacte P.1812)
  Trajectory-Greedy  plus grand gain de U par euro (U de l'environnement, mêmes poids que l'agent)
  ILP                coût minimal pour couvrir tous les points blancs couvrables (HiGHS)
  agents             checkpoints donnés par --agent (3 graines)

Règles communes :
  - mêmes actions permises pour toutes les méthodes : masque de l'environnement (budget, antenne déjà
    déployée, portée + LOS vers un point blanc restant) ;
  - même budget, même limite de temps TEMPS_MAX_S par méthode et par configuration ;
  - chaque plan est rejoué dans le même environnement et mesuré par les mêmes critères ;
  - tout tourne dans le même processus, sur la même machine, cache des simulations vidé avant chaque méthode ; le temps de calcul de chaque méthode est mesuré
    (le temps d'entraînement de l'agent est compté à part, dans son train_log.csv).

Lancer (depuis experiences/, avec le DEM et crc-covlib) :
  python benchmark_commun.py --agent "Dueling=seed0.pt" --agent "Dueling=seed1.pt" --agent "Dueling=seed2.pt"
  python benchmark_commun.py --agent "Dueling cv0=chemin.pt@sans_secteur_0"     (agent testé sur une seule configuration)
Sorties : <out>/benchmark_commun.csv (une ligne par méthode, configuration et graine) et benchmark_resume.csv
"""
import argparse
import heapq
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, os.path.join(RACINE, "placement"))
sys.path.insert(0, ICI)
os.environ.setdefault("CRC_COVLIB_PATH", os.path.join(RACINE, "crc-covlib", "python-wrapper"))

import reglage_commun as R                                       # noqa: E402
from cost_model import C_SITE_EUR, get_radio_cost, get_total_cost   # noqa: E402
from environment_rl import FerroMobileEnv, STOP, _CACHE_SIMULATION   # noqa: E402


def nouvel_env(config):
    env = FerroMobileEnv(**R.options_env(**R.CONFIGURATIONS[config]))
    env.reset()
    return env


# ---------------------------------------------------------------- mêmes critères pour tous
def mesurer(env):
    """Critères du plan rejoué (DP de l'environnement) ; un point blanc compte pour une qualité 0."""
    dp = env._dernier_resultat_dp
    points, cells = env._fusionner_cells_par_point()
    qos = {}
    for pid, cand, cel in zip(points, cells, dp["trajectory"]):
        qos[pid] = next(c["qos"] for c in cand if c["cellule"] == cel)
    blancs = set(env._points_blancs)
    hors = sorted(env.points_hors_tunnel)
    q = np.array([0.0 if (p in blancs or p not in qos) else qos[p] for p in hors])
    est_blanc = [p in blancs for p in hors]
    n_cut = sum(1 for i, b in enumerate(est_blanc) if b and (i == 0 or not est_blanc[i - 1]))
    return {"n_white": len(blancs), "q_mean": q.mean(), "q_min": q.min(), "N_H": dp["N_H"], "N_O": dp["N_O"],
            "N_T": dp["N_T"], "N_cut": n_cut}


def rejouer(config, plan):
    """Rejoue le plan dans un environnement neuf de la configuration, puis STOP."""
    env = nouvel_env(config)
    n_avant, U0 = env._dernier_n_white, env.U0_reference
    done, info, n = False, {}, 0
    for a in plan:
        if done:
            break
        _, _, done, info = env.step(a)
        n += 1
    if not done:
        _, _, done, info = env.step(STOP)
    nouveaux = len({tuple(a[:2]) for a in plan[:n]} - SITES_INITIAUX[config])
    return {"n_white_avant": n_avant, **mesurer(env), "cout_eur": info.get("cout_total", 0.0),
            "deploiements": n, "nouveaux_sites": nouveaux,
            "return_ref": env.U_reference() - U0 + info.get("r_fin", 0.0)}


SITES_INITIAUX = {}


# ---------------------------------------------------------------- couverture (même masque que l'agent)
def couverture_superset(env):
    """Action -> points blancs qu'elle pourrait couvrir (portée de la bande + LOS), actions permises seulement."""
    c = env._los
    blancs = [p for p in env._points_blancs if p in c["index_point"]]
    cols = np.array([c["index_point"][p] for p in blancs], dtype=int)
    permises = env._actions_faisables()
    out = {}
    if not len(cols):
        return out
    dist = c["dist"][:, cols]
    for (g, portee), acts in c["groupes"].items():
        m = ((dist <= portee) & c["los"][g][:, cols])[c["ligne"][acts]]
        for i, ligne in zip(acts, m):
            if permises[i + 1] and ligne.any():
                out[i + 1] = {blancs[j] for j in np.flatnonzero(ligne)}
    return out


class Exacte:
    """Couverture exacte (P.1812, seuil D_COV) d'une action, mise en cache."""

    def __init__(self, env):
        self.env, self.cache = env, {}

    def __call__(self, i):
        if i not in self.cache:
            a = self.env.actions[i]
            lat, lon = self.env.grille.cellule_vers_gps(a[0], a[1])
            self.cache[i] = {r["point_id"] for r in self.env._simuler(a, lat, lon)
                             if r["debit_adj_mbps"] >= R.D_COV_MBPS}
        return self.cache[i]


def cout_action(env, i, sites):
    a = env.actions[i]
    return get_total_cost((a[0], a[1]), a[2], a[3], sites)


# ---------------------------------------------------------------- méthodes
def aleatoire(config, graine):
    env, rng, plan = nouvel_env(config), np.random.default_rng(graine), []
    done = False
    while not done:
        a = env.actions[int(rng.choice(np.flatnonzero(env._actions_faisables())))]
        _, _, done, _ = env.step(a)
        if a != STOP:
            plan.append(a)
    return plan, {}


def coverage_greedy(config, debut):
    env = nouvel_env(config)
    sup, exacte = couverture_superset(env), Exacte(env)
    blancs = set(env._points_blancs)
    reste, sites, budget, plan = set(blancs), set(env.sites_deja_presents), R.ENV["budget"], []

    def cle(i):
        cov = exacte.cache.get(i, sup[i])
        return len(cov & reste) / cout_action(env, i, sites)

    tas = [(-cle(i), i) for i in sup]
    heapq.heapify(tas)
    pris = set()
    while reste and tas and time.time() - debut < R.TEMPS_MAX_S:
        _, i = heapq.heappop(tas)
        if i in pris or cout_action(env, i, sites) > budget:
            continue
        exacte(i)
        v = cle(i)
        if v <= 0:
            continue
        if tas and v < -tas[0][0] - 1e-12:      # clé périmée : remise dans le tas
            heapq.heappush(tas, (-v, i))
            continue
        a = env.actions[i]
        budget -= cout_action(env, i, sites)
        plan.append(a); pris.add(i); reste -= exacte(i)
        if (a[0], a[1]) not in sites:           # le site existe désormais : les autres bandes coûtent moins
            sites.add((a[0], a[1]))
            for j in sup:
                if j not in pris and env.actions[j][:2] == a[:2]:
                    heapq.heappush(tas, (-cle(j), j))
    return plan, {"arret": "temps" if reste and time.time() - debut >= R.TEMPS_MAX_S else "fin"}


def gain_U(env, a, cache):
    """U(I + a) - U(I) sans modifier l'environnement."""
    if a not in cache:
        lat, lon = env.grille.cellule_vers_gps(a[0], a[1])
        cache[a] = [r for r in env._simuler(a, lat, lon) if r["point_id"] in env.points_hors_tunnel]
    n = len(env._resultats_hypo)
    env._resultats_hypo.extend(cache[a])
    dp, n_white, n_eval = env._evaluer_dp_courant()
    u = env._calculer_U(dp, n_eval, n_white)
    del env._resultats_hypo[n:]
    return u - env.U_courant


def trajectory_greedy(config, debut):
    """À chaque étape : présélection des K actions qui couvrent le plus de points blancs par euro,
    puis l'action de plus grand gain de U par euro parmi elles. STOP si aucun gain positif."""
    env = nouvel_env(config)
    sup, cache, plan, done = couverture_superset(env), {}, [], False
    arret = "fin"
    while not done:
        if time.time() - debut >= R.TEMPS_MAX_S:
            arret = "temps"
            break
        permises = env._actions_faisables()
        blancs = set(env._points_blancs)
        sites = env.sites_deja_presents
        score = {i: len(s & blancs) / cout_action(env, i, sites) for i, s in sup.items() if permises[i]}
        liste = [i for i in sorted(score, key=score.get, reverse=True) if score[i] > 0][:R.K_PRESELECTION]
        meilleur, ratio = None, 0.0
        for i in liste:
            r = gain_U(env, env.actions[i], cache) / cout_action(env, i, sites)
            if r > ratio:
                meilleur, ratio = env.actions[i], r
        if meilleur is None:
            break
        _, _, done, _ = env.step(meilleur)
        plan.append(meilleur)
    return plan, {"arret": arret}


def ilp(config, debut):
    """Couverture de coût minimal (sites + radios), contrôle exact paresseux des actions choisies."""
    env = nouvel_env(config)
    sup, exacte = couverture_superset(env), Exacte(env)
    sites0 = set(env.sites_deja_presents)
    it, statut, gap = 0, "", None
    while True:
        it += 1
        cov = {i: exacte.cache.get(i, s) & set(env._points_blancs) for i, s in sup.items()}
        acts = [i for i in cov if cov[i]]
        cellules = sorted({env.actions[i][:2] for i in acts} - sites0)
        iy = {c: k for k, c in enumerate(cellules)}
        pts = sorted(set().union(*[cov[i] for i in acts]))
        nA, nY = len(acts), len(cellules)
        c = np.r_[[get_radio_cost(*env.actions[i][2:]) for i in acts], [C_SITE_EUR] * nY]
        rows, cols, vals, lo, r = [], [], [], [], 0
        for j, i in enumerate(acts):
            if env.actions[i][:2] in iy:
                rows += [r, r]; cols += [j, nA + iy[env.actions[i][:2]]]; vals += [1, -1]; lo.append(-np.inf); r += 1
        couvreurs = {p: [] for p in pts}
        for j, i in enumerate(acts):
            for p in cov[i]:
                couvreurs[p].append(j)
        for p in pts:
            rows += [r] * len(couvreurs[p]); cols += couvreurs[p]; vals += [1] * len(couvreurs[p]); lo.append(1); r += 1
        hi = [0 if v == -np.inf else np.inf for v in lo]
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nA + nY))
        reste = max(1.0, R.TEMPS_MAX_S - (time.time() - debut))
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nA + nY), bounds=Bounds(0, 1),
                   options={"time_limit": reste, "mip_rel_gap": 0.0})
        if res.x is None:
            return [], {"arret": f"échec : {res.message}", "iterations": it}
        statut, gap = ("optimal" if res.status == 0 else "limite de temps"), getattr(res, "mip_gap", None)
        choisies = [i for j, i in enumerate(acts) if res.x[j] > 0.5]
        a_verifier = [i for i in choisies if i not in exacte.cache]
        if not a_verifier or time.time() - debut >= R.TEMPS_MAX_S:
            break
        for i in a_verifier:
            exacte(i)
    plan = [env.actions[i] for i in choisies]
    return plan, {"arret": statut, "gap_milp": gap, "iterations": it, "points_couvrables": len(pts)}


def agent(config, chemin):
    import torch
    import train_dqn as T
    from evaluate_policy import charger_politique
    from environment_rl import BANDES_PAR_GENERATION
    device = torch.device("cpu")                # même machine que les autres méthodes
    etat, classe = charger_politique(chemin, device)
    cfg = etat.get("config") or {}
    modele = classe(sum(len(v) for v in BANDES_PAR_GENERATION.values()), nouvel_env(config).grille.corridor_mask,
                    n_canaux=T.N_CANAUX_ETAT + (3 if cfg.get("canal_couverture") else 0),
                    dilatations=T.DILATATIONS + ((32,) if cfg.get("vue_large") else ()))
    poids = dict(etat["modele"])
    poids["indices_actions"] = modele.indices_actions
    modele.load_state_dict(poids)
    modele.eval()
    env = nouvel_env(config)
    _, trace = T.episode_glouton(env, modele, device)
    import ast
    plan = [ast.literal_eval(t["action"]) for t in trace if t["action"] != STOP]
    return plan, {"episode": etat["episode"]}


# ---------------------------------------------------------------- boucle
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default=",".join(R.CONFIGURATIONS))
    p.add_argument("--methodes", default="existant,aleatoire,cg,tg,ilp")
    p.add_argument("--agent", action="append", default=[], help="NOM=CHEMIN.pt ou NOM=CHEMIN.pt@config1,config2")
    p.add_argument("--ligne", default="ligne1")
    p.add_argument("--out", default=os.path.join(RACINE, "resultats", "benchmark_commun"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fichier = os.path.join(a.out, "benchmark_commun.csv")
    lignes = pd.read_csv(fichier).to_dict("records") if os.path.exists(fichier) else []
    faits = {(l["configuration"], l["methode"], l["graine"]) for l in lignes}
    methodes = a.methodes.split(",")
    agents, vus = [], {}
    for s in a.agent:      # même nom répété = graines 0, 1, 2 du même agent
        nom, reste = s.split("=", 1)
        chemin, _, cfgs = reste.partition("@")
        vus[nom] = vus.get(nom, -1) + 1
        agents.append((nom, vus[nom], chemin, cfgs.split(",") if cfgs else None))
    print(f"Réglage : {R.ENV}, Q_critique {R.Q_CRITIQUE}, graines {R.GRAINES}, temps max {R.TEMPS_MAX_S} s, "
          f"{os.cpu_count()} CPU", flush=True)

    for config in a.configs.split(","):
        SITES_INITIAUX[config] = set(nouvel_env(config).sites_deja_presents)
        taches = []
        if "existant" in methodes:
            taches.append(("Existing network", -1, lambda d, c=config: ([], {})))
        if "aleatoire" in methodes:
            taches += [("Random", g, lambda d, c=config, g=g: aleatoire(c, g)) for g in R.GRAINES]
        if "cg" in methodes:
            taches.append(("Coverage-Greedy", -1, lambda d, c=config: coverage_greedy(c, d)))
        if "tg" in methodes:
            taches.append(("Trajectory-Greedy", -1, lambda d, c=config: trajectory_greedy(c, d)))
        if "ilp" in methodes:
            taches.append(("ILP", -1, lambda d, c=config: ilp(c, d)))
        for nom, graine, chemin, cfgs in agents:
            if cfgs is None or config in cfgs:
                taches.append((nom, graine, lambda d, c=config, ch=chemin: agent(c, ch)))
        for nom, graine, f in taches:
            if (config, nom, graine) in faits:
                continue
            _CACHE_SIMULATION.clear()   # chaque méthode part sans simulations en cache (temps comparables)
            debut = time.time()
            plan, extra = f(debut)
            duree = time.time() - debut
            ligne = {"ligne": a.ligne, "configuration": config, "methode": nom, "graine": graine,
                     **rejouer(config, plan), "temps_calcul_s": round(duree, 1), **extra}
            lignes.append(ligne)
            print(f"{config:16s} {nom:20s} graine {graine:2d} : {ligne['n_white_avant']} -> {ligne['n_white']} blancs, "
                  f"{ligne['cout_eur'] / 1e6:.3f} M€, q {ligne['q_mean']:.3f}, q_min {ligne['q_min']:.3f}, "
                  f"N_H {ligne['N_H']}, N_O {ligne['N_O']}, N_T {ligne['N_T']}, {duree:.0f} s", flush=True)
            pd.DataFrame(lignes).to_csv(fichier, index=False)   # reprise possible après une coupure

    df = pd.DataFrame(lignes)
    crit = ["n_white", "cout_eur", "q_mean", "q_min", "N_H", "N_O", "N_T", "N_cut", "return_ref", "temps_calcul_s"]
    resume = df.groupby(["ligne", "configuration", "methode"])[crit].agg(["mean", "std", "count"])
    resume.to_csv(os.path.join(a.out, "benchmark_resume.csv"))
    print(resume.to_string())


if __name__ == "__main__":
    main()
