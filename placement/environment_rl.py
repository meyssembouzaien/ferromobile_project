"""
environment_rl.py — Environnement RL (§13, §14)
==================================================================
Orchestre les modules déjà validés, n'en réimplémente aucun :
grid_config.py, build_state.py, build_state_dynamic.py, trajectory_dp.py,
simulateur_deploiement.py, cost_model.py, utility.py. Aucun de ces
fichiers n'est modifié pour construire cet environnement.

Trois identités de site, jamais unifiées de force (voir guide de codage,
§6) :
    - action RL / "déjà déployé" : (ix, iy, g, f)
    - coût (§12)                 : (ix, iy)
    - DP / cellule radio (§9)    : GPS exact (x_s, y_s, o, g, f)

deploiements_reels ∪ infrastructure_deployee = combinaisons (site,g,f)
déjà présentes dans I (§13). deploiements_reels est dérivé UNE FOIS à
reset() depuis le réseau réel (par ant_id, pas point par point — un
ant_id identifie une infrastructure radio réelle unique dans le
dataset, §4). Pour l'espace d'action RL, l'opérateur n'est pas une
dimension de décision (§13 : o ∉ a) : les déploiements réels sont donc
délibérément PROJETÉS sur (ix,iy,g,f) sans opérateur. Conséquence
assumée : si deux opérateurs différents ont chacun de la 4G/1800 sur le
même site réel (même cellule de grille), les deux se fusionnent en une
seule entrée dans deploiements_reels — cohérent avec le fait que
l'action RL elle-même ne distingue jamais les opérateurs.

self.actions est restreint aux cellules du corridor (corridor_mask,
§15) dès la construction, pas filtré dynamiquement dans _est_faisable() :
ce sont deux concepts différents (espace d'action vs faisabilité
courante). Les bandes possibles par génération sont dérivées DIRECTEMENT
des clés de TECH_PROFILES (prepare_data.py) — pas une copie maintenue à
la main, pour ne jamais diverger si TECH_PROFILES change.

STOP est représenté par la chaîne "STOP", toujours en première position
de self.actions (action_id = 0).

JOURNALISATION (dataset_enrichi.csv n'est JAMAIS modifié, lecture seule) :
chaque déploiement hypothétique simulé pendant un épisode peut être
journalisé dans un fichier SÉPARÉ, UN PAR run_id (par défaut
data/processed/dataset_enrichi_rl/run_<run_id>.csv), distinct du
dataset réel, pour pouvoir inspecter après coup ce que le RL a
réellement simulé sans jamais risquer de corrompre la version de
référence en cas d'erreur d'entraînement. Un fichier par run (pas un
fichier cumulatif partagé) élimine la condition de concurrence entre
processus qui écriraient en parallèle : deux run_id différents ->
deux fichiers différents -> jamais deux processus qui écrivent le même
en-tête au même endroit en même temps. self._resultats_hypo (en
mémoire) reste la seule source de vérité PENDANT l'épisode — le CSV
n'est qu'une trace persistante écrite en plus, jamais relue pendant
l'épisode. Chaque ligne porte run_id + episode_id (un épisode dans ce
run) + artificielle=True (pour ne jamais confondre avec une mesure
réelle en relisant le fichier des semaines plus tard). Désactivable via
sauvegarder_deploiements_rl=False (écrire un CSV à chaque step a un
coût, à éviter pour un entraînement massif si le log n'est pas
nécessaire).
"""

import os
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from grid_config import construire_grille_depuis_dataset
from build_state import construire_canaux_statiques
from build_state_dynamic import construire_canaux_infrastructure, construire_canaux_performance
from trajectory_dp import trajectory_dp
from simulateur_deploiement import simuler_deploiement
from cost_model import get_total_cost
from utility import normaliser_metriques, compute_U, compute_r_t, compute_r_fin
from prepare_data import TECH_PROFILES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_BASE = os.path.join(BASE_DIR, "data", "processed", "dataset_base.csv")
DATASET_ENRICHI = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi.csv")
DATASET_ENRICHI_RL_DIR = os.path.join(BASE_DIR, "data", "processed", "dataset_enrichi_rl")

D_COV = 1.0
LAMBDA_H, LAMBDA_O, LAMBDA_T = 0.05, 0.05, 0.05

# Bandes possibles par génération, DÉRIVÉES des clés de TECH_PROFILES
# (prepare_data.py) — source unique de vérité, §13 : f in F_g, "parmi
# celles définies pour cette génération dans les profils techniques du
# pipeline". Ne jamais dupliquer cette liste à la main : si TECH_PROFILES
# change, cette dérivation suit automatiquement.
BANDES_PAR_GENERATION = {}
for _gen, _bande in TECH_PROFILES.keys():
    BANDES_PAR_GENERATION.setdefault(_gen, []).append(_bande)
for _gen in BANDES_PAR_GENERATION:
    BANDES_PAR_GENERATION[_gen].sort()

STOP = "STOP"

# Schéma attendu de dataset_enrichi_rl.csv — utilisé pour détecter si un
# fichier préexistant à cet emplacement n'est PAS un journal RL généré
# par ce code (par exemple une copie accidentelle de dataset_enrichi.csv,
# dont le schéma est complètement différent : ant_id, scenario_id,
# operateur... au lieu de run_id, episode_id, artificielle...).
COLONNES_LOG_RL = ["run_id", "episode_id", "t", "scenario", "artificielle",
                    "ix", "iy", "generation", "bande_mhz", "lat_site", "lon_site",
                    "cout_eur", "point_id", "qos", "debit_adj_mbps", "rtt_ms", "ber"]


class FerroMobileEnv:
    def __init__(self, scenario="S1", budget=500_000.0, poids=None,
                 Q_critique=0.5, R_couverture=100.0, R_qualite=50.0,
                 R_succes=20.0, R_eff=20.0, t_max=200,
                 dataset_base_path=DATASET_BASE, dataset_enrichi_path=DATASET_ENRICHI,
                 sauvegarder_deploiements_rl=True, dataset_enrichi_rl_path=None,
                 run_id=None, episode_id=None):
        self.scenario = scenario
        self.budget_total = budget
        self.poids = poids or {"w_Q": 1.0, "w_W": 1.0, "w_H": 1.0, "w_O": 1.0, "w_T": 1.0}
        self.Q_critique = Q_critique
        self.R_couverture = R_couverture
        self.R_qualite = R_qualite
        self.R_succes = R_succes
        self.R_eff = R_eff
        self.t_max = t_max
        self.sauvegarder_deploiements_rl = sauvegarder_deploiements_rl
        # run_id distingue les entraînements entre eux. Le suffixe uuid4
        # évite une collision si deux environnements sont créés dans la
        # même seconde (arrive facilement en lançant plusieurs runs en
        # parallèle ou via un script qui instancie vite plusieurs envs).
        self.run_id = run_id if run_id is not None else f"{datetime.now():%Y-%m-%d_%H%M%S}_{uuid4().hex[:8]}"
        # Un fichier PAR run_id, pas un fichier cumulatif partagé : élimine
        # la condition de concurrence si plusieurs environnements/processus
        # tournent en parallèle (chacun a son propre run_id -> son propre
        # fichier, donc plus jamais deux processus qui écrivent le même
        # en-tête au même endroit en même temps). Un dataset_enrichi_rl_path
        # explicite reste possible (ex. pour forcer un chemin précis dans
        # un test), mais alors la responsabilité d'éviter la concurrence
        # entre processus qui partageraient ce même chemin explicite
        # retombe sur l'appelant.
        self.dataset_enrichi_rl_path = (
            dataset_enrichi_rl_path if dataset_enrichi_rl_path is not None
            else os.path.join(DATASET_ENRICHI_RL_DIR, f"run_{self.run_id}.csv"))
        self.episode_id = episode_id if episode_id is not None else 0

        self.df_base = pd.read_csv(dataset_base_path, low_memory=False)
        self.df_enrichi_complet = pd.read_csv(dataset_enrichi_path, low_memory=False)

        self.grille = construire_grille_depuis_dataset(self.df_base)
        self.df_points = (self.df_base[["point_id", "lat", "lon", "altitude_m", "vegetation", "in_tunnel"]]
                           .drop_duplicates("point_id"))
        self.points_hors_tunnel = set(self.df_points[~self.df_points["in_tunnel"]]["point_id"])
        self.n_total = len(self.points_hors_tunnel)

        # Espace d'action restreint au corridor (§15), construit une seule
        # fois — indépendant de l'épisode.
        self.actions = [STOP] + [
            (int(ix), int(iy), g, f)
            for ix in range(self.grille.nx) for iy in range(self.grille.ny)
            if self.grille.corridor_mask[ix, iy]
            for g in BANDES_PAR_GENERATION
            for f in BANDES_PAR_GENERATION[g]
        ]
        self.action_index = {a: i for i, a in enumerate(self.actions)}

        self._reset_appele = False

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self):
        if self.sauvegarder_deploiements_rl and not self._reset_appele:
            self._verifier_schema_log_rl()

        if self._reset_appele:
            self.episode_id += 1
        df_meteo = (self.df_enrichi_complet[self.df_enrichi_complet["scenario_id"] == self.scenario]
                    [["point_id", "pluie_mm_h", "temp_c"]].drop_duplicates("point_id"))
        self.canaux_statiques = construire_canaux_statiques(self.grille, self.df_points, df_meteo)

        df_scenario = self.df_enrichi_complet[self.df_enrichi_complet["scenario_id"] == self.scenario]
        self.df_reel = df_scenario[(~df_scenario["in_tunnel"]) & (df_scenario["ant_id"].notna())]

        # cells réelles par point (site en GPS exact, comme trajectory_dp.py
        # l'a toujours attendu — non modifié).
        self._cells_reelles_par_point = {}
        for pid, lignes in self.df_reel.groupby("point_id"):
            self._cells_reelles_par_point[pid] = [
                {"cellule": ((round(row.ant_lat, 6), round(row.ant_lon, 6)),
                             row.operateur, row.generation, row.bande_mhz),
                 "qos": row.qos, "debit": row.debit_adj_mbps}
                for row in lignes.itertuples()
            ]
        self._points_reels = sorted(self._cells_reelles_par_point.keys())
        self.debit_reel_par_point = df_scenario[~df_scenario["in_tunnel"]].groupby("point_id")["debit_adj_mbps"].max()

        # Ajouts hypothétiques de l'épisode, vide au départ.
        self._resultats_hypo = []  # liste de dicts {point_id, cellule, qos, debit_adj_mbps}

        # deploiements_reels : une seule fois par ant_id, PAS point par point.
        antennes_uniques = self.df_reel[["ant_id", "ant_lat", "ant_lon", "generation", "bande_mhz"]].drop_duplicates("ant_id")
        self.deploiements_reels = set()
        self.sites_deja_presents = set()
        for row in antennes_uniques.itertuples():
            c = self.grille.gps_vers_cellule(row.ant_lat, row.ant_lon)
            if c is None:
                continue  # site réel hors grille : ne devrait jamais arriver (0/29 vérifié)
            self.deploiements_reels.add((c[0], c[1], row.generation, row.bande_mhz))
            self.sites_deja_presents.add(c)

        self.infrastructure_deployee = set()
        self.budget_restant = self.budget_total
        self.t = 0
        self.done = False

        # U(I_0), calculé une fois pour servir de référence au premier r_t.
        resultat_dp, n_white, n_eval = self._evaluer_dp_courant()
        self._dernier_resultat_dp = resultat_dp
        self._dernier_n_white = n_white
        self._dernier_n_eval = n_eval
        self.U_courant = self._calculer_U(resultat_dp, n_eval, n_white)

        self._reset_appele = True
        return self._construire_etat()

    # ------------------------------------------------------------------
    # ÉVALUATION (DP + N_white), factorisée pour reset() et step()
    # ------------------------------------------------------------------
    def _fusionner_cells_par_point(self):
        """Même logique que ajouter_cellules_hypothetiques() de
        valider_chaine_complete.py — un point qui n'avait aucun candidat
        réel et reçoit une cellule hypothétique est AJOUTÉ, pas seulement
        complété."""
        cellules_hypo_par_point = {}
        for r in self._resultats_hypo:
            cellules_hypo_par_point.setdefault(r["point_id"], []).append(r)

        tous_les_pid = sorted(set(self._points_reels) | set(cellules_hypo_par_point.keys()))
        points, cells_par_point = [], []
        for pid in tous_les_pid:
            candidats = list(self._cells_reelles_par_point.get(pid, []))
            candidats += [{"cellule": r["cellule"], "qos": r["qos"], "debit": r["debit_adj_mbps"]}
                          for r in cellules_hypo_par_point.get(pid, [])]
            if candidats:
                points.append(pid)
                cells_par_point.append(candidats)
        return points, cells_par_point

    @staticmethod
    def _serie_max_par_point(resultats, cle):
        """Construit une Series point_id -> max(valeur) à partir d'une
        liste de dicts, SANS passer par un dict intermédiaire {point_id: v}
        qui écraserait silencieusement les doublons (plusieurs cellules
        hypothétiques peuvent couvrir le même point_id) avant qu'un
        éventuel groupby ait la moindre chance d'agir. Toujours construire
        la Series à partir de listes (valeurs, index) séparées, puis
        grouper — jamais l'inverse."""
        if not resultats:
            return pd.Series(dtype=float)
        return pd.Series(
            [r[cle] for r in resultats],
            index=[r["point_id"] for r in resultats],
        ).groupby(level=0).max()

    def _evaluer_dp_courant(self):
        points, cells_par_point = self._fusionner_cells_par_point()
        resultat_dp = trajectory_dp(points, cells_par_point, LAMBDA_H, LAMBDA_O, LAMBDA_T)

        # N_white : meilleur débit (réel + hypothétique) sur TOUT le
        # corridor (N_total), pas seulement les points évalués par la DP
        # (N_eval) — §11.
        if self._resultats_hypo:
            debit_hypo = self._serie_max_par_point(self._resultats_hypo, "debit_adj_mbps")
            debit_combine = pd.concat([self.debit_reel_par_point, debit_hypo], axis=1).max(axis=1)
        else:
            debit_combine = self.debit_reel_par_point

        debit_sur_corridor = debit_combine.reindex(list(self.points_hors_tunnel))
        n_white = int((debit_sur_corridor.fillna(0) < D_COV).sum())
        n_eval = len(points)
        return resultat_dp, n_white, n_eval

    def _calculer_U(self, resultat_dp, n_eval, n_white):
        metriques = normaliser_metriques(resultat_dp, n_eval, n_white, self.n_total)
        return compute_U(metriques, self.poids["w_Q"], self.poids["w_W"],
                          self.poids["w_H"], self.poids["w_O"], self.poids["w_T"])

    # ------------------------------------------------------------------
    # FAISABILITÉ
    # ------------------------------------------------------------------
    def _est_faisable(self, action):
        if action == STOP:
            return True
        ix, iy, g, f = action
        if (ix, iy, g, f) in self.deploiements_reels or (ix, iy, g, f) in self.infrastructure_deployee:
            return False
        cout = get_total_cost((ix, iy), g, f, self.sites_deja_presents)
        return cout <= self.budget_restant

    def _actions_faisables(self):
        return np.array([self._est_faisable(a) for a in self.actions], dtype=bool)

    # ------------------------------------------------------------------
    # ÉTAT
    # ------------------------------------------------------------------
    def _construire_etat(self):
        canaux_infra = construire_canaux_infrastructure(
            self.grille, self.df_reel, cellules_hypothetiques=self._resultats_hypo_pour_infra())

        if self._resultats_hypo:
            debit_hypo = self._serie_max_par_point(self._resultats_hypo, "debit_adj_mbps")
            qos_hypo = self._serie_max_par_point(self._resultats_hypo, "qos")
            debit_combine = pd.concat([self.debit_reel_par_point, debit_hypo], axis=1).max(axis=1)
        else:
            debit_combine = self.debit_reel_par_point

        meilleur_par_point = pd.DataFrame({"point_id": debit_combine.index, "meilleur_debit": debit_combine.values})
        # Canal QoS : le maximum des QoS disponibles (réelles et
        # hypothétiques) à ce point, indépendamment de quelle cellule a le
        # meilleur débit — ce n'est PAS "la qos de la cellule au meilleur
        # débit" (ce serait une agrégation différente, plus fine, pas
        # celle utilisée ici). Approximation simple, cohérente avec
        # l'agrégation déjà faite dans build_state_dynamic.py.
        #
        # IMPORTANT — ce canal représente le potentiel radio LOCAL maximal
        # à chaque point (max sur les cellules disponibles), PAS le
        # résultat de la trajectoire globalement optimale que sélectionne
        # la DP (qui tient compte des pénalités de transition N_H/N_O/N_T,
        # §10-11). Les deux quantités peuvent légitimement différer : un
        # point peut avoir un excellent potentiel local (ce canal) tout en
        # étant desservi par une cellule différente dans la trajectoire
        # réellement choisie par la DP, si changer de cellule coûterait
        # plus cher en pénalité de transition que le gain de QoS. La DP
        # reste la SEULE source utilisée pour l'évaluation séquentielle
        # (qos_mean, qos_min, N_H/N_O/N_T) et le calcul de U(I)/r_fin — ce
        # canal ne nourrit que l'état visuel donné au CNN, jamais U(I).
        qos_reelle = self.df_reel.groupby("point_id")["qos"].max()
        if self._resultats_hypo:
            meilleure_qos = pd.concat([qos_reelle, qos_hypo], axis=1).max(axis=1)
        else:
            meilleure_qos = qos_reelle
        meilleur_par_point = meilleur_par_point.merge(
            meilleure_qos.rename("meilleur_qos"), left_on="point_id", right_index=True, how="left")

        canaux_perf = construire_canaux_performance(self.grille, self.df_points, meilleur_par_point)

        etat_spatial = np.concatenate([self.canaux_statiques, canaux_infra, canaux_perf], axis=0)
        b_t = self.budget_restant / self.budget_total
        progression = self.t / self.t_max
        return {"grille": etat_spatial, "b_t": b_t, "progression": progression}

    def _resultats_hypo_pour_infra(self):
        """build_state_dynamic.construire_canaux_infrastructure attend des
        cellules hypothétiques sous forme {"cellule": (site,o,g,f)} où
        site=(lat,lon) — déjà le format de self._resultats_hypo, aucune
        conversion nécessaire."""
        return self._resultats_hypo

    # ------------------------------------------------------------------
    # JOURNALISATION — fichier SÉPARÉ, dataset_enrichi.csv n'est jamais
    # touché (lu une seule fois, en lecture seule, à __init__). En cas
    # d'erreur d'entraînement, on peut inspecter exactement ce qui a été
    # simulé sans jamais risquer de corrompre la version de référence.
    # ------------------------------------------------------------------
    def _verifier_schema_log_rl(self):
        """Échoue vite et clairement si dataset_enrichi_rl_path pointe déjà
        vers un fichier qui n'a pas le schéma attendu — typiquement une
        copie accidentelle de dataset_enrichi.csv. Sans ce garde-fou,
        _journaliser() ajouterait des lignes au format RL sous l'en-tête
        d'un fichier au format réseau réel, corrompant silencieusement le
        CSV (colonnes désalignées entre les lignes)."""
        if not os.path.exists(self.dataset_enrichi_rl_path):
            return  # le fichier sera créé proprement au premier step()
        try:
            colonnes_existantes = set(pd.read_csv(self.dataset_enrichi_rl_path, nrows=0).columns)
        except Exception as e:
            raise RuntimeError(f"Impossible de lire '{self.dataset_enrichi_rl_path}' : {e}")
        if colonnes_existantes != set(COLONNES_LOG_RL):
            raise RuntimeError(
                f"'{self.dataset_enrichi_rl_path}' existe déjà mais n'a pas le "
                f"schéma attendu du journal RL.\n"
                f"Attendu : {sorted(COLONNES_LOG_RL)}\n"
                f"Trouvé  : {sorted(colonnes_existantes)}\n"
                "Ce fichier n'est probablement pas un journal RL généré par "
                "ce code (par exemple une copie de dataset_enrichi.csv). "
                "Supprime-le ou choisis un autre dataset_enrichi_rl_path — "
                "ne jamais faire pointer ce chemin vers une copie du réseau "
                "réel : le schéma est incompatible et l'ajout de lignes "
                "corromprait le fichier."
            )

    def _journaliser(self, action, lat_site, lon_site, cout, resultats_sim):
        # Revérifié à chaque écriture, pas seulement au premier reset() :
        # le fichier pourrait être remplacé entre deux épisodes (process
        # externe, run concurrent...). Coût faible (nrows=0, juste l'en-tête).
        self._verifier_schema_log_rl()

        ix, iy, g, f = action
        base = {
            "run_id": self.run_id, "episode_id": self.episode_id, "t": self.t,
            "scenario": self.scenario, "artificielle": True,
            "ix": ix, "iy": iy, "generation": g, "bande_mhz": f,
            "lat_site": lat_site, "lon_site": lon_site, "cout_eur": cout,
        }
        if resultats_sim:
            lignes = [{**base, "point_id": r["point_id"], "qos": r["qos"],
                       "debit_adj_mbps": r["debit_adj_mbps"],
                       "rtt_ms": r.get("rtt_ms"), "ber": r.get("ber")}
                      for r in resultats_sim]
        else:
            # Déploiement faisable mais qui ne couvre aucun point (portée/LOS) :
            # journalisé quand même, avec point_id=None, pour ne pas perdre la
            # trace de l'action elle-même (coût engagé pour rien).
            lignes = [{**base, "point_id": None, "qos": None, "debit_adj_mbps": None,
                       "rtt_ms": None, "ber": None}]
        df_log = pd.DataFrame(lignes)
        ecrire_entete = not os.path.exists(self.dataset_enrichi_rl_path)
        dossier = os.path.dirname(self.dataset_enrichi_rl_path)
        if dossier:
            os.makedirs(dossier, exist_ok=True)
        df_log.to_csv(self.dataset_enrichi_rl_path, mode="a", header=ecrire_entete, index=False)

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, action):
        if self.done:
            raise RuntimeError("step() appelé après done=True — appeler reset() d'abord.")
        if not self._est_faisable(action):
            raise ValueError(f"Action infaisable : {action}. "
                              "Le masque de _actions_faisables() doit être appliqué avant argmax.")

        if action == STOP:
            return self._step_stop()

        ix, iy, g, f = action
        lat, lon = self.grille.cellule_vers_gps(ix, iy)

        resultats_sim = simuler_deploiement(lat, lon, g, f, self.df_points)
        resultats_sim = [r for r in resultats_sim if r["point_id"] in self.points_hors_tunnel]

        cout = get_total_cost((ix, iy), g, f, self.sites_deja_presents)

        if self.sauvegarder_deploiements_rl:
            self._journaliser(action, lat, lon, cout, resultats_sim)

        self._resultats_hypo.extend(resultats_sim)
        self.infrastructure_deployee.add((ix, iy, g, f))
        self.sites_deja_presents.add((ix, iy))  # add() sur un set déjà présent : pas de second C_site
        self.budget_restant -= cout

        resultat_dp, n_white, n_eval = self._evaluer_dp_courant()
        U_apres = self._calculer_U(resultat_dp, n_eval, n_white)
        r_t = compute_r_t(U_apres, self.U_courant)

        self.U_courant = U_apres
        self._dernier_resultat_dp = resultat_dp
        self._dernier_n_white = n_white
        self._dernier_n_eval = n_eval
        self.t += 1

        # Fin forcée si plus aucune action de déploiement n'est faisable
        # (budget épuisé, §14 : "STOP, ou budget épuisé") ou si t_max
        # atteint (garde-fou d'implémentation, pas une contrainte du
        # cahier des charges — évite un épisode infini).
        aucune_action_deploiement_possible = not any(
            self._est_faisable(a) for a in self.actions if a != STOP)
        if aucune_action_deploiement_possible or self.t >= self.t_max:
            return self._finaliser(r_t)

        return self._construire_etat(), r_t, False, {"cout": cout, "n_white": n_white}

    def _step_stop(self):
        return self._finaliser(0.0)

    def _finaliser(self, r_t_dernier_step):
        # Le coût total engagé est directement le budget consommé : il est
        # décrémenté correctement à CHAQUE step() dans l'ordre réel des
        # décisions (get_total_cost dépend de l'ordre d'ajout pour C_site,
        # donc le recalculer hors-ligne sans cet historique serait faux).
        # self.budget_restant est la seule source de vérité pour Cost(I_T).
        cout_total = self.budget_total - self.budget_restant

        r_fin = compute_r_fin(
            self._dernier_n_white, self._dernier_resultat_dp["qos_min"], cout_total,
            self.budget_total, self.Q_critique, self.R_couverture, self.R_qualite,
            self.R_succes, self.R_eff)

        self.done = True
        return self._construire_etat(), r_t_dernier_step + r_fin, True, {
            "cout_total": cout_total, "n_white": self._dernier_n_white, "r_fin": r_fin}
