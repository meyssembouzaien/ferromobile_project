"""
environment_rl.py : environnement RL de FerroMobile (§13, §14 du cahier des charges).

Orchestre les modules déjà validés, sans les réimplémenter :
grid_config, build_state, build_state_dynamic, trajectory_dp,
simulateur_deploiement, cost_model, utility.

Trois identités de site, jamais fusionnées :
    - action RL / déjà déployé : (ix, iy, g, f)
    - coût (§12)               : (ix, iy)
    - DP / cellule radio (§9)  : GPS exact (x_s, y_s, o, g, f)

STOP = action d'indice 0.
dataset_enrichi.csv n'est jamais modifié. Les déploiements simulés
peuvent être journalisés dans un CSV séparé, un fichier par run_id.

Corrections documentées :
    - compute_r_fin() appelé en arguments nommés (un appel positionnel
      avait déjà décalé les arguments sans erreur).
    - reset() : itertuples() limité aux 8 colonnes utiles (61 colonnes
      rendaient reset() très lent). Résultat identique.
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

# 2G exclu : son débit max (0,24 Mbps) n'atteint jamais D_cov
GENERATIONS_EXCLUES = {"2G"}

# Bandes par génération, dérivées de TECH_PROFILES (source unique)
BANDES_PAR_GENERATION = {}
for _gen, _bande in TECH_PROFILES.keys():
    if _gen in GENERATIONS_EXCLUES:
        continue
    BANDES_PAR_GENERATION.setdefault(_gen, []).append(_bande)
for _gen in BANDES_PAR_GENERATION:
    BANDES_PAR_GENERATION[_gen].sort()

STOP = "STOP"

# Colonnes attendues du journal RL (protège contre un mauvais fichier)
COLONNES_LOG_RL = ["run_id", "episode_id", "t", "scenario", "artificielle",
                    "ix", "iy", "generation", "bande_mhz", "lat_site", "lon_site",
                    "cout_eur", "point_id", "qos", "debit_adj_mbps", "rtt_ms", "ber"]

# Colonnes utilisées pour les cellules réelles de la DP
COLONNES_CELLULES = ["point_id", "ant_lat", "ant_lon", "operateur", "generation",
                     "bande_mhz", "qos", "debit_adj_mbps"]


class FerroMobileEnv:
    # R_* rééchelonnés (100/50/20/20 -> 1/0.5/0.2/0.2) pour rester du même
    # ordre que la somme des r_t (environ 0,1 par épisode). Hiérarchie gardée.
    def __init__(self, scenario="S1", budget=500_000.0, poids=None,
                 Q_critique=0.5, R_couverture=1.0, R_qualite=0.5,
                 R_succes=0.2, R_eff=0.2, t_max=200,
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

        # Suffixe uuid : évite deux run_id identiques créés dans la même seconde
        self.run_id = run_id if run_id is not None else f"{datetime.now():%Y-%m-%d_%H%M%S}_{uuid4().hex[:8]}"
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

        # Espace d'action : cellules du corridor x (g, f), construit une fois
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
        self.df_meteo_scenario = (
            self.df_enrichi_complet[self.df_enrichi_complet["scenario_id"] == self.scenario]
            [["point_id", "pluie_mm_h", "temp_c"]].drop_duplicates("point_id"))
        self.canaux_statiques = construire_canaux_statiques(self.grille, self.df_points, self.df_meteo_scenario)

        df_scenario = self.df_enrichi_complet[self.df_enrichi_complet["scenario_id"] == self.scenario]
        self.df_reel = df_scenario[(~df_scenario["in_tunnel"]) & (df_scenario["ant_id"].notna())]

        # Cellules réelles par point. CORRECTION : 8 colonnes au lieu de 61 (vitesse)
        self._cells_reelles_par_point = {}
        for pid, lignes in self.df_reel[COLONNES_CELLULES].groupby("point_id"):
            self._cells_reelles_par_point[pid] = [
                {"cellule": ((round(row.ant_lat, 6), round(row.ant_lon, 6)),
                             row.operateur, row.generation, row.bande_mhz),
                 "qos": row.qos, "debit": row.debit_adj_mbps}
                for row in lignes.itertuples()
            ]
        self._points_reels = sorted(self._cells_reelles_par_point.keys())
        self.debit_reel_par_point = df_scenario[~df_scenario["in_tunnel"]].groupby("point_id")["debit_adj_mbps"].max()

        self._resultats_hypo = []

        # Réseau réel projeté sur (ix, iy, g, f), sans opérateur (o n'est pas dans l'action)
        antennes_uniques = self.df_reel[["ant_id", "ant_lat", "ant_lon", "generation", "bande_mhz"]].drop_duplicates("ant_id")
        self.deploiements_reels = set()
        self.sites_deja_presents = set()
        for row in antennes_uniques.itertuples():
            c = self.grille.gps_vers_cellule(row.ant_lat, row.ant_lon)
            if c is None:
                continue
            self.deploiements_reels.add((c[0], c[1], row.generation, row.bande_mhz))
            self.sites_deja_presents.add(c)

        self.infrastructure_deployee = set()
        self.budget_restant = self.budget_total
        self.t = 0
        self.done = False

        resultat_dp, n_white, n_eval = self._evaluer_dp_courant()
        self._dernier_resultat_dp = resultat_dp
        self._dernier_n_white = n_white
        self._dernier_n_eval = n_eval
        self.U_courant = self._calculer_U(resultat_dp, n_eval, n_white)

        self._reset_appele = True
        return self._construire_etat()

    # ------------------------------------------------------------------
    # ÉVALUATION (DP + N_white)
    # ------------------------------------------------------------------
    def _fusionner_cells_par_point(self):
        """Candidats par point : cellules réelles + cellules simulées."""
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
        """Maximum de `cle` par point parmi les résultats simulés."""
        if not resultats:
            return pd.Series(dtype=float)
        return pd.Series(
            [r[cle] for r in resultats],
            index=[r["point_id"] for r in resultats],
        ).groupby(level=0).max()

    def _evaluer_dp_courant(self):
        points, cells_par_point = self._fusionner_cells_par_point()
        resultat_dp = trajectory_dp(points, cells_par_point, LAMBDA_H, LAMBDA_O, LAMBDA_T)

        if self._resultats_hypo:
            debit_hypo = self._serie_max_par_point(self._resultats_hypo, "debit_adj_mbps")
            debit_combine = pd.concat([self.debit_reel_par_point, debit_hypo], axis=1).max(axis=1)
        else:
            debit_combine = self.debit_reel_par_point

        # Zone blanche : meilleur débit < D_cov, sur tout le corridor hors tunnel
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
        """Infaisable si (ix, iy, g, f) existe déjà ou si le budget ne suffit pas."""
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
        """État = 27 canaux spatiaux + b_t + progression."""
        canaux_infra = construire_canaux_infrastructure(
            self.grille, self.df_reel, cellules_hypothetiques=self._resultats_hypo_pour_infra())

        if self._resultats_hypo:
            debit_hypo = self._serie_max_par_point(self._resultats_hypo, "debit_adj_mbps")
            qos_hypo = self._serie_max_par_point(self._resultats_hypo, "qos")
            debit_combine = pd.concat([self.debit_reel_par_point, debit_hypo], axis=1).max(axis=1)
        else:
            debit_combine = self.debit_reel_par_point

        meilleur_par_point = pd.DataFrame({"point_id": debit_combine.index, "meilleur_debit": debit_combine.values})
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
        return self._resultats_hypo

    # ------------------------------------------------------------------
    # JOURNALISATION
    # ------------------------------------------------------------------
    def _verifier_schema_log_rl(self):
        """Refuse d'écrire dans un fichier qui n'est pas un journal RL."""
        if not os.path.exists(self.dataset_enrichi_rl_path):
            return
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
                "Supprime ce fichier ou choisis un autre dataset_enrichi_rl_path."
            )

    def _journaliser(self, action, lat_site, lon_site, cout, resultats_sim):
        """Ajoute les résultats d'un déploiement simulé au journal du run."""
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
            raise RuntimeError("step() appelé après done=True : appeler reset() d'abord.")
        if not self._est_faisable(action):
            raise ValueError(f"Action infaisable : {action}. "
                              "Appliquer le masque de _actions_faisables() avant argmax.")

        if action == STOP:
            return self._step_stop()

        ix, iy, g, f = action
        lat, lon = self.grille.cellule_vers_gps(ix, iy)   # antenne au centre de la cellule

        resultats_sim = simuler_deploiement(lat, lon, g, f, self.df_points, self.df_meteo_scenario)
        resultats_sim = [r for r in resultats_sim if r["point_id"] in self.points_hors_tunnel]

        cout = get_total_cost((ix, iy), g, f, self.sites_deja_presents)

        if self.sauvegarder_deploiements_rl:
            self._journaliser(action, lat, lon, cout, resultats_sim)

        self._resultats_hypo.extend(resultats_sim)
        self.infrastructure_deployee.add((ix, iy, g, f))
        self.sites_deja_presents.add((ix, iy))
        self.budget_restant -= cout

        resultat_dp, n_white, n_eval = self._evaluer_dp_courant()
        U_apres = self._calculer_U(resultat_dp, n_eval, n_white)
        r_t = compute_r_t(U_apres, self.U_courant)

        self.U_courant = U_apres
        self._dernier_resultat_dp = resultat_dp
        self._dernier_n_white = n_white
        self._dernier_n_eval = n_eval
        self.t += 1

        # Fin forcée : plus d'action possible, couverture complète, ou t_max
        aucune_action_deploiement_possible = not any(
            self._est_faisable(a) for a in self.actions if a != STOP)
        couverture_complete = n_white == 0
        if aucune_action_deploiement_possible or couverture_complete or self.t >= self.t_max:
            return self._finaliser(r_t)

        return self._construire_etat(), r_t, False, {"cout": cout, "n_white": n_white}

    def _step_stop(self):
        return self._finaliser(0.0)

    def _finaliser(self, r_t_dernier_step):
        cout_total = self.budget_total - self.budget_restant

        # Arguments nommés : un changement de signature lèvera une erreur claire
        r_fin = compute_r_fin(
            n_white=self._dernier_n_white,
            n_total=self.n_total,
            qos_min=self._dernier_resultat_dp["qos_min"],
            cost_total=cout_total,
            budget=self.budget_total,
            Q_critique=self.Q_critique,
            R_couverture=self.R_couverture,
            R_qualite=self.R_qualite,
            R_succes=self.R_succes,
            R_eff=self.R_eff,
        )

        self.done = True
        return self._construire_etat(), r_t_dernier_step + r_fin, True, {
            "cout_total": cout_total, "n_white": self._dernier_n_white, "r_fin": r_fin}