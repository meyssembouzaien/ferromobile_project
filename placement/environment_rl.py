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
    - État à 28 canaux : une carte des mâts construits par le RL. Sans elle, l'agent
      ne voit pas ses propres antennes (sans opérateur), alors que le coût en dépend (§12).
    - b_t = budget restant / B_train (équation 3.29 du mémoire), et non / budget de
      l'épisode : avec B != B_train (Expérience G), b_t peut dépasser 1.

Accélérations sans effet sur les résultats (tests : test_chemins_rapides.py) :
    - masque de faisabilité vectorisé (numpy) au lieu d'une boucle sur 156 061 actions ;
    - canaux d'infrastructure calculés une fois par reset (ils ne changent pas : une
      cellule hypothétique n'a pas d'opérateur) ;
    - canaux de performance vectorisés (même ordre de sommation) ;
    - cache des simulations radio : simuler_deploiement() est une fonction pure
      (shadowing tiré par lien, graine stable), une antenne déjà simulée n'est pas
      recalculée. Taille : FM_SIM_CACHE entrées (défaut 30000, 0 pour désactiver).

Options pour les Expériences C et F (aucune n'est active par défaut) :
    - points_evalues : sous-ensemble de points (blocs de la validation croisée spatiale) ;
    - points_stats : points utilisés pour les statistiques de normalisation des canaux ;
    - operateurs_exclus : opérateurs retirés du réseau existant (niveau 2) ;
    - actions_autorisees : masque fixe des actions permises (petite instance, buffer).
"""

import os
from collections import OrderedDict
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from grid_config import construire_grille_depuis_dataset
from build_state import construire_canaux_statiques
from build_state_dynamic import (construire_canaux_infrastructure, construire_canaux_performance,
                                 construire_canal_mats_rl)
from trajectory_dp import trajectory_dp
from simulateur_deploiement import simuler_deploiement
import cost_model
from cost_model import get_total_cost, get_radio_cost
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

# Cache des simulations radio, partagé par tous les environnements du processus.
# Clé : (scénario, ix, iy, g, f). Valeur : résultats bruts de simuler_deploiement.
TAILLE_CACHE_SIMULATION = int(os.environ.get("FM_SIM_CACHE", 30000))
_CACHE_SIMULATION = OrderedDict()


def _compacter(resultats):
    """Résultats d'une simulation -> forme compacte (même valeurs exactes)."""
    if not resultats:
        return None
    return (resultats[0]["cellule"],
            [r["point_id"] for r in resultats],
            np.array([[r["qos"], r["debit_adj_mbps"], r["rtt_ms"], r["ber"]] for r in resultats],
                     dtype=np.float64))


def _decompacter(compact):
    if compact is None:
        return []
    cellule, pids, valeurs = compact
    return [{"point_id": pid, "cellule": cellule, "qos": q, "debit_adj_mbps": d, "rtt_ms": rtt, "ber": ber}
            for pid, (q, d, rtt, ber) in zip(pids, valeurs.tolist())]


def vider_cache_simulation():
    _CACHE_SIMULATION.clear()

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
                 run_id=None, episode_id=None, budget_reference=None,
                 points_evalues=None, points_stats=None, operateurs_exclus=(),
                 actions_autorisees=None):
        self.scenario = scenario
        # b_t = budget restant / B_train (éq. 3.29) ; B_train = budget de l'épisode par défaut
        self.budget_reference = budget_reference if budget_reference is not None else budget
        self.operateurs_exclus = set(operateurs_exclus)
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
        # Points de la performance (canaux) : tous, ou le sous-ensemble évalué (Expérience C, niveau 1)
        self.df_points_perf = self.df_points
        if points_evalues is not None:
            points_evalues = set(points_evalues)
            self.points_hors_tunnel &= points_evalues
            self.df_points_perf = self.df_points[self.df_points["point_id"].isin(points_evalues)]
        self.points_stats = None if points_stats is None else set(points_stats)
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

        # Tableaux pour le masque vectorisé (même règle que _est_faisable)
        self._cellule_action = np.array([a[0] * self.grille.ny + a[1] for a in self.actions[1:]], dtype=np.int64)
        self._cout_radio = np.array([get_radio_cost(a[2], a[3]) for a in self.actions[1:]], dtype=np.int64)
        if actions_autorisees is not None:
            actions_autorisees = np.asarray(actions_autorisees, dtype=bool).copy()
            assert actions_autorisees.shape == (len(self.actions),), "actions_autorisees : une valeur par action"
            actions_autorisees[0] = True   # STOP est toujours permis
        self.actions_autorisees = actions_autorisees

        # Cellule de grille de chaque point (canaux de performance), calculée une fois
        self._pid_perf = self.df_points_perf["point_id"].values
        cellules = [self.grille.gps_vers_cellule(la, lo)
                    for la, lo in zip(self.df_points_perf["lat"].values, self.df_points_perf["lon"].values)]
        self._cellule_perf = np.array([-1 if c is None else c[0] * self.grille.ny + c[1] for c in cellules],
                                      dtype=np.int64)

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
        self.canaux_statiques = construire_canaux_statiques(self.grille, self.df_points, self.df_meteo_scenario,
                                                            points_stats=self.points_stats)

        df_scenario = self.df_enrichi_complet[self.df_enrichi_complet["scenario_id"] == self.scenario]
        if self.operateurs_exclus:
            # Niveau 2 : les liens de ces opérateurs disparaissent ; les points restent
            df_scenario = df_scenario[~df_scenario["operateur"].isin(self.operateurs_exclus)]
        # Réseau réel complet (antennes, sites, canaux d'infrastructure) ...
        self.df_reel_complet = df_scenario[(~df_scenario["in_tunnel"]) & (df_scenario["ant_id"].notna())]
        # ... et ses liens vers les points évalués (DP, N_white)
        self.df_reel = self.df_reel_complet[self.df_reel_complet["point_id"].isin(self.points_hors_tunnel)]

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
        self.debit_reel_par_point = self.debit_reel_par_point[
            self.debit_reel_par_point.index.isin(self.points_hors_tunnel)]
        self.qos_reelle_par_point = self.df_reel.groupby("point_id")["qos"].max()

        self._resultats_hypo = []

        # Réseau réel projeté sur (ix, iy, g, f), sans opérateur (o n'est pas dans l'action)
        antennes_uniques = self.df_reel_complet[["ant_id", "ant_lat", "ant_lon", "generation", "bande_mhz"]].drop_duplicates("ant_id")
        self.deploiements_reels = set()
        self.sites_deja_presents = set()
        for row in antennes_uniques.itertuples():
            c = self.grille.gps_vers_cellule(row.ant_lat, row.ant_lon)
            if c is None:
                continue
            self.deploiements_reels.add((c[0], c[1], row.generation, row.bande_mhz))
            self.sites_deja_presents.add(c)

        # Canaux d'infrastructure : fixes (une cellule hypothétique n'a pas d'opérateur)
        self.canaux_infra = construire_canaux_infrastructure(self.grille, self.df_reel_complet)

        # État du masque vectorisé
        self._deploye = np.zeros(len(self.actions), dtype=bool)
        for d in self.deploiements_reels:
            idx = self.action_index.get(d)
            if idx is not None:
                self._deploye[idx] = True
        self._site_present = np.zeros(self.grille.nx * self.grille.ny, dtype=bool)
        for ix, iy in self.sites_deja_presents:
            self._site_present[ix * self.grille.ny + iy] = True

        self.infrastructure_deployee = set()
        self.budget_restant = self.budget_total
        self.t = 0
        self.done = False

        resultat_dp, n_white, n_eval = self._evaluer_dp_courant()
        self._dernier_resultat_dp = resultat_dp
        self._dernier_n_white = n_white
        self._dernier_n_eval = n_eval
        self.U_courant = self._calculer_U(resultat_dp, n_eval, n_white)
        self.U0_reference = self.U_reference()

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

    def U_reference(self):
        """U du réseau courant avec les poids de référence (tous à 1), quels que soient les
        poids de l'environnement : sert à comparer les méthodes (tableau 4.8)."""
        metriques = normaliser_metriques(self._dernier_resultat_dp, self._dernier_n_eval,
                                         self._dernier_n_white, self.n_total)
        return compute_U(metriques, 1.0, 1.0, 1.0, 1.0, 1.0)

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
        if self.actions_autorisees is not None and not self.actions_autorisees[self.action_index[action]]:
            return False
        ix, iy, g, f = action
        if (ix, iy, g, f) in self.deploiements_reels or (ix, iy, g, f) in self.infrastructure_deployee:
            return False
        cout = get_total_cost((ix, iy), g, f, self.sites_deja_presents)
        return cout <= self.budget_restant

    def _actions_faisables(self):
        """Masque de faisabilité, vectorisé. Même règle que _est_faisable (équation 3.18)."""
        cout = self._cout_radio + cost_model.C_SITE_EUR * (~self._site_present[self._cellule_action])
        masque = np.empty(len(self.actions), dtype=bool)
        masque[0] = True
        masque[1:] = (~self._deploye[1:]) & (cout <= self.budget_restant)
        if self.actions_autorisees is not None:
            masque &= self.actions_autorisees
        return masque

    def _actions_faisables_reference(self):
        """Version d'origine (boucle Python), gardée pour les tests d'équivalence."""
        return np.array([self._est_faisable(a) for a in self.actions], dtype=bool)

    # ------------------------------------------------------------------
    # ÉTAT
    # ------------------------------------------------------------------
    def _construire_etat(self):
        """État = 28 canaux spatiaux + b_t + progression."""
        canaux_infra = self.canaux_infra

        if self._resultats_hypo:
            debit_hypo = self._serie_max_par_point(self._resultats_hypo, "debit_adj_mbps")
            qos_hypo = self._serie_max_par_point(self._resultats_hypo, "qos")
            debit_combine = pd.concat([self.debit_reel_par_point, debit_hypo], axis=1).max(axis=1)
        else:
            debit_combine = self.debit_reel_par_point

        qos_reelle = self.qos_reelle_par_point
        if self._resultats_hypo:
            meilleure_qos = pd.concat([qos_reelle, qos_hypo], axis=1).max(axis=1)
        else:
            meilleure_qos = qos_reelle

        canaux_perf = self._canaux_performance(debit_combine, meilleure_qos)

        canal_mats = construire_canal_mats_rl(self.grille, self.infrastructure_deployee)
        etat_spatial = np.concatenate([self.canaux_statiques, canaux_infra, canaux_perf, canal_mats], axis=0)
        b_t = self.budget_restant / self.budget_reference
        progression = self.t / self.t_max
        return {"grille": etat_spatial, "b_t": b_t, "progression": progression}

    def _canaux_performance(self, debit_par_point, qos_par_point):
        """Version vectorisée de build_state_dynamic.construire_canaux_performance
        (mêmes conventions (-1, -1) et (0, -1), même ordre de sommation)."""
        nx, ny = self.grille.nx, self.grille.ny
        debit = debit_par_point.reindex(self._pid_perf).fillna(0).values
        qos = qos_par_point.reindex(self._pid_perf).values
        dans_grille = self._cellule_perf >= 0
        cellules = self._cellule_perf[dans_grille]
        couvert = debit[dans_grille] >= D_COV

        n_points = np.bincount(cellules, minlength=nx * ny)
        n_couverts = np.bincount(cellules[couvert], minlength=nx * ny)

        couverture = np.full(nx * ny, -1.0)
        qos_moy = np.full(nx * ny, -1.0)
        avec_points = n_points > 0
        couverture[avec_points] = n_couverts[avec_points] / n_points[avec_points]

        # Moyenne par cellule avec sum() de Python, dans l'ordre des points : résultat
        # identique au bit près à la version d'origine (sum() compensé en Python >= 3.12)
        cellules_cov = cellules[couvert]
        qos_cov = qos[dans_grille][couvert].tolist()
        ordre = np.argsort(cellules_cov, kind="stable")
        debuts = np.flatnonzero(np.r_[True, np.diff(cellules_cov[ordre]) != 0]) if len(ordre) else []
        fins = list(debuts[1:]) + [len(ordre)]
        for d, f in zip(debuts, fins):
            groupe = ordre[d:f]
            qos_moy[cellules_cov[groupe[0]]] = sum(qos_cov[i] for i in groupe) / len(groupe)
        return np.stack([couverture.reshape(nx, ny), qos_moy.reshape(nx, ny)])

    def _simuler(self, action, lat, lon):
        """simuler_deploiement() avec cache (fonction pure : même entrée, même sortie)."""
        ix, iy, g, f = action
        cle = (self.scenario, ix, iy, g, f)
        if TAILLE_CACHE_SIMULATION > 0 and cle in _CACHE_SIMULATION:
            _CACHE_SIMULATION.move_to_end(cle)
            return _decompacter(_CACHE_SIMULATION[cle])
        resultats = simuler_deploiement(lat, lon, g, f, self.df_points, self.df_meteo_scenario)
        if TAILLE_CACHE_SIMULATION > 0:
            _CACHE_SIMULATION[cle] = _compacter(resultats)
            if len(_CACHE_SIMULATION) > TAILLE_CACHE_SIMULATION:
                _CACHE_SIMULATION.popitem(last=False)
        return resultats

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

        resultats_sim = self._simuler(action, lat, lon)
        resultats_sim = [r for r in resultats_sim if r["point_id"] in self.points_hors_tunnel]

        cout = get_total_cost((ix, iy), g, f, self.sites_deja_presents)

        if self.sauvegarder_deploiements_rl:
            self._journaliser(action, lat, lon, cout, resultats_sim)

        self._resultats_hypo.extend(resultats_sim)
        self.infrastructure_deployee.add((ix, iy, g, f))
        self.sites_deja_presents.add((ix, iy))
        self._deploye[self.action_index[action]] = True
        self._site_present[ix * self.grille.ny + iy] = True
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
        aucune_action_deploiement_possible = not self._actions_faisables()[1:].any()
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