"""
utility.py — U(I) et r_fin (cahier des charges §11, §14)
==========================================================================
Calcule l'utilité normalisée U(I) et la récompense terminale r_fin à
partir du résultat de trajectory_dp.py et du coût de cost_model.py.
Ce fichier ne recalcule rien lui-même (ni la DP, ni le coût, ni N_white) :
il combine des résultats déjà produits ailleurs.

    U(I) = w_Q*Q̂ - w_W*Ŵ - w_H*Ĥ - w_O*Ô - w_T*T̂         (§14)
    r_t  = U(I_{t+1}) - U(I_t)                              (§14)
    r_fin = -R_couverture               si N_white(I_T) > 0
          = -R_qualité                  si N_white=0 mais Q_min < Q_critique
          = R_succès + R_eff*(1-Cost(I_T)/B)   sinon           (§14)

Aucun poids par défaut n'est codé en dur dans les fonctions ci-dessous :
le cahier des charges (§14) précise explicitement que w_Q, w_W, w_H, w_O,
w_T, Q_critique, R_couverture, R_qualité, R_succès, R_eff sont "des
paramètres de modèle, testés par sensibilité" (§23, Expérience D), pas
des valeurs fixées a priori. L'appelant doit donc les fournir.
WEIGHTS_REFERENCE ci-dessous n'est PAS une valeur verrouillée : c'est un
point de départ (poids égaux) à faire varier dans l'analyse de
sensibilité, au même titre que toute autre combinaison testée.

Le budget B n'est pas une constante de ce fichier : il est passé en
paramètre à compute_r_fin(), cohérent avec la décision prise pour
cost_model.py (§13 : B est un paramètre de l'environnement RL, pas du
modèle de coût ni de la récompense elle-même).
"""

# Point de départ pour l'analyse de sensibilité, PAS une valeur
# verrouillée (voir docstring ci-dessus). Poids égaux, cohérent avec le
# choix déjà fait pour les 3 composantes internes de qos (§7).
WEIGHTS_REFERENCE = {"w_Q": 1.0, "w_W": 1.0, "w_H": 1.0, "w_O": 1.0, "w_T": 1.0}


def normaliser_metriques(resultat_dp, n_eval, n_white, n_total):
    """Construit Q̂, Ŵ, Ĥ, Ô, T̂ (§11) à partir du résultat de
    trajectory_dp() et d'un comptage de zones blanches déjà fait
    séparément (ex. compter_zones_blanches() de valider_chaine_complete.py).

    resultat_dp : dict retourné par trajectory_dp() — utilise qos_mean,
        N_H, N_O, N_T.
    n_eval : nombre de points évalués par la DP (= longueur de la
        trajectoire, N_eval du §11).
    n_white : nombre de points en zone blanche selon D_cov (§6), calculé
        sur TOUT le corridor, pas seulement les points évalués par la DP.
    n_total : nombre total de points du trajet hors tunnel (N_total, §11).

    Retourne un dict {"Q", "W", "H", "O", "T"}.
    """
    # Q̂ est déjà une moyenne sur les points de N_eval (§11), directement
    # qos_mean retourné par trajectory_dp() — aucune transformation.
    q_hat = resultat_dp["qos_mean"]

    # Protection ajoutée (pas dans le texte du §11, qui suppose N_eval>1) :
    # si un seul point est évalué, aucune transition n'est possible, donc
    # Ĥ/Ô/T̂ sont conventionnellement nuls plutôt que de diviser par zéro.
    denom = n_eval - 1
    if denom <= 0:
        h_hat = o_hat = t_hat = 0.0
    else:
        h_hat = resultat_dp["N_H"] / denom
        o_hat = resultat_dp["N_O"] / denom
        t_hat = resultat_dp["N_T"] / denom

    w_hat = n_white / n_total

    return {"Q": q_hat, "W": w_hat, "H": h_hat, "O": o_hat, "T": t_hat}


def compute_U(metriques, w_Q, w_W, w_H, w_O, w_T):
    """U(I), §14. `metriques` = sortie de normaliser_metriques()."""
    return (w_Q * metriques["Q"]
            - w_W * metriques["W"]
            - w_H * metriques["H"]
            - w_O * metriques["O"]
            - w_T * metriques["T"])


def compute_r_t(U_apres, U_avant):
    """r_t = U(I_{t+1}) - U(I_t), §14."""
    return U_apres - U_avant


def compute_r_fin(n_white, n_total, qos_min, cost_total, budget,
                   Q_critique, R_couverture, R_qualite, R_succes, R_eff):
    """r_fin, §14.

    MODIFIÉ : la pénalité de couverture est maintenant proportionnelle
    à Ŵ = n_white/n_total, pas une constante fixe. AVANT : n_white>0
    donnait toujours -R_couverture, quel que soit le degré de
    couverture manquante — même signal pour "1 point blanc" et "tous
    les points blancs". APRÈS : -R_couverture * Ŵ, donc quasi 0 si
    presque tout est couvert, -R_couverture si rien n'est couvert.

    Raison : la version constante donnait le même signal terminal
    (~-R_couverture) pour n'importe quelle dernière action d'épisode,
    peu importe sa qualité réelle — un bruit d'étiquetage qui empêchait
    l'apprentissage plutôt que de le ralentir (observé sur un run réel
    de 746 épisodes : perte qui augmente au lieu de baisser, n_white
    qui ne progresse pas malgré epsilon tombé à 0.106).

    n_white : N_white(I_T), compte brut (pas Ŵ).
    n_total : N_total, pour normaliser n_white en Ŵ (§11).
    qos_min : Q_min(I_T), le pire cas le long de la trajectoire choisie
        par la DP (qos_min retourné par trajectory_dp()).
    cost_total : Cost(I_T), coût total engagé (cost_model.get_total_cost()
        cumulé sur tous les déploiements de l'épisode).
    budget : B, fourni par l'appelant (§13 : paramètre de
        l'environnement RL, pas une constante de ce fichier).
    """
    if not (R_couverture > R_qualite > 0):
        raise ValueError(
            "compute_r_fin: le cahier des charges (§14) exige "
            "R_couverture > R_qualité > 0, reçu "
            f"R_couverture={R_couverture}, R_qualité={R_qualite}"
        )
    if n_total <= 0:
        raise ValueError("compute_r_fin: n_total doit être positif")

    if n_white > 0:
        return -R_couverture * (n_white / n_total)
    if qos_min < Q_critique:
        return -R_qualite
    return R_succes + R_eff * (1 - cost_total / budget)