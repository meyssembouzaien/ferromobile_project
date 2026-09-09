"""
trajectory_dp.py — Programmation dynamique pour la sélection de cellule
=========================================================================
Étant donné un trajet (une liste de points ordonnés) et, pour chaque
point, les cellules radio disponibles, cette fonction trouve la
meilleure séquence de cellules : celle qui maximise la QoS totale tout
en pénalisant les changements de cellule, d'opérateur et de technologie
(cahier des charges, §10-11).

Cette fonction ne connaît ni la ligne complète, ni un tronçon, ni le
RL, ni le nombre 1923. Elle prend juste des points et des cellules en
entrée. Elle sert donc aussi bien à évaluer toute la ligne qu'un
sous-tronçon (généralisation, niveau 1, §20).

Une cellule est le tuple (site, operateur, generation, bande). Pour une
cellule hypothétique créée par le RL, operateur vaut None (§9-10).

Convention sur les zones blanches : les points sans aucune cellule
candidate ne font PAS partie de la séquence donnée à cette fonction.
Ils sont comptés séparément dans N_white (§6, §11), indépendamment de
la trajectoire sélectionnée ici. Quand deux points évaluables sont
séparés par un ou plusieurs points sans candidat, la transition entre
eux utilise la pénalité standard lambda_H, sans pénalité
supplémentaire pour la coupure : c'est un choix de modélisation
volontaire (pas un oubli), pour ne pas pénaliser deux fois la même
zone blanche (une fois via N_white, une fois via une pénalité de
coupure). C'est à l'appelant (pas à cette fonction) de décider quels
points entrent dans la séquence.
"""


def cout_transition(cellule_avant, cellule_apres, lambda_H, lambda_O, lambda_T):
    """Pénalité pour passer de cellule_avant à cellule_apres.

    Les trois changements (cellule, opérateur, technologie) sont
    détectés séparément et peuvent se cumuler sur une même transition.
    Un changement d'opérateur ne compte que si les DEUX cellules ont un
    opérateur réel : None (cellule hypothétique) n'est jamais pénalisé.
    """
    _, operateur_avant, gen_avant, _ = cellule_avant
    _, operateur_apres, gen_apres, _ = cellule_apres

    changement_cellule = cellule_avant != cellule_apres
    changement_tech = gen_avant != gen_apres
    changement_operateur = (
        operateur_avant is not None
        and operateur_apres is not None
        and operateur_avant != operateur_apres
    )

    cout = 0.0
    if changement_cellule:
        cout += lambda_H
    if changement_operateur:
        cout += lambda_O
    if changement_tech:
        cout += lambda_T
    return cout


def trajectory_dp(points, cells_per_point, lambda_H, lambda_O, lambda_T):
    """Trouve la meilleure séquence de cellules sur un trajet.

    points : liste de points du trajet, dans l'ordre. Le contenu n'est
        pas utilisé directement ici (la QoS est déjà dans
        cells_per_point) ; ça sert juste à connaître T = len(points).
        Ne doit contenir QUE des points ayant au moins une cellule
        candidate (voir la convention sur les zones blanches ci-dessus).
    cells_per_point : liste de listes, une par point. cells_per_point[t]
        est une liste de dicts {"cellule": (site, operateur, gen, bande),
        "qos": valeur_qos_a_ce_point_pour_cette_cellule}. Si une même
        cellule apparaît deux fois au même point, seule la dernière
        valeur de qos est gardée (les doublons doivent être évités en
        amont, cette fonction ne les fusionne pas).
    lambda_H, lambda_O, lambda_T : poids des pénalités de transition.

    Retourne un dict : trajectory, J_star, qos_mean, qos_min, N_H, N_O, N_T.
    """
    if len(points) != len(cells_per_point):
        raise ValueError(
            f"trajectory_dp: points ({len(points)}) et cells_per_point "
            f"({len(cells_per_point)}) doivent avoir la même longueur"
        )

    T = len(points)
    if T == 0:
        raise ValueError("trajectory_dp: le trajet ne contient aucun point")
    for t in range(T):
        if not cells_per_point[t]:
            raise ValueError(f"trajectory_dp: aucune cellule disponible au point {t}")

    # meilleur_score[cellule] = meilleur J cumulé en terminant sur cette
    # cellule, au point qu'on est en train de traiter.
    # precedent[t][cellule] = quelle cellule choisie au point t-1 pour
    # arriver au mieux sur cette cellule au point t (sert à reconstruire
    # la trajectoire à la fin).
    precedent = [dict() for _ in range(T)]

    # Point 0 : aucune pénalité de transition, c'est le début du trajet fourni.
    meilleur_score = {c["cellule"]: c["qos"] for c in cells_per_point[0]}

    for t in range(1, T):
        nouveau_score = {}
        for candidat in cells_per_point[t]:
            cellule = candidat["cellule"]
            meilleur_precedent, meilleure_valeur = None, None
            for cellule_prec, score_prec in meilleur_score.items():
                cout = cout_transition(cellule_prec, cellule, lambda_H, lambda_O, lambda_T)
                valeur = score_prec - cout
                if meilleure_valeur is None or valeur > meilleure_valeur:
                    meilleure_valeur, meilleur_precedent = valeur, cellule_prec
            nouveau_score[cellule] = candidat["qos"] + meilleure_valeur
            precedent[t][cellule] = meilleur_precedent
        meilleur_score = nouveau_score

    # Meilleure cellule au dernier point, puis on remonte la chaîne des précédents.
    derniere_cellule = max(meilleur_score, key=meilleur_score.get)
    J_star = meilleur_score[derniere_cellule]

    trajectoire = [None] * T
    trajectoire[T - 1] = derniere_cellule
    for t in range(T - 1, 0, -1):
        trajectoire[t - 1] = precedent[t][trajectoire[t]]

    # Statistiques de la trajectoire trouvée (§11 du cahier des charges).
    qos_par_point = []
    for t in range(T):
        for candidat in cells_per_point[t]:
            if candidat["cellule"] == trajectoire[t]:
                qos_par_point.append(candidat["qos"])
                break

    N_H = N_O = N_T = 0
    for t in range(1, T):
        if trajectoire[t - 1] != trajectoire[t]:
            N_H += 1
        if trajectoire[t - 1][2] != trajectoire[t][2]:
            N_T += 1
        if (trajectoire[t - 1][1] is not None and trajectoire[t][1] is not None
                and trajectoire[t - 1][1] != trajectoire[t][1]):
            N_O += 1

    return {
        "trajectory": trajectoire,
        "J_star": J_star,
        "qos_mean": sum(qos_par_point) / T,
        "qos_min": min(qos_par_point),
        "N_H": N_H,
        "N_O": N_O,
        "N_T": N_T,
    }