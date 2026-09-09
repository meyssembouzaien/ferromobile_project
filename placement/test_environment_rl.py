"""
test_environment_rl.py — Tests unitaires de FerroMobileEnv
==================================================================
S'exécute sur les VRAIES données (dataset_base.csv, dataset_enrichi.csv)
du repo, pas sur des données synthétiques : environment_rl.py importe
prepare_data.py (via simulateur_deploiement.py) au niveau du module, ce
qui nécessite crc-covlib de toute façon, même pour les tests qui
n'appellent jamais simuler_deploiement() eux-mêmes.

Deux catégories de tests, séparées explicitement pour ne jamais
transformer un smoke test en fausse validation scientifique :

  - CATÉGORIE STRUCTURE (tests 1 à 4) : n'appellent jamais step() avec
    une action de déploiement réelle, donc jamais simuler_deploiement().
    Rapides, déterministes, vérifient grille/actions/faisabilité/coût.

  - CATÉGORIE PHYSIQUE (tests 5 à 7) : appellent step() avec de vraies
    actions de déploiement, donc la vraie chaîne P.1812/LOS/débit. Plus
    lentes, et leur résultat dépend de la géométrie réelle du corridor
    (impossible de prédire à la main sans avoir tourné le code une
    première fois) — les assertions portent sur des INVARIANTS
    (cohérence interne, sens des variations), pas des valeurs numériques
    figées comme pour test_r_fin.py.

Lancer : python test_environment_rl.py
"""

from environment_rl import FerroMobileEnv, STOP, BANDES_PAR_GENERATION


def _env_test(**kwargs):
    """Environnement de test : pas de journalisation (évite de polluer
    dataset_enrichi_rl.csv à chaque lancement des tests), budget large
    par défaut pour ne pas être bloqué artificiellement."""
    defaults = dict(scenario="S1", budget=2_000_000.0, t_max=50,
                     sauvegarder_deploiements_rl=False, run_id="test_environment_rl")
    defaults.update(kwargs)
    return FerroMobileEnv(**defaults)


# ======================================================================
# CATÉGORIE STRUCTURE — pas de simuler_deploiement() réel
# ======================================================================

def test_1_reset():
    env = _env_test()
    etat = env.reset()

    assert etat["grille"].shape == (27, env.grille.nx, env.grille.ny), \
        f"Forme d'état inattendue : {etat['grille'].shape}"
    assert etat["b_t"] == 1.0, "b_t doit valoir 1.0 juste après reset (budget intact)"
    assert etat["progression"] == 0.0, "progression doit valoir 0.0 juste après reset"
    assert len(env.infrastructure_deployee) == 0, "aucun ajout RL avant le premier step()"
    assert len(env.deploiements_reels) > 0, \
        "deploiements_reels doit être construit depuis le réseau réel (non vide sur les vraies données)"
    assert len(env.sites_deja_presents) > 0, "sites_deja_presents doit contenir les sites réels"
    # deploiements_reels ne doit jamais contenir l'opérateur (§9 : action RL = (ix,iy,g,f))
    for d in env.deploiements_reels:
        assert len(d) == 4, f"deploiements_reels doit être des tuples (ix,iy,g,f), reçu {d}"
    print(f"test_1_reset : OK ({len(env.deploiements_reels)} déploiements réels, "
          f"{len(env.sites_deja_presents)} sites)")


def test_2_espace_actions():
    env = _env_test()
    env.reset()

    n_cellules_corridor = int(env.grille.corridor_mask.sum())
    n_combos_gf = sum(len(v) for v in BANDES_PAR_GENERATION.values())
    attendu = n_cellules_corridor * n_combos_gf + 1  # +1 pour STOP

    assert len(env.actions) == attendu, \
        f"|A|={len(env.actions)} != attendu={attendu} (cellules corridor={n_cellules_corridor}, combos g,f={n_combos_gf})"
    assert env.actions[0] == STOP, "STOP doit être en première position (action_id=0)"

    for a in env.actions:
        if a == STOP:
            continue
        ix, iy, g, f = a
        assert env.grille.corridor_mask[ix, iy], \
            f"Action {a} hors corridor_mask — §15 : le RL ne doit jamais pouvoir choisir cette cellule"
        assert f in BANDES_PAR_GENERATION[g], f"Bande {f} invalide pour {g}"

    print(f"test_2_espace_actions : OK (|A|={len(env.actions)} = "
          f"{n_cellules_corridor} cellules x {n_combos_gf} combos + 1 STOP)")


def test_3_faisabilite():
    env = _env_test(budget=10.0)  # budget quasi nul, exprès
    env.reset()

    assert env._est_faisable(STOP), "STOP doit toujours être faisable, même à budget quasi nul"

    # Une combinaison déjà dans le réseau réel doit être masquée
    exemple_reel = next(iter(env.deploiements_reels))
    assert not env._est_faisable(exemple_reel), \
        f"{exemple_reel} est déjà dans le réseau réel, doit être infaisable"

    # Avec un budget quasi nul, la quasi-totalité des déploiements doit être infaisable
    masque = env._actions_faisables()
    assert env.actions[0] == STOP
    assert masque[0], "STOP doit toujours être faisable"
    n_faisables_hors_stop = sum(1 for i, ok in enumerate(masque) if ok and env.actions[i] != STOP)
    assert n_faisables_hors_stop == 0, \
        f"Avec budget=10€, aucun déploiement ne devrait être faisable, {n_faisables_hors_stop} le sont"

    # Avec un gros budget, au moins une action de déploiement doit être faisable
    env2 = _env_test(budget=2_000_000.0)
    env2.reset()
    masque2 = env2._actions_faisables()
    n_faisables2 = sum(1 for i, ok in enumerate(masque2) if ok and env2.actions[i] != STOP)
    assert n_faisables2 > 0, "Avec un budget large, au moins une action de déploiement doit être faisable"

    print(f"test_3_faisabilite : OK (0 faisable à budget=10€, "
          f"{n_faisables2} faisables à budget=2M€)")


def test_4_cout():
    """Ce test valide l'intégration entre environment_rl et cost_model
    (les identifiants (ix,iy)/sites_deja_presents construits par
    l'environnement se comportent correctement avec get_total_cost) —
    ce n'est PAS un test du comportement complet de step() sur le coût
    (qui nécessiterait deux vrais déploiements physiques successifs,
    donc catégorie physique). cost_model.py lui-même est déjà testé
    séparément et exhaustivement dans test_cost_model.py."""
    from cost_model import get_total_cost, C_SITE_EUR
    env = _env_test()
    env.reset()

    # Trouver une action sur une cellule qui n'est PAS déjà un site réel
    ix_iy_reels = {(ix, iy) for (ix, iy, g, f) in env.deploiements_reels}
    action_nouveau_site = next(
        a for a in env.actions if a != STOP and (a[0], a[1]) not in ix_iy_reels)
    ix, iy, g, f = action_nouveau_site
    cout_1 = get_total_cost((ix, iy), g, f, env.sites_deja_presents)

    # Deuxième technologie sur CE MÊME nouveau site : seulement le radio, pas C_site
    autres_combos = [(gg, ff) for gg in BANDES_PAR_GENERATION for ff in BANDES_PAR_GENERATION[gg]
                      if (gg, ff) != (g, f)]
    g2, f2 = autres_combos[0]
    sites_apres_1er_ajout = env.sites_deja_presents | {(ix, iy)}
    cout_2 = get_total_cost((ix, iy), g2, f2, sites_apres_1er_ajout)

    assert cout_1 >= C_SITE_EUR, f"Premier ajout sur un nouveau site doit inclure C_site, cout={cout_1}"
    assert cout_2 < C_SITE_EUR, f"Deuxième techno sur le même site ne doit PAS repayer C_site, cout={cout_2}"
    print(f"test_4_cout : OK (1er ajout={cout_1:.0f}€ incl. C_site, "
          f"2e techno même site={cout_2:.0f}€ radio seul)")


# ======================================================================
# CATÉGORIE PHYSIQUE — appelle step() avec de vraies actions,
# donc la vraie chaîne P.1812/LOS/débit (simuler_deploiement réel).
# ======================================================================

def test_5_step_deploiement():
    env = _env_test()
    env.reset()
    budget_avant = env.budget_restant
    n_infra_avant = len(env.infrastructure_deployee)

    masque = env._actions_faisables()
    idx = next(i for i, ok in enumerate(masque) if ok and env.actions[i] != STOP)
    action = env.actions[idx]

    etat, r_t, done, info = env.step(action)

    assert not done, "un seul step() de déploiement ne devrait pas terminer l'épisode (budget large)"
    assert len(env.infrastructure_deployee) == n_infra_avant + 1, "l'action doit être enregistrée"
    assert action in env.infrastructure_deployee
    assert env.budget_restant < budget_avant, "le budget doit diminuer après un déploiement facturé"
    assert isinstance(r_t, float), "r_t doit être un nombre"
    assert etat["grille"].shape == (27, env.grille.nx, env.grille.ny)
    assert etat["b_t"] == env.budget_restant / env.budget_total
    assert env.t == 1, "le compteur temporel doit avancer après un step()"
    assert etat["progression"] == env.t / env.t_max
    assert info["cout"] > 0, "le déploiement facturé doit avoir un coût strictement positif"
    assert info["n_white"] == env._dernier_n_white, "info['n_white'] doit refléter le dernier état interne"

    # Redéployer EXACTEMENT la même action doit maintenant être infaisable
    assert not env._est_faisable(action), "l'action vient d'être prise, ne doit plus être faisable"

    print(f"test_5_step_deploiement : OK (action={action}, r_t={r_t:.4f}, "
          f"budget {budget_avant:.0f}->{env.budget_restant:.0f}€)")


def test_6_stop():
    env = _env_test()
    env.reset()

    # Un déploiement d'abord, pour avoir un état non trivial avant STOP
    masque = env._actions_faisables()
    idx = next(i for i, ok in enumerate(masque) if ok and env.actions[i] != STOP)
    env.step(env.actions[idx])

    etat, retour, done, info = env.step(STOP)

    assert done is True, "STOP doit terminer l'épisode"
    assert "r_fin" in info, "l'info de fin doit exposer r_fin séparément"
    assert "cout_total" in info
    assert info["cout_total"] == env.budget_total - env.budget_restant
    print(f"test_6_stop : OK (r_fin={info['r_fin']:.4f}, cout_total={info['cout_total']:.0f}€, "
          f"n_white={info['n_white']})")


def test_7_fin_forcee_automatique():
    """Budget volontairement petit : l'épisode doit atteindre rapidement
    une condition de fin automatique. Selon les coûts des technologies
    disponibles, la cause effective peut être l'absence d'action de
    déploiement faisable ou l'atteinte de t_max — nom volontairement
    pas "budget_epuise" : ce test vérifie explicitement LAQUELLE des
    deux s'est produite, plutôt que de supposer que c'est le budget."""
    env = _env_test(budget=50_000.0, t_max=50)
    env.reset()

    done = False
    n_steps = 0
    while not done and n_steps < env.t_max:
        masque = env._actions_faisables()
        idxs_deploiement = [i for i, ok in enumerate(masque) if ok and env.actions[i] != STOP]
        if not idxs_deploiement:
            # Ne devrait pas arriver : step() doit avoir déjà déclenché
            # la fin forcée dès que plus aucun déploiement n'est faisable.
            raise AssertionError("Aucune action faisable mais done=False : la fin forcée a raté")
        action = env.actions[idxs_deploiement[0]]
        etat, r, done, info = env.step(action)
        n_steps += 1

    assert done, f"L'épisode doit se terminer (budget épuisé ou t_max), arrêté après {n_steps} steps"
    assert env.budget_restant >= 0, "le budget ne doit jamais devenir négatif"

    # Vérifie explicitement la cause réelle de la fin forcée, plutôt que
    # de supposer que c'est le budget : soit plus aucune action de
    # déploiement n'est faisable, soit t_max est atteint (exactement la
    # disjonction utilisée dans step()).
    masque_final = env._actions_faisables()
    reste_action_deploiement = any(
        ok and env.actions[i] != STOP for i, ok in enumerate(masque_final))
    assert not reste_action_deploiement or env.t >= env.t_max, \
        "la fin forcée a eu lieu sans qu'aucune des deux causes attendues ne soit vraie"

    cause = "budget épuisé" if not reste_action_deploiement else "t_max atteint"
    print(f"test_7_fin_forcee_automatique : OK (terminé après {n_steps} steps, "
          f"cause={cause}, budget_restant={env.budget_restant:.0f}€)")


if __name__ == "__main__":
    print("=== Catégorie structure (rapide, pas de simuler_deploiement) ===")
    test_1_reset()
    test_2_espace_actions()
    test_3_faisabilite()
    test_4_cout()

    print("\n=== Catégorie physique (appelle simuler_deploiement réel) ===")
    test_5_step_deploiement()
    test_6_stop()
    test_7_fin_forcee_automatique()

    print("\nTous les tests passent.")