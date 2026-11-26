"""
preparer_kaggle_commun.py : écrit les notebooks Kaggle (JSON de POST /api/v1/kernels/push) du protocole commun.
N'envoie rien : chaque fichier est poussé plus tard, à la main.

  entraînement  graines 0, 1, 2 sur la ligne entière ; plis de CV spatiale 0-3 (graine 0) ; GPU
  benchmark     toutes les méthodes de référence sur toutes les configurations ; CPU
Les deux lisent reglage_commun.py et les données du dépôt (ligne 1, végétation WorldCover, data/processed).

Lancer : python preparer_kaggle_commun.py <modele.json> <dossier_sortie>
  <modele.json> : un notebook d'entraînement existant (cellules d'installation reprises telles quelles)
"""
import json
import os
import sys

import reglage_commun as R

BRANCHE = "claude/project-thread-7krjjo"


def cellule(code):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code}




def notebook(modele, cellules):
    nb = json.loads(modele["text"])
    nb["cells"] = cellules
    return nb


def payload(modele, slug, cellules, gpu, sources):
    d = dict(modele)
    d.update({"slug": f"meyssembouzaien/{slug}", "newTitle": slug, "enableGpu": gpu,
              "kernelDataSources": sources, "text": json.dumps(notebook(modele, cellules))})
    return d


def main():
    modele = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    c = json.loads(modele["text"])["cells"]
    intro, installation, dem, tests, secrets, reprise, entrainement, journaux = c[0], c[2], c[3], c[4], c[5], c[6], c[8], c[9]

    runs = [(g, None) for g in R.GRAINES] + [(0, k) for k in range(4)]
    for g, k in runs:
        v = dict(R.variables_entrainement(g, k), FM_SYNC="0")
        code = ("# 1. Recette commune (experiences/reglage_commun.py), depuis zéro\nimport os\nRUN = "
                + json.dumps(v, indent=1) + f'\nBRANCHE = "{BRANCHE}"\nos.environ.update(RUN)\nprint(RUN)\n')
        slug = "ferromobile-" + v["FM_RUN"].replace("_", "-")
        d = payload(modele, slug, [intro, cellule(code), installation, dem, tests, secrets,
                                   reprise, entrainement, journaux], True, [])   # reprise : ajouter sa propre sortie
        json.dump(d, open(os.path.join(out, f"kaggle_push_{v['FM_RUN']}.json"), "w"))

    # Benchmark en 3 notebooks CPU (chacun tient dans une session de 12 h)
    groupes = {"complet": ["complet"], "secteurs": [f"sans_secteur_{k}" for k in range(4)],
               "operateurs": [n for n in R.CONFIGURATIONS if n.startswith("sans_") and "secteur" not in n]}
    for nom, configs in groupes.items():
        bench = ("# Toutes les méthodes de référence, même réglage, même machine (CPU)\n"
                 "!cd /kaggle/working/ferromobile_project/experiences && python benchmark_commun.py "
                 f"--configs {','.join(configs)} --out /kaggle/working/benchmark_commun 2>&1 | tee /kaggle/working/benchmark.log\n")
        code = f'import os\nBRANCHE = "{BRANCHE}"\nos.environ["PYTHONUNBUFFERED"] = "1"\n'
        d = payload(modele, f"ferromobile-benchmark-{nom}",
                    [cellule(f"# FerroMobile : benchmark commun, {nom}"), cellule(code), installation, dem,
                     tests, cellule(bench)], False, [])
        json.dump(d, open(os.path.join(out, f"kaggle_push_benchmark_{nom}.json"), "w"))
    print(sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
