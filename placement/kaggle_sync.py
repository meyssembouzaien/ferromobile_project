"""
kaggle_sync.py : sauvegarde du run dans un dataset Kaggle privé.

Après chaque checkpoint, le dossier du run est envoyé comme nouvelle version du dataset.
Au démarrage, la dernière version est téléchargée et l'entraînement reprend.
Identifiants : secrets Kaggle KAGGLE_USERNAME et KAGGLE_KEY.
"""

import json
import os
import shutil
import subprocess
import tempfile

FICHIER_METADONNEES = "dataset-metadata.json"
FICHIER_ETAT = "run_etat.json"


def configurer_identifiants():
    """True si l'API Kaggle peut s'authentifier."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    for chemin in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
        if os.path.exists(os.path.expanduser(chemin)):
            with open(os.path.expanduser(chemin)) as f:
                os.environ.setdefault("KAGGLE_USERNAME", json.load(f).get("username", ""))
            return True
    try:
        from kaggle_secrets import UserSecretsClient   # disponible seulement dans un notebook Kaggle
        secrets = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
        return True
    except Exception as e:
        print(f"[kaggle_sync] pas d'identifiants Kaggle ({type(e).__name__}) : "
              "synchronisation désactivée, sauvegardes locales seulement")
        return False


def slug_par_defaut(nom_run):
    """Nom de dataset valide (minuscules, tirets, 50 caractères au plus)."""
    s = "".join(c if c.isalnum() else "-" for c in f"fm-{nom_run}".lower())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")[:50]


def lire_etat(dossier):
    chemin = os.path.join(dossier, FICHIER_ETAT)
    if not os.path.exists(chemin):
        return None
    with open(chemin) as f:
        return json.load(f)


def ecrire_etat(dossier, etat):
    temporaire = os.path.join(dossier, FICHIER_ETAT + ".tmp")
    with open(temporaire, "w") as f:
        json.dump(etat, f, indent=1)
    os.replace(temporaire, os.path.join(dossier, FICHIER_ETAT))


def _kaggle(*arguments, timeout=1800):
    """Lance la commande kaggle ; renvoie (code, sortie)."""
    try:
        p = subprocess.run(["kaggle", *arguments], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, "commande kaggle introuvable (pip install kaggle)"
    except subprocess.TimeoutExpired:
        return 124, f"délai dépassé ({timeout} s)"


class SauvegardeKaggle:
    """Un dataset privé par run. actif = False si pas d'identifiants ou désactivé."""

    def __init__(self, dossier_run, nom_run, slug=None, actif=True):
        self.dossier = dossier_run
        self.actif = actif and configurer_identifiants()
        utilisateur = os.environ.get("KAGGLE_USERNAME", "")
        if slug and "/" in slug:
            self.id = slug
        else:
            self.id = f"{utilisateur}/{slug or slug_par_defaut(nom_run)}"
        self.titre = f"FerroMobile run {nom_run}"[:50]
        self._existe = None
        if self.actif:
            print(f"[kaggle_sync] dataset du run : {self.id} (privé)")

    def _ecrire_metadonnees(self):
        os.makedirs(self.dossier, exist_ok=True)
        with open(os.path.join(self.dossier, FICHIER_METADONNEES), "w") as f:
            json.dump({"title": self.titre, "id": self.id, "licenses": [{"name": "CC0-1.0"}]}, f)

    def restaurer(self):
        """Télécharge la dernière version du run. Renvoie l'épisode restauré, ou None."""
        candidats = []
        if self.actif:
            temporaire = tempfile.mkdtemp(prefix="fm_restauration_")
            code, sortie = _kaggle("datasets", "download", self.id, "-p", temporaire, "--unzip", "-q")
            if code != 0 and "usage" in sortie.lower():   # anciennes versions du client : option -d
                code, sortie = _kaggle("datasets", "download", "-d", self.id, "-p", temporaire, "--unzip", "-q")
            if code == 0:
                self._existe = True
                candidats.append(temporaire)
            else:
                self._existe = False if ("404" in sortie or "not found" in sortie.lower()) else None
                print(f"[kaggle_sync] pas de version téléchargée ({sortie[-200:]})")
        # Dataset ajouté au notebook avec Add Input
        nom = self.id.split("/")[-1]
        for base in (f"/kaggle/input/{nom}", f"/kaggle/input/datasets/{self.id}"):
            if os.path.isdir(base):
                candidats.append(base)

        episode_local = (lire_etat(self.dossier) or {}).get("episode", -1)
        meilleur, episode_max = None, episode_local
        for c in candidats:
            # Le zip peut contenir le dossier du run comme sous-dossier
            for racine, _, fichiers in os.walk(c):
                if FICHIER_ETAT in fichiers:
                    e = (lire_etat(racine) or {}).get("episode", -1)
                    if e > episode_max:
                        meilleur, episode_max = racine, e
        if meilleur is None:
            print(f"[kaggle_sync] rien de plus récent que le dossier local (épisode {episode_local})")
            return None
        os.makedirs(self.dossier, exist_ok=True)
        for f in os.listdir(meilleur):
            source = os.path.join(meilleur, f)
            if os.path.isfile(source) and f != FICHIER_METADONNEES:
                shutil.copy2(source, os.path.join(self.dossier, f))
        print(f"[kaggle_sync] run restauré depuis {meilleur} : épisode {episode_max}")
        return episode_max

    def envoyer(self, message):
        """Nouvelle version du dataset avec tout le dossier du run. Ne lève jamais d'exception."""
        if not self.actif:
            return False
        try:
            self._ecrire_metadonnees()
            if self._existe is None:
                code, _ = _kaggle("datasets", "status", self.id, timeout=120)
                self._existe = code == 0
            if self._existe:
                code, sortie = _kaggle("datasets", "version", "-p", self.dossier, "-m", message, "-q")
            else:
                code, sortie = _kaggle("datasets", "create", "-p", self.dossier, "-q")
                if code == 0:
                    self._existe = True
            if code != 0:
                print(f"[kaggle_sync] ÉCHEC de l'envoi ({sortie[-300:]}) : retenté au prochain checkpoint")
                return False
            print(f"[kaggle_sync] version envoyée : {message}")
            return True
        except Exception as e:   # la sauvegarde ne doit jamais arrêter l'entraînement
            print(f"[kaggle_sync] ÉCHEC de l'envoi ({e}) : retenté au prochain checkpoint")
            return False
