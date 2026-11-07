"""
fetch_trace_ligne.py — tracé et tunnels d'un autre tronçon (test sur une 2e ligne)

Tracé : forme de la ligne RFN (SNCF open data), coupée entre deux gares.
Tunnels : voies ferrées tunnel=yes d'OpenStreetMap le long du tracé.

Usage : python fetch_trace_ligne.py 785000 Ambert Sembadel <dossier_sortie>
Sorties : trace.csv (point_id, latitude, longitude), tunnels.csv (même format que data/raw/tunnels.csv)
"""
import json, math, os, sys
import numpy as np
import pandas as pd
import requests

SNCF = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
UA = {"User-Agent": "FerroMobile/2.0 (railway connectivity research; academic)"}


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def sncf(dataset, where):
    r = requests.get(f"{SNCF}/{dataset}/records", params={"where": where, "limit": 100}, headers=UA, timeout=60)
    r.raise_for_status()
    return r.json()["results"]


def gare(nom):
    for ds in ["liste-des-gares", "gares-de-voyageurs"]:
        try:
            res = sncf(ds, f'search("{nom}")')
        except Exception as e:
            print("  ", ds, e)
            continue
        for g in res:
            geo = g.get("geo_point_2d") or g.get("position_geographique") or {}
            lib = str(g.get("libelle") or g.get("nom") or "")
            if geo and nom.lower() in lib.lower():
                print(f"  gare {lib} ({ds}) : {geo}")
                return geo["lat"], geo["lon"]
    raise RuntimeError(f"gare {nom} introuvable")


def chainer(morceaux):
    """Assemble les morceaux de polyligne en une seule, de bout en bout."""
    morceaux = [list(m) for m in morceaux if len(m) > 1]
    ligne = morceaux.pop(0)
    while morceaux:
        best = None
        for i, m in enumerate(morceaux):
            for inv in (False, True):
                mm = m[::-1] if inv else m
                for fin in (True, False):
                    d = hav(ligne[-1] if fin else ligne[0], mm[0] if fin else mm[-1])
                    if best is None or d < best[0]:
                        best = (d, i, inv, fin)
        d, i, inv, fin = best
        m = morceaux.pop(i)
        m = m[::-1] if inv else m
        ligne = ligne + m if fin else m + ligne
    return ligne


def main():
    code, g1, g2, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    res = sncf("formes-des-lignes-du-rfn", f'code_ligne="{code}"')
    print(f"{len(res)} formes pour la ligne {code}")
    morceaux = []
    for f in res:
        geom = f["geo_shape"]["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        morceaux += [[(c[1], c[0]) for c in p] for p in parts]
    ligne = chainer(morceaux)
    print(f"ligne assemblée : {len(ligne)} sommets")
    a, b = gare(g1), gare(g2)
    ia = int(np.argmin([hav(p, a) for p in ligne]))
    ib = int(np.argmin([hav(p, b) for p in ligne]))
    print(f"  distance gare-tracé : {hav(ligne[ia], a):.3f} km, {hav(ligne[ib], b):.3f} km")
    tr = ligne[min(ia, ib):max(ia, ib) + 1]
    if ia > ib:
        tr = tr[::-1]
    L = sum(hav(tr[i], tr[i + 1]) for i in range(len(tr) - 1))
    print(f"tronçon {g1} -> {g2} : {len(tr)} sommets, {L:.2f} km")
    pd.DataFrame({"point_id": range(len(tr)), "latitude": [p[0] for p in tr], "longitude": [p[1] for p in tr]}).to_csv(
        os.path.join(out, "trace.csv"), index=False)

    # tunnels OSM dans la boîte du tronçon, gardés s'ils touchent le tracé (< 50 m)
    la = [p[0] for p in tr]; lo = [p[1] for p in tr]
    bbox = f"({min(la) - 0.01},{min(lo) - 0.01},{max(la) + 0.01},{max(lo) + 0.01})"
    q = f'[out:json][timeout:120];way["railway"]["tunnel"="yes"]{bbox};out geom tags;'
    data = None
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=180)
            r.raise_for_status(); data = r.json(); break
        except Exception as e:
            print("  overpass", url, e)
    tunnels = []
    for w in (data or {}).get("elements", []):
        g = [(n["lat"], n["lon"]) for n in w.get("geometry", [])]
        if len(g) < 2 or min(min(hav(p, q2) for q2 in tr) for p in (g[0], g[-1])) > 0.05:
            continue
        lg = sum(hav(g[i], g[i + 1]) for i in range(len(g) - 1)) * 1000
        tunnels.append({"nom": w.get("tags", {}).get("name", f"Tunnel OSM {w['id']}"), "longueur_m": round(lg),
                        "lat_entree": g[0][0], "lon_entree": g[0][1], "lat_sortie": g[-1][0], "lon_sortie": g[-1][1]})
    df_t = pd.DataFrame(tunnels, columns=["nom", "longueur_m", "lat_entree", "lon_entree", "lat_sortie", "lon_sortie"])
    df_t.to_csv(os.path.join(out, "tunnels.csv"), index=False)
    print(f"{len(df_t)} tunnels :\n{df_t.to_string()}")


if __name__ == "__main__":
    main()
