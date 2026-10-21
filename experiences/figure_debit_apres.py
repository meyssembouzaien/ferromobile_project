"""
figure_debit_apres.py : best throughput along the track, before and after a deployment plan.

Colour = generation of the best link; red bands = white zones; grey bands = tunnels.
Run: python figure_debit_apres.py <liens_dueling_ep725.csv> <out_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
D_COV = 1.0
COULEURS = {"2G": "#9e9e9e", "3G": "#e69f00", "4G": "#1f3b73", "5G": "#56b4e9"}

e = pd.read_csv(os.path.join(RACINE, "data", "processed", "dataset_enrichi.csv"), low_memory=False,
                usecols=["point_id", "distance_km", "in_tunnel", "generation", "debit_adj_mbps", "scenario_id"])
s1 = e[e.scenario_id == "S1"]
pts = s1.drop_duplicates("point_id").set_index("point_id")[["distance_km", "in_tunnel"]]
pts["in_tunnel"] = pts.in_tunnel.astype(str).str.lower().isin(["true", "1"])
reels = s1[["point_id", "generation", "debit_adj_mbps"]].dropna()


def meilleur(liens):
    b = liens.loc[liens.groupby("point_id").debit_adj_mbps.idxmax()].set_index("point_id")
    out = pts.join(b)
    out["debit_adj_mbps"] = out.debit_adj_mbps.fillna(0)
    out["blanc"] = (out.debit_adj_mbps < D_COV) & ~out.in_tunnel
    return out


def bandes(ax, masque, x, couleur, alpha):
    debut = None
    for xi, m in zip(x, masque):
        if m and debut is None:
            debut = xi
        if not m and debut is not None:
            ax.axvspan(debut, xi, color=couleur, alpha=alpha, lw=0); debut = None
    if debut is not None:
        ax.axvspan(debut, x.iloc[-1], color=couleur, alpha=alpha, lw=0)


def tracer(ax, d, titre):
    d = d.sort_values("distance_km")
    bandes(ax, d.in_tunnel, d.distance_km, "grey", 0.25)
    bandes(ax, d.blanc, d.distance_km, "red", 0.15)
    ok = d[~d.in_tunnel & ~d.blanc]
    for g, c in COULEURS.items():
        m = ok[ok.generation == g]
        ax.scatter(m.distance_km, m.debit_adj_mbps, s=3, c=c, label=g)
    bl = d[d.blanc]
    ax.scatter(bl.distance_km, [0.15] * len(bl), s=4, c="red", marker="s", label=f"White point ({len(bl)})")
    ax.axhline(D_COV, ls="--", c="k", lw=0.8)
    ax.set_yscale("log"); ax.set_ylim(0.1, None)
    ax.set_title(titre, fontsize=10)
    ax.set_ylabel("Best throughput (Mbps)")
    ax.legend(fontsize=7, markerscale=3, loc="upper right", ncol=6)


liens_plan = pd.read_csv(sys.argv[1])
out_dir = sys.argv[2]
avant = meilleur(reels)
apres = meilleur(pd.concat([reels, liens_plan[["point_id", "generation", "debit_adj_mbps"]]]))
nom = os.path.basename(sys.argv[1]).replace("liens_", "").replace(".csv", "")
titre = {"dueling_ep725": "Dueling DQN (episode 725, 8 sites, 1.295 M€)",
         "ilp": "ILP (6 sites, 0.87 M€)"}.get(nom, nom)

fig, ax = plt.subplots(figsize=(11, 3.2))
tracer(ax, apres, f"Best throughput after deployment: {titre}")
ax.set_xlabel("Distance from Courpière (km)")
fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"best_throughput_after_{nom}.png"), dpi=200)

fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
tracer(axs[0], avant, "Existing network $I_0$")
tracer(axs[1], apres, f"After deployment: {titre}")
axs[1].set_xlabel("Distance from Courpière (km)")
fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"best_throughput_before_after_{nom}.png"), dpi=200)
print(nom, "white before", int(avant.blanc.sum()), "after", int(apres.blanc.sum()))
