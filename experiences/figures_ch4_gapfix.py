"""
figures_ch4_gapfix.py : Chapter 4 figures on the corrected data (219 white points), measured values only.

Run: python figures_ch4_gapfix.py
Output: /mnt/project-files/training/gap_fix_results/figures/
"""
import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = "/mnt/project-files/training/"
R = T + "gap_fix_results/"
OUT = R + "figures/"
PLANS = "/home/user/ferromobile_project/experiences/plans_gapfix/"
COUL = {"Existing network": "#8A94A6", "Coverage-Greedy": "#0F9D9A", "Trajectory-Greedy": "#F5A623",
        "ILP": "#1B2A49", "Dueling DQN": "#F26B5B"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

cc = pd.read_csv(R + "criteres_communs/criteres_communs.csv")
cc["methode"] = cc.methode.replace({"Dueling DQN ep725": "Dueling DQN"})
full = cc[cc.methode.isin(["Coverage-Greedy", "Trajectory-Greedy", "ILP", "Dueling DQN"])].set_index("methode")

# 1) training of the Dueling DQN: greedy evaluations every 25 episodes
ev = pd.read_csv(T + "kaggle_run_wW5_dueling_gapfix/session2/eval_log.csv")
tr = pd.read_csv(T + "kaggle_run_wW5_dueling_gapfix/session2/train_log.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(tr.episode, tr.n_white.rolling(25).mean(), color="#8A94A6", lw=1, label="training (moving avg. 25)")
ax[0].plot(ev.episode, ev.n_white, "o-", ms=3, color="#F26B5B", label="greedy evaluation")
ax[0].set_ylabel("white points")
ax[1].plot(ev.episode, ev.cout_eur / 1e6, "o-", ms=3, color="#F26B5B", label="Dueling DQN (greedy evaluation)")
ax[1].axhline(0.87, ls="--", color=COUL["ILP"], label="ILP 0.87 M€")
ax[1].set_ylabel("cost (M€)"); ax[1].set_ylim(0, 10.5)
for a in ax:
    a.axvspan(700, 800, color="#0F9D9A", alpha=0.12, label="full coverage (ep. 700-800)")
    a.axvline(725, color="k", lw=0.8, ls=":")
    a.set_xlabel("episode"); a.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "training_dueling_gapfix.png", dpi=200); plt.close(fig)

# 2) cost against mean QoS and worst QoS
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, col, lab in [(ax[0], "q_mean", "mean QoS $\\bar q$"), (ax[1], "q_min", "worst QoS $q_{min}$")]:
    for m, r in full.iterrows():
        a.scatter(r.cout / 1e6, r[col], s=90, color=COUL[m], zorder=3, label=m)
        a.annotate(f"{r[col]:.3f}", (r.cout / 1e6, r[col]), xytext=(6, 6), textcoords="offset points", fontsize=8)
    a.set_xlabel("deployment cost (M€)"); a.set_ylabel(lab)
ax[1].axhline(0.5, ls="--", color="grey", lw=0.8); ax[1].text(1.6, 0.505, "critical QoS 0.5", fontsize=8, color="grey")
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "cost_vs_qos_gapfix.png", dpi=200); plt.close(fig)

# 3) white points against cumulative cost
du = pd.read_csv(PLANS + "dueling_ep725.csv")
du = du[du.action != "STOP"]
tg = pd.read_csv(PLANS + "trajectory_greedy.csv")
cg_cout = [0, 145, 290, 435, 580, 725, 870, 1015]          # fullspace console.log, CG sites 1-7
cg_white = [219, 117, 61, 27, 13, 7, 3, 0]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.step([0] + list(du.cout_eur / 1e6), [219] + list(du.n_white), where="post", color=COUL["Dueling DQN"], lw=2, label="Dueling DQN (ep. 725)")
ax.step([0] + list(tg.cout.cumsum() / 1e6), [219] + list(tg.n_white), where="post", color=COUL["Trajectory-Greedy"], lw=2, label="Trajectory-Greedy")
ax.step(np.array(cg_cout) / 1e3, cg_white, where="post", color=COUL["Coverage-Greedy"], lw=2, label="Coverage-Greedy")
ax.scatter([0.87], [0], s=110, marker="D", color=COUL["ILP"], zorder=4, label="ILP (final plan, 0.87 M€)")
ax.set_xlabel("cumulative cost (M€)"); ax.set_ylabel("white points"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "white_vs_cost_gapfix.png", dpi=200); plt.close(fig)

# 4) rank of each method on each criterion (1 = best, ties share the best rank)
crit = {"Cost": ("cout", True), "$\\bar q$": ("q_mean", False), "$q_{min}$": ("q_min", False),
        "$N_H$": ("N_H_cut", True), "$N_T$": ("N_T", True), "Return": ("return_ref", False)}
rangs = pd.DataFrame({k: full[c].round(3).rank(method="min", ascending=asc).astype(int) for k, (c, asc) in crit.items()})
rangs["worst rank"] = rangs.max(axis=1)
fig, ax = plt.subplots(figsize=(8, 3))
ax.imshow(rangs.values, cmap="RdYlGn_r", vmin=1, vmax=4, aspect="auto")
ax.set_xticks(range(rangs.shape[1]), rangs.columns); ax.set_yticks(range(len(rangs)), rangs.index)
for i in range(rangs.shape[0]):
    for j in range(rangs.shape[1]):
        ax.text(j, i, rangs.values[i, j], ha="center", va="center", fontsize=11)
ax.spines[:].set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "ranks_gapfix.png", dpi=200); plt.close(fig)
rangs.to_csv(OUT + "ranks_gapfix.csv")

# 5) cumulative computation time (indicative: GPU training against CPU search)
train_h = tr[tr.episode <= 725].duree_s.sum() / 3600
n = np.arange(0, 121)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(n, 12 * n / 60, color=COUL["Trajectory-Greedy"], lw=2, label="Trajectory-Greedy (12 min / plan)")
ax.plot(n, 6 * n / 60, color=COUL["ILP"], lw=2, label="ILP (6 min / plan, after coverage precompute)")
ax.plot(n, train_h + 0 * n, color=COUL["Dueling DQN"], lw=2, label=f"Dueling DQN ({train_h:.1f} h training to ep. 725, < 1 s / plan)")
be = train_h * 60 / 12
ax.axvline(be, color="grey", ls="--"); ax.text(be + 1, 1, f"break-even with TG ≈ {be:.0f} plans", fontsize=8)
ax.set_xlabel("number of plans"); ax.set_ylabel("cumulative time (h)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "break_even_gapfix.png", dpi=200); plt.close(fig)

# 6) budget experiment
ex = pd.read_csv(R + "experiences_ep725/evaluation_dueling_softmax_wW5_gapfix_seed0.csv")
bu = ex[ex.experience.isin(["budget", "final"])].copy()
bu["B"] = bu.configuration.str.extract(r"B = (\d+)").astype(float)
bu = bu.sort_values("B")
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.bar(bu.B, bu.n_white, color=COUL["Dueling DQN"], width=0.7, label="white points left")
ax.set_xlabel("budget B (M€)"); ax.set_ylabel("white points left")
ax2 = ax.twinx(); ax2.plot(bu.B, bu.cout_eur / 1e6, "ko-", label="cost spent (M€)"); ax2.set_ylabel("cost spent (M€)")
ax2.spines["top"].set_visible(False)
ax.axvspan(7, 13, color="#0F9D9A", alpha=0.08); ax.text(8.3, max(bu.n_white) * 0.9, "near training budget", fontsize=8)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc=(0.6, 0.5))
fig.tight_layout(); fig.savefig(OUT + "budget_gapfix.png", dpi=200); plt.close(fig)

# 7) reward-weight sensitivity: best method per cost weight
se = pd.read_csv(R + "sensibilite_poids/sensibilite_poids.csv")
tab = se.groupby(["R_eff", "meilleur"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(7, 3.6))
bas = np.zeros(len(tab))
for m in tab.columns:
    nom = "Dueling DQN" if m.startswith("Dueling") else m
    ax.bar(tab.index.astype(str), tab[m], bottom=bas, color=COUL.get(nom, "grey"), label=nom)
    bas += tab[m].values
ax.set_xlabel("cost weight $R_{eff}$ (reference 0.2)"); ax.set_ylabel("weight combinations won (out of 81)")
ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout(); fig.savefig(OUT + "sensitivity_gapfix.png", dpi=200); plt.close(fig)

print(rangs.to_string()); print("training h to ep725", round(train_h, 2), "break-even", round(be, 1))
