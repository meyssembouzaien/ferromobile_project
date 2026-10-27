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
COUL = {"Existing network": "#C3C9D3", "Coverage-Greedy": "#A9BCD6", "Trajectory-Greedy": "#6A8CC7",
        "ILP": "#1F3A6E", "Dueling DQN": "#C2185B", "Vanilla DQN": "#00897B"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

cc = pd.read_csv(R + "criteres_communs/criteres_communs.csv")
cc["methode"] = cc.methode.replace({"Dueling DQN ep725": "Dueling DQN", "Vanilla DQN ep550": "Vanilla DQN"})
full = cc[cc.methode.isin(["Coverage-Greedy", "Trajectory-Greedy", "ILP", "Dueling DQN"])].set_index("methode")
comp = cc[cc.methode.isin(list(full.index) + ["Vanilla DQN"])].set_index("methode")   # + Vanilla (not full coverage)

# 1) training of the Dueling DQN: greedy evaluations every 25 episodes
ev = pd.read_csv(T + "kaggle_run_wW5_dueling_gapfix/session2/eval_log.csv")
tr = pd.read_csv(T + "kaggle_run_wW5_dueling_gapfix/session2/train_log.csv")
ev, tr = ev[ev.episode <= 725], tr[tr.episode <= 725]       # final agent = episode 725
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(tr.episode, tr.n_white.rolling(25).mean(), color="#9AA3B2", lw=1, label="training (moving avg. 25)")
ax[0].plot(ev.episode, ev.n_white, "o-", ms=3, color=COUL["Dueling DQN"], label="evaluation without exploration")
ax[0].set_ylabel("white points")
ax[1].plot(ev.episode, ev.cout_eur / 1e6, "o-", ms=3, color=COUL["Dueling DQN"], label="Dueling DQN (evaluation without exploration)")
ax[1].axhline(0.87, ls="--", color=COUL["ILP"], label="ILP 0.87 M€")
ax[1].set_ylabel("cost (M€)"); ax[1].set_ylim(0, 10.5)
for a in ax:
    a.axvline(725, color="k", lw=0.8, ls=":")
    a.annotate("final agent (ep. 725)", (725, a.get_ylim()[1] * 0.9), xytext=(-8, 0), textcoords="offset points", ha="right", fontsize=8)
    a.set_xlim(0, 740); a.set_xlabel("episode"); a.legend(frameon=False, fontsize=8, loc="center right" if a is ax[0] else (0.03, 0.12))
fig.tight_layout(); fig.savefig(OUT + "training_dueling_gapfix.png", dpi=200); plt.close(fig)

# 2) cost against mean QoS and worst QoS
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, col, lab in [(ax[0], "q_mean", "mean QoS $\\bar q$"), (ax[1], "q_min", "worst QoS $q_{min}$")]:
    for m, r in comp.iterrows():
        duel, van = m == "Dueling DQN", m == "Vanilla DQN"
        a.scatter(r.cout / 1e6, r[col], s=260 if duel else 90, marker="*" if duel else ("s" if van else "o"), color=COUL[m], zorder=3,
                  label=f"{m} ({r.n_white:.0f} white points left)" if van else m)
        dx, dy, ha = ((-8, 6, "right") if col == "q_mean" else (0, -16, "center")) if m == "ILP" else (6, 6, "left")     # ILP and CG share values: labels on both sides
        a.annotate(f"{r[col]:.3f}", (r.cout / 1e6, r[col]), xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha)
    a.set_xscale("log"); a.set_xlim(0.7, 13); a.set_xticks([0.8, 1, 1.5, 2, 3, 5, 10], ["0.8", "1", "1.5", "2", "3", "5", "10"]); a.minorticks_off()
    a.set_xlabel("deployment cost (M€, log scale)"); a.set_ylabel(lab)
ax[1].axhline(0.5, ls="--", color="grey", lw=0.8); ax[1].text(3, 0.515, "critical QoS 0.5", fontsize=8, color="grey")
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "cost_vs_qos_gapfix.png", dpi=200); plt.close(fig)

# 3) white points against cumulative cost
du = pd.read_csv(PLANS + "dueling_ep725.csv")
du = du[du.action != "STOP"]
tg = pd.read_csv(PLANS + "trajectory_greedy.csv")
cg_cout = [0, 145, 290, 435, 580, 725, 870, 1015]          # fullspace console.log, CG sites 1-7
cg_white = [219, 117, 61, 27, 13, 7, 3, 0]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.step([0] + list(du.cout_eur / 1e6), [219] + list(du.n_white), where="post", color=COUL["Dueling DQN"], lw=3, zorder=5, label="Dueling DQN (ep. 725)")
ax.step([0] + list(tg.cout.cumsum() / 1e6), [219] + list(tg.n_white), where="post", color=COUL["Trajectory-Greedy"], lw=2, label="Trajectory-Greedy")
ax.step(np.array(cg_cout) / 1e3, cg_white, where="post", color=COUL["Coverage-Greedy"], lw=2, label="Coverage-Greedy")
ax.scatter([0.87], [0], s=110, marker="D", color=COUL["ILP"], zorder=4, label="ILP (final plan, 0.87 M€)")
vt = pd.read_csv(T + "kaggle_run_wW5_vanilla_gapfix/session2/eval_trace_ep550.csv")
vt = vt[vt.action != "STOP"]
ax.step([0] + list(vt.cout_eur / 1e6), [219] + list(vt.n_white), where="post", color=COUL["Vanilla DQN"], lw=2, ls="--",
        label=f"Vanilla DQN (best evaluation: {vt.n_white.iloc[-1]} left at {vt.cout_eur.iloc[-1] / 1e6:.2f} M€)")
ax.set_xlim(-0.05, 2.5)
ax.set_xlabel("cumulative cost (M€)"); ax.set_ylabel("white points"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "white_vs_cost_gapfix.png", dpi=200); plt.close(fig)

# 4) rank of each method on each criterion (1 = best, ties share the best rank)
crit = {"White pts": ("n_white", True), "Cost": ("cout", True), "$\\bar q$": ("q_mean", False), "$q_{min}$": ("q_min", False),
        "$N_H$": ("N_H_cut", True), "$N_T$": ("N_T", True), "Return": ("return_ref", False)}
rangs = pd.DataFrame({k: comp[c].round(3).rank(method="min", ascending=asc).astype(int) for k, (c, asc) in crit.items()})
rangs["Worst rank"] = rangs.max(axis=1)
rangs = rangs.loc[["Dueling DQN", "ILP", "Trajectory-Greedy", "Coverage-Greedy", "Vanilla DQN"]]
FOND = {1: "#DCE6F2", 2: "#B9CCE5", 3: "#8EA9D1", 4: "#5B7DB1", 5: "#2E4A7D"}
fig, ax = plt.subplots(figsize=(9.5, 3.4))
for i, (m, r) in enumerate(rangs.iterrows()):
    for j, v in enumerate(r.values):
        x = j + (0.25 if j == rangs.shape[1] - 1 else 0)
        ax.add_patch(plt.Rectangle((x, i), 0.94, 0.88, color=FOND[v], lw=0))
        ax.text(x + 0.47, i + 0.44, v, ha="center", va="center", fontsize=12,
                fontweight="bold" if v == 1 else "normal", color="white" if v >= 4 else "#1F3A6E")
ax.set_xlim(0, rangs.shape[1] + 0.25); ax.set_ylim(len(rangs), -0.35)
ax.set_xticks([j + 0.47 + (0.25 if j == rangs.shape[1] - 1 else 0) for j in range(rangs.shape[1])], rangs.columns)
ax.set_yticks([i + 0.44 for i in range(len(rangs))], rangs.index)
ax.tick_params(length=0); ax.xaxis.tick_top()
ax.get_yticklabels()[0].set_color(COUL["Dueling DQN"]); ax.get_yticklabels()[0].set_fontweight("bold")
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
ax.axvspan(7, 13, color="#00897B", alpha=0.08); ax.text(8.3, max(bu.n_white) * 0.9, "near training budget", fontsize=8)
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

# 8) ablation: Dueling (final agent, ep. 725) against Vanilla DQN (same settings, 1000 episodes)
va = pd.read_csv(T + "kaggle_run_wW5_vanilla_gapfix/session2/eval_log.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, col, lab in [(ax[0], "n_white", "white points (evaluation every 25 episodes)"), (ax[1], "cout_eur", "cost (M€)")]:
    k = 1e6 if col == "cout_eur" else 1
    a.plot(ev.episode, ev[col] / k, "o-", ms=3, color=COUL["Dueling DQN"], label="Dueling DQN (to final agent, ep. 725)")
    a.plot(va.episode, va[col] / k, "s-", ms=3, color=COUL["Vanilla DQN"], label="Vanilla DQN")
    a.set_xlabel("episode"); a.set_ylabel(lab)
ax[1].set_ylim(0, 10.5)
fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=9, ncol=2, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT + "ablation_vanilla_gapfix.png", dpi=200); plt.close(fig)

print(rangs.to_string()); print("training h to ep725", round(train_h, 2), "break-even", round(be, 1))

# 9) generalisation: one operator removed, trained policy (no retraining) against the ILP recomputed
ge = pd.read_csv(R + "generalisation_operateurs/comparaison_generalisation.csv")
ge["op"] = ge.config.map({"sans ORANGE": "No Orange", "sans SFR": "No SFR", "sans BOUYGUES TELECOM": "No Bouygues", "sans FREE MOBILE": "No Free"})
ops = list(dict.fromkeys(ge.op))
x = np.arange(len(ops)); w = 0.27
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
avant = ge.groupby("op", sort=False).n_avant.first().values
du_w = ge[ge.methode == "Dueling ep725"].n_white.values
ilp_w = ge[ge.methode == "ILP"].n_white.values
ax[0].bar(x - w, avant, w, color=COUL["Existing network"], label="existing network")
ax[0].bar(x, du_w, w, color=COUL["Dueling DQN"], label="Dueling DQN (ep. 725, no retraining)")
ax[0].bar(x + w, ilp_w, w, color=COUL["ILP"], label="ILP (recomputed)")
for i, (a, d) in enumerate(zip(avant, du_w)):
    ax[0].text(i, d + 5, f"{100 * (a - d) / a:.0f}%", ha="center", fontsize=8)
    ax[0].text(i + w, 5, "0", ha="center", fontsize=8, color=COUL["ILP"])
ax[0].set_ylabel("white points"); ax[0].set_xticks(x, ops)
ax[1].bar(x - w / 2, ge[ge.methode == "Dueling ep725"].cout.values, w, color=COUL["Dueling DQN"])
ax[1].bar(x + w / 2, ge[ge.methode == "ILP"].cout.values, w, color=COUL["ILP"])
ax[1].set_ylabel("cost (M€)"); ax[1].set_xticks(x, ops)
fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=9, ncol=3, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT + "generalisation_gapfix.png", dpi=200); plt.close(fig)
