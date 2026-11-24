"""
Generates visual figures for the Critical Validation and Experiment Plan.
Saved under reports/model_performance_audit/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path("D:/driver ml model/driver-monitoring-agent")
AUDIT_DIR = REPO_ROOT / "reports" / "model_performance_audit"
RESULTS_JSON = AUDIT_DIR / "task2_validation_investigation_results.json"

with open(RESULTS_JSON) as f:
    res = json.load(f)

# Figure 7: Session Confounding & Mounting Shift
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Session distribution across splits
split_data = [
    {"Split": "Train", "Session": "Session 1", "Normal": 1951, "Aggressive": 3268},
    {"Split": "Train", "Session": "Session 2", "Normal": 1791, "Aggressive": 2757},
    {"Split": "Val", "Session": "Session 1", "Normal": 471, "Aggressive": 568},
    {"Split": "Val", "Session": "Session 2", "Normal": 346, "Aggressive": 550},
    {"Split": "Test", "Session": "Session 1", "Normal": 1141, "Aggressive": 0},
    {"Split": "Test", "Session": "Session 2", "Normal": 0, "Aggressive": 997},
]
df_splits = pd.DataFrame(split_data)
df_splits["Total"] = df_splits["Normal"] + df_splits["Aggressive"]
df_splits["Agg_Pct"] = df_splits["Aggressive"] / df_splits["Total"] * 100

sns.barplot(data=df_splits, x="Split", y="Agg_Pct", hue="Session", palette=["#2b5c8f", "#d95f02"], ax=axes[0])
axes[0].set_title("Aggressive Driving Proportion by Session & Split\n(Note Complete Segregation in Test Set: 0% vs 100%)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("% Aggressive Samples")
axes[0].set_ylim(0, 110)
for p in axes[0].patches:
    height = p.get_height()
    if height > 0:
        axes[0].annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height + 2),
                         ha='center', va='bottom', fontsize=9)

# Plot 2: 3D Mean Gravity Orientation Vector Shift
g1 = res["task1_session_confounding"]["sensor_shift"]["gravity_s1"]
g2 = res["task1_session_confounding"]["sensor_shift"]["gravity_s2"]
tilt = res["task1_session_confounding"]["sensor_shift"]["tilt_angle_degrees"]

axes[1].bar(["Acc X", "Acc Y", "Acc Z"], g1, width=0.35, label="Session 1 Mean (5:14 PM)", color="#2b5c8f", align="center")
axes[1].bar([x + 0.35 for x in range(3)], g2, width=0.35, label="Session 2 Mean (6:45 PM)", color="#d95f02", align="center")
axes[1].set_xticks([x + 0.175 for x in range(3)])
axes[1].set_xticklabels(["Acc X", "Acc Y", "Acc Z"])
axes[1].set_title(f"Smartphone Mean Acceleration Shift Between Sessions\n(Physical Mounting Tilt Shift = {tilt:.2f}°)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Mean Acceleration (g)")
axes[1].legend(loc="upper left")

plt.tight_layout()
fig7_path = AUDIT_DIR / "07_dataset_c_session_confounding.png"
fig.savefig(fig7_path, dpi=300)
plt.close()
print(f"Saved: {fig7_path.name}")

# Figure 8: Validation Threshold Optimization Trade-Off Curve
df_thresh = pd.DataFrame(res["task4_threshold_optimization"]["full_threshold_sweep"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(df_thresh["threshold"], df_thresh["aggressive_recall"] * 100, label="Aggressive Recall (%)", color="dodgerblue", lw=2)
axes[0].plot(df_thresh["threshold"], df_thresh["fpr"] * 100, label="False Positive Rate (%)", color="crimson", lw=2)
axes[0].plot(df_thresh["threshold"], df_thresh["macro_f1"] * 100, label="Macro F1-Score (%)", color="green", lw=2)
axes[0].axvline(0.84, color="black", linestyle="--", label="Current Threshold (tau=0.84)")
axes[0].axhline(80.0, color="dodgerblue", linestyle=":", alpha=0.7, label="Recall Target >= 80%")
axes[0].axhline(15.0, color="crimson", linestyle=":", alpha=0.7, label="FPR Target <= 15%")
axes[0].set_title("Validation Threshold Sweep: Precision-Recall Trade-Off", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Decision Threshold (tau)")
axes[0].set_ylabel("Metric Value (%)")
axes[0].set_xlim(0.1, 0.95)
axes[0].legend(loc="lower left", fontsize=8)

# Trade-off curve: FPR vs Recall
axes[1].plot(df_thresh["fpr"] * 100, df_thresh["aggressive_recall"] * 100, marker='o', markersize=3, color="purple", lw=2)
axes[1].scatter([49.20], [89.98], color="black", s=100, zorder=5, label="tau=0.84 (FPR=49.2%, Rec=90.0%)")
axes[1].fill_between([0, 15], 80, 100, color="green", alpha=0.2, label="Desired Target Zone (FPR<=15%, Rec>=80%)")
axes[1].set_title("FPR vs. Aggressive Recall Operating Characteristic\n(Target Zone is Mathematically Unreachable)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("False Positive Rate (%)")
axes[1].set_ylabel("Aggressive Recall (%)")
axes[1].set_xlim(35, 75)
axes[1].set_ylim(60, 100)
axes[1].legend(loc="lower right", fontsize=8)

plt.tight_layout()
fig8_path = AUDIT_DIR / "08_threshold_optimization_pareto.png"
fig.savefig(fig8_path, dpi=300)
plt.close()
print(f"Saved: {fig8_path.name}")

# Figure 9: Pipeline B Independently Reproduced LODO
lodo_folds = res["task5_lodo_reproduction"]["folds"]
lodo_summary = res["task5_lodo_reproduction"]["summary"]

fig, ax = plt.subplots(figsize=(8, 5))
drivers = list(lodo_folds.keys())
ev_f1s = [lodo_folds[d]["event_metrics"]["macro_f1"] * 100 for d in drivers]
ev_accs = [lodo_folds[d]["event_metrics"]["accuracy"] * 100 for d in drivers]
samp_f1s = [lodo_folds[d]["sample_metrics"]["macro_f1"] * 100 for d in drivers]

x = np.arange(len(drivers))
w = 0.25
ax.bar(x - w, ev_f1s, w, label="Event Macro-F1 (%)", color="#1b9e77")
ax.bar(x, ev_accs, w, label="Event Accuracy (%)", color="#7570b3")
ax.bar(x + w, samp_f1s, w, label="Sample Macro-F1 (%)", color="#e7298a")

ax.axhline(85.0, color="red", linestyle="--", label="85% Benchmark Target")
ax.set_xticks(x)
ax.set_xticklabels(drivers)
ax.set_ylim(50, 105)
ax.set_title(f"Independently Reproduced Pipeline B LODO Metrics\n(Overall Mean Event Macro-F1 = {lodo_summary['mean_event_macro_f1']*100:.2f}%)", fontsize=11, fontweight="bold")
ax.set_ylabel("Score (%)")
ax.legend(loc="lower left")

plt.tight_layout()
fig9_path = AUDIT_DIR / "09_pipeline_b_lodo_reproduced.png"
fig.savefig(fig9_path, dpi=300)
plt.close()
print(f"Saved: {fig9_path.name}")

# Figure 10: Safety Agent Temporal Mitigation Simulation
sim_test = res["task6_safety_agent_simulation"]["test_simulation"]
rules = [v["confirmation_rule"].replace("Consecutive ", "").replace(" frames", "f") for v in sim_test.values()]
caution_pcts = [v["time_in_caution_pct"] * 100 for v in sim_test.values()]
false_rates = [v["false_caution_rate_on_normal"] * 100 for v in sim_test.values()]
agg_recs = [v["aggressive_detection_recall"] * 100 for v in sim_test.values()]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(rules, false_rates, marker='o', lw=2, color="crimson", label="False Alert Rate on Normal (%)")
axes[0].plot(rules, caution_pcts, marker='s', lw=2, color="orange", label="Total Time in CAUTION (%)")
axes[0].set_title("Safety Agent False Alert Reduction vs. Dwell Time\n(Dataset C Test Split)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Confirmation Rule")
axes[0].set_ylabel("Percentage (%)")
axes[0].tick_params(axis='x', rotation=25)
axes[0].legend(loc="upper right")

axes[1].plot(false_rates, agg_recs, marker='o', lw=2, color="purple")
for idx, r in enumerate(rules):
    axes[1].annotate(r, (false_rates[idx] + 0.5, agg_recs[idx] - 0.5), fontsize=8)
axes[1].set_title("False Alert Rate vs. Aggressive Detection Recall Trade-Off", fontsize=11, fontweight="bold")
axes[1].set_xlabel("False Alert Rate on Normal (%)")
axes[1].set_ylabel("Aggressive Detection Recall (%)")

plt.tight_layout()
fig10_path = AUDIT_DIR / "10_safety_agent_temporal_mitigation.png"
fig.savefig(fig10_path, dpi=300)
plt.close()
print(f"Saved: {fig10_path.name}")

print("All 4 validation figures generated successfully!")
