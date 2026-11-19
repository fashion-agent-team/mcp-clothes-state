"""Chart for the confidence sweep: recall (stain, torn) and clean photos flagged vs threshold."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))["by_conf"]
c = [r["conf"] for r in d]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
ax.plot(c, [r["stain_recall"] for r in d], "o-", color="#c0392b", label="stain recall (test)")
ax.plot(c, [r["torn_recall"] for r in d], "s-", color="#2471a3", label="torn recall (test)")
ax.plot(c, [r["stain_precision"] for r in d], "o--", color="#c0392b", alpha=0.5, label="stain precision")
ax.plot(c, [r["torn_precision"] for r in d], "s--", color="#2471a3", alpha=0.5, label="torn precision")
ax.set_xlabel("confidence threshold"); ax.set_ylabel("recall / precision"); ax.set_ylim(0, 1)
ax2 = ax.twinx()
ax2.bar(c, [r["clean_imgs_flagged"] / 2 for r in d], width=0.03, color="#7f8c8d", alpha=0.35, label="clean photos flagged (%)")
ax2.set_ylabel("clean photos flagged (% of 200)"); ax2.set_ylim(0, 40)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
ax.set_title("stain (v4_neg) + torn (v2): threshold sweep")
fig.tight_layout(); fig.savefig(sys.argv[2])
