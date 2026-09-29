import math
import numpy as np
import matplotlib.pyplot as plt

# 100 days: X = clouds (yes/no), Y = rain (yes/no)
joint = np.array([[0.45, 0.05],
                  [0.05, 0.45]])
px = joint.sum(axis=1)             # row sums    -> p(x)
py = joint.sum(axis=0)             # column sums -> p(y)
indep = np.outer(px, py)           # what the table would be if X and Y were independent
contrib = joint * np.log2(joint / indep)   # each cell's share of MI

for i in range(2):
    for j in range(2):
        print(f"cell ({i},{j}): actual {joint[i,j]:.2f}  if-independent {indep[i,j]:.2f}  "
              f"ratio {joint[i,j]/indep[i,j]:.1f}  contribution {contrib[i,j]:+.4f} bits")
print(f"MI = sum of contributions = {contrib.sum():.4f} bits")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
titles = ["Actual joint p(x,y)", "If independent: p(x)·p(y)", "Contribution per cell (bits)"]
labels_x, labels_y = ["clouds", "no clouds"], ["rain", "no rain"]
for ax, data, title, cmap in zip(axes, [joint, indep, contrib], titles, ["Blues", "Blues", "RdBu"]):
    vmax = abs(data).max()
    im = ax.imshow(data, cmap=cmap, vmin=(-vmax if cmap == "RdBu" else 0), vmax=vmax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{data[i,j]:+.3f}" if cmap == "RdBu" else f"{data[i,j]:.2f}",
                    ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], labels_y); ax.set_yticks([0, 1], labels_x); ax.set_title(title)
fig.suptitle(f"Mutual information = {contrib.sum():.3f} bits")
plt.tight_layout()
plt.savefig("mi_step5.png", dpi=110)
