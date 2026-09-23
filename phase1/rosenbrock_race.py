import sys
sys.path.insert(0, "/Users/marctub/Documents/AI-engineering/ai-engineering-from-scratch/phases/01-math-foundations/08-optimization/code")
import numpy as np
import matplotlib.pyplot as plt
from optimizers import rosenbrock, rosenbrock_gradient, GradientDescent, SGDMomentum, Adam, optimize

start = [-1.0, 1.0]
runs = {
    "GD (lr=0.0005)":        optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start),
    "SGD+M (lr=0.0001, β=0.9)": optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start),
    "Adam (lr=0.01)":        optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start),
}

xs, ys = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-0.5, 1.5, 300))
Z = (1 - xs) ** 2 + 100 * (ys - xs ** 2) ** 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.contourf(xs, ys, np.log10(Z + 1e-3), levels=40, cmap="viridis")
ax1.plot(xs[0], xs[0] ** 2, "w--", lw=0.8, label="valley floor y = x²")
for name, h in runs.items():
    h = np.array(h); ax1.plot(h[:, 0], h[:, 1], lw=1.5, label=name)
ax1.plot(1, 1, "r*", ms=14); ax1.plot(-1, 1, "wo")
ax1.set_title("Rosenbrock (colour = log₁₀ loss) and the three paths"); ax1.legend(fontsize=8)
for name, h in runs.items():
    ax2.semilogy([rosenbrock(p) + 1e-12 for p in h], label=name)
ax2.set_xlabel("step"); ax2.set_ylabel("loss (log scale)"); ax2.set_title("loss per step"); ax2.legend()
plt.tight_layout(); plt.savefig("rosenbrock_race.png", dpi=110)
