import numpy as np
import matplotlib.pyplot as plt
from P1L8E2 import SGDMomentum, rosenbrock_gradient

xs, ys = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-1, 3, 200))
Z = (1 - xs) ** 2 + 100 * (ys - xs ** 2) ** 2

# path of momentum β=0.99 from Exercise 2 (first 1500 steps)
opt, p, path = SGDMomentum(lr=0.0001, momentum=0.99), [-1.0, 1.0], []
for _ in range(1500):
    path.append(p); p = opt.step(p, rosenbrock_gradient(p))
path = np.array(path)
pz = (1 - path[:, 0]) ** 2 + 100 * (path[:, 1] - path[:, 0] ** 2) ** 2

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(xs, ys, Z, cmap="viridis", linewidth=0, alpha=0.9)
ax1.set_title("Rosenbrock f(x, y), raw height"); ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f")
ax1.view_init(elev=30, azim=-120)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(xs, ys, np.log10(Z + 1e-2), cmap="viridis", linewidth=0, alpha=0.7)
ax2.plot(path[:, 0], path[:, 1], np.log10(pz + 1e-2), "r-", lw=2, label="momentum β=0.99 path")
ax2.scatter([1], [1], [np.log10(1e-2)], color="red", s=60, marker="*")
ax2.set_title("log₁₀ f — shows the curved valley floor"); ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("log₁₀ f")
ax2.view_init(elev=45, azim=-120); ax2.legend()

plt.tight_layout(); plt.savefig("rosenbrock_3d.png", dpi=110)
plt.show()   # opens an interactive window: drag with the mouse to rotate
