import math, random
import matplotlib.pyplot as plt

def trajectory(schedule, seed=0, T=5000, lr0=0.01):
    random.seed(seed)
    x, m, v, xs = 1.0, 0.0, 0.0, []
    for t in range(1, T + 1):
        g = 2 * x + random.gauss(0, 1.0)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
        lr = lr0 if schedule == "fixed" else 0.5 * lr0 * (1 + math.cos(math.pi * t / T))
        x -= lr * mh / (math.sqrt(vh) + 1e-8)
        xs.append(x)
    return xs

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for ax, schedule in zip(axes, ("fixed", "cosine")):
    xs = trajectory(schedule)
    ax.plot(range(1000, 5000), xs[1000:], lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-0.15, 0.15); ax.set_ylabel("x  (minimum at 0)")
    ax.set_title(f"Adam on noisy x², {schedule} lr — steps 1000–5000")
axes[1].set_xlabel("training step")
plt.tight_layout(); plt.savefig("adam_decay_trajectory.png", dpi=110)
