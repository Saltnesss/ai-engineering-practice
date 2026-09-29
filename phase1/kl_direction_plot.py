import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-8, 8, 801)
dx = x[1] - x[0]


def normal(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# Truth P: two peaks (e.g. "cat" vs "dog")
P = 0.5 * normal(x, -3, 0.8) + 0.5 * normal(x, 3, 0.8)
eps = 1e-300


def kl(a, b):
    return np.sum(a * np.log((a + eps) / (b + eps))) * dx


# Model Q can only be ONE bell curve; search for the best mu, sigma under each KL
best_fwd, best_rev = None, None
for mu in np.linspace(-5, 5, 101):
    for sigma in np.linspace(0.3, 5, 95):
        Q = normal(x, mu, sigma)
        f, r = kl(P, Q), kl(Q, P)
        if best_fwd is None or f < best_fwd[0]:
            best_fwd = (f, mu, sigma)
        if best_rev is None or r < best_rev[0]:
            best_rev = (r, mu, sigma)

print(f"best under KL(P||Q): mu={best_fwd[1]:.2f}, sigma={best_fwd[2]:.2f}")
print(f"best under KL(Q||P): mu={best_rev[1]:.2f}, sigma={best_rev[2]:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (name, best, color) in zip(
    axes,
    [("KL(P‖Q): mass-covering", best_fwd, "tab:orange"),
     ("KL(Q‖P): mode-seeking", best_rev, "tab:green")],
):
    ax.fill_between(x, P, alpha=0.3, color="tab:blue", label="truth P (two peaks)")
    ax.plot(x, normal(x, best[1], best[2]), color=color, lw=2.5, label="best single-peak Q")
    ax.set_title(name)
    ax.legend()
plt.tight_layout()
plt.savefig("kl_direction.png", dpi=110)
