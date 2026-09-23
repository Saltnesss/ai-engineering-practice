"""Sequential Monte Carlo (particle filter) — a robot on a 1-D line.
Tangent from Lesson 1.7. Not in the course; asked for 2026-09-22."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
T, N = 30, 1000              # 30 time steps, 1000 particles
move_noise, sensor_noise = 0.5, 1.0

# ---- the truth (hidden from the filter) ----
true_x = np.zeros(T); true_x[0] = 5.0
for t in range(1, T):
    true_x[t] = true_x[t-1] + 1.0 + rng.normal(0, move_noise)   # robot tries to move +1 each step
readings = true_x + rng.normal(0, sensor_noise, T)              # noisy sensor: distance from the wall

# ---- the filter ----
particles = rng.uniform(0, 20, N)     # step 0: prior = "somewhere between 0 and 20", 1000 guesses
history = []
for t in range(T):
    # 1. predict: move every particle the way the robot moved, plus noise
    if t > 0:
        particles = particles + 1.0 + rng.normal(0, move_noise, N)
    # 2. weight: likelihood of the sensor reading under each particle  (Normal PDF, 1.6)
    w = np.exp(-0.5 * ((readings[t] - particles) / sensor_noise) ** 2)
    w /= w.sum()
    # 3. resample: draw N particles with probability = weight (good guesses copied, bad ones dropped)
    particles = rng.choice(particles, size=N, p=w)
    history.append(particles.copy())

est = np.array([h.mean() for h in history]); spread = np.array([h.std() for h in history])
for t in [0, 1, 2, 5, 10, 29]:
    print(f"t={t:>2}  true={true_x[t]:6.2f}  reading={readings[t]:6.2f}  estimate={est[t]:6.2f} ± {spread[t]:.2f}")

fig, ax = plt.subplots(figsize=(10, 4.5))
for t, h in enumerate(history):
    ax.scatter(np.full(N, t), h, s=1, c="tab:blue", alpha=0.05)
ax.plot(true_x, "k-", lw=2, label="true position")
ax.plot(readings, "r.", label="sensor reading (noisy)")
ax.plot(est, "b-", lw=1.5, label="particle mean")
ax.set_xlabel("time step"); ax.set_ylabel("position"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Particle filter: blue cloud = posterior over position, one update per reading")
plt.tight_layout(); plt.savefig("/Users/marctub/Documents/AI-engineering/practice/phase1/particle_filter.png", dpi=110)
