import math, random

def run(schedule, seed, T=5000, lr0=0.01):
    random.seed(seed)
    x, m, v = 1.0, 0.0, 0.0
    for t in range(1, T + 1):
        g = 2 * x + random.gauss(0, 1.0)      # true gradient of x², plus mini-batch noise (std 1)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
        lr = lr0 if schedule == "fixed" else 0.5 * lr0 * (1 + math.cos(math.pi * t / T))
        x -= lr * mh / (math.sqrt(vh) + 1e-8)
    return x

# one run's final x is itself random, so average over 200 runs: RMS distance from the minimum x = 0
for schedule in ("fixed", "cosine"):
    finals = [run(schedule, s) for s in range(200)]
    rms = math.sqrt(sum(x * x for x in finals) / len(finals))
    print(f"{schedule:6s}: typical distance from minimum after 5000 steps = {rms:.4f}")
