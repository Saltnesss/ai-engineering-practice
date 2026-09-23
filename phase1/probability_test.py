"""Phase 1 · Lesson 6 — Probability & Distributions, Build It."""

import math
import random


# ---- Step 1: probability basics -------------------------------------------
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b


p_king_given_face = conditional_probability(4 / 52, 12 / 52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
print(f"C(52, 5) = {combinations(52, 5)}")


# ---- Step 2: PMF and PDF from scratch -------------------------------------
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)


def categorical_pmf(k, probs):
    return probs[k]


def poisson_pmf(k, lam):
    return (lam**k) * math.exp(-lam) / factorial(k)


def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


print("Poisson λ=3:", [f"P({k})={poisson_pmf(k, 3):.4f}" for k in range(7)])
print(f"normal_pdf(0, 0, 0.1) = {normal_pdf(0, 0, 0.1):.4f}   (a density, not a probability)")


# ---- Step 3: expected value and variance ----------------------------------
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))


def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))


die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1 / 6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")


# ---- Step 4: sampling from distributions ----------------------------------
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]


def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples


def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)  # 1-u1: never 0
        samples.append(mu + sigma * z)
    return samples


random.seed(0)
b = sample_bernoulli(0.3, 10_000)
print(f"Bernoulli(0.3): mean of 10k samples = {sum(b)/len(b):.4f}")
c = sample_categorical([0.7, 0.2, 0.1], 10_000)
print(f"Categorical [0.7,0.2,0.1]: fractions = {[round(c.count(i)/len(c), 4) for i in range(3)]}")
s = sample_normal_box_muller(0, 1, 10_000)
within_1sd = sum(1 for v in s if abs(v) < 1) / len(s)
print(f"Box-Muller N(0,1): mean = {sum(s)/len(s):.4f}, within 1σ = {within_1sd:.4f}")


# ---- Step 5: softmax and log probabilities --------------------------------
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]


def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]


print("softmax([2, 1, 0.1])       =", [round(p, 4) for p in softmax([2, 1, 0.1])])
print("softmax([100, 101, 102])   =", [round(p, 4) for p in softmax([100, 101, 102])])
print("log_softmax([-1000, 0])    =", log_softmax([-1000, 0]))
print("naive log(softmax) same    =", [math.log(p) if p > 0 else "log(0)!" for p in softmax([-1000, 0])])
print(f"CE loss([2, 1, 0.1], target 0) = {cross_entropy_loss([2, 1, 0.1], 0):.4f}")
print(f"CE loss([2, 1, 0.1], target 2) = {cross_entropy_loss([2, 1, 0.1], 2):.4f}")


# ---- Step 6: Central Limit Theorem ----------------------------------------
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages


def text_histogram(values, bins, lo, hi, scale):
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        i = min(int((v - lo) / width), bins - 1)
        counts[i] += 1
    for i, c in enumerate(counts):
        print(f"{lo + i * width:5.2f} {'#' * (c // scale)}")


random.seed(0)
die = lambda: random.randint(1, 6)
for n in (1, 2, 30):
    print(f"\naverage of {n} dice, 20k trials")
    text_histogram(demonstrate_clt(die, n, 20_000), 11, 1, 6, 150)


# ---- Step 7: visualization ------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.set_title("Normal PDF")
xs = [i * 0.01 - 5 for i in range(1001)]
for mu_val, sigma_val, label in [(0, 1, "N(0,1)"), (0, 2, "N(0,2)"), (2, 0.5, "N(2,0.5)")]:
    ys = [normal_pdf(x, mu_val, sigma_val) for x in xs]
    ax.plot(xs, ys, label=label, linewidth=2)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()

ax = axes[1]
ax.set_title("CLT: mean of n uniforms")
for n_val in (1, 2, 5, 30):
    avgs = demonstrate_clt(random.random, n_val, 10_000)
    ax.hist(avgs, bins=50, alpha=0.5, label=f"n={n_val}", density=True)
ax.set_xlabel("sample mean")
ax.set_ylabel("density")
ax.legend()

plt.tight_layout()
plt.savefig("probability_distributions.png", dpi=150)
print("Saved: probability_distributions.png")


# ---- Use It: the library versions -----------------------------------------
import numpy as np
from scipy import stats
from scipy.special import softmax as sp_softmax, log_softmax as sp_log_softmax

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000, random_state=0)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")
print(f"normal.pdf(0) = {normal.pdf(0):.4f}   ours: {normal_pdf(0, 0, 1):.4f}")

logits = np.array([2.0, 1.0, 0.1])
print(f"Softmax:     {sp_softmax(logits)}   ours: {[round(p, 4) for p in softmax(list(logits))]}")
print(f"Log-softmax: {sp_log_softmax(logits)}")


# ---- Exercise 1: inverse transform sampling, exponential -------------------
def exponential_pdf(x, lam):
    return lam * math.exp(-lam * x) if x >= 0 else 0.0


def sample_exponential(lam, n):
    # CDF F(x) = 1 − e^(−λx); solve u = F(x) → x = −ln(1−u)/λ
    return [-math.log(1 - random.random()) / lam for _ in range(n)]


random.seed(0)
lam = 2.0
ex = sample_exponential(lam, 10_000)
print(f"exponential λ={lam}: sample mean = {sum(ex)/len(ex):.4f}  (theory 1/λ = {1/lam})")
print(f"  fraction below 1: {sum(1 for v in ex if v < 1)/len(ex):.4f}  (theory F(1) = {1 - math.exp(-lam):.4f})")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(ex, bins=60, density=True, alpha=0.5, label="10k samples")
xs_e = [i * 0.01 for i in range(400)]
ax.plot(xs_e, [exponential_pdf(x, lam) for x in xs_e], "r", linewidth=2, label="true PDF λe^(−λx)")
ax.set_title("Exercise 1: inverse transform, exponential λ=2")
ax.legend()
plt.savefig("exercise1_exponential.png", dpi=150)


# ---- Exercise 2: joint table for two loaded dice ---------------------------
def joint_from_independent(p_a, p_b):
    return [[pa * pb for pb in p_b] for pa in p_a]


def marginals(joint):
    p_a = [sum(row) for row in joint]
    p_b = [sum(joint[i][j] for i in range(len(joint))) for j in range(len(joint[0]))]
    return p_a, p_b


def is_independent(joint, tol=1e-9):
    p_a, p_b = marginals(joint)
    return all(
        abs(joint[i][j] - p_a[i] * p_b[j]) < tol
        for i in range(len(joint))
        for j in range(len(joint[0]))
    )


def show_joint(joint, title):
    p_a, p_b = marginals(joint)
    print(f"\n{title}")
    print("        B=1    B=2    B=3    B=4    B=5    B=6  | P(A)")
    for i, row in enumerate(joint):
        print(f"A={i+1}  " + "  ".join(f"{v:.3f}" for v in row) + f"  | {p_a[i]:.3f}")
    print("P(B)  " + "  ".join(f"{v:.3f}" for v in p_b) + f"  | {sum(p_b):.3f}")
    print(f"independent? {is_independent(joint)}")


die_a = [0.10, 0.10, 0.10, 0.10, 0.10, 0.50]
die_b = [0.30, 0.14, 0.14, 0.14, 0.14, 0.14]
show_joint(joint_from_independent(die_a, die_b), "two loaded dice, rolled separately")

# linked pair: when A shows 6, B's odds shift toward 6 as well
linked = joint_from_independent(die_a, die_b)
linked[5] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.25]     # row A=6 still sums to 0.50
show_joint(linked, "two loaded dice, linked")


# ---- Exercise 3: cross-entropy vs PyTorch ----------------------------------
import torch
import torch.nn as nn

logits5 = [2.0, 0.5, -1.0, 3.0, 0.1]
ours = cross_entropy_loss(logits5, 3)
theirs = nn.CrossEntropyLoss()(torch.tensor([logits5]), torch.tensor([3]))
print(f"ours:    {ours:.6f}   P(correct) = {softmax(logits5)[3]:.4f}")
print(f"PyTorch: {theirs.item():.6f}")

close = [1.0, 1.0, 1.0, 1.0, 1.1]
print(f"argmax right but barely: P(correct) = {softmax(close)[4]:.4f}, loss = {cross_entropy_loss(close, 4):.4f}")


# ---- Exercise 4: log-probs back to probability -----------------------------
def sequence_probability(log_probs):
    total_log = sum(log_probs)
    return total_log, math.exp(total_log)


def most_likely_sequence(candidates):
    """candidates: {name: [log P(word1), log P(word2), ...]} → best name, its log-prob, raw prob."""
    best = max(candidates, key=lambda name: sum(candidates[name]))
    total_log, raw = sequence_probability(candidates[best])
    return best, total_log, raw


sentence = [math.log(0.01)] * 50
total_log, raw = sequence_probability(sentence)
print(f"50 words at P=0.01: log-prob = {total_log:.2f}, raw = {raw}")
print(f"  same in float32: {np.exp(np.float32(total_log))}")
print(f"  float64 floor ≈ {np.finfo(np.float64).tiny:.1e}, float32 floor ≈ {np.finfo(np.float32).tiny:.1e}")

cands = {
    "50 words @ 0.01": sentence,
    "50 words @ 0.02": [math.log(0.02)] * 50,
    "60 words @ 0.05": [math.log(0.05)] * 60,
}
best, tl, r = most_likely_sequence(cands)
print(f"most likely: '{best}'  log-prob {tl:.2f}  raw {r:.3e}")
