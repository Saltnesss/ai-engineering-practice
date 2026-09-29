import math


def entropy(probs, base=2):
    return -sum(p * math.log(p, base) for p in probs if p > 0)


def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float("inf")
            total += -pi * math.log(qi, base)
    return total


def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)


def kl_terms(p, q, base=2):
    """Each x's contribution: p(x) * log(p(x) / q(x)), the weight is p(x)."""
    return [pi * math.log(pi / qi, base) if pi > 0 else 0.0 for pi, qi in zip(p, q)]


P = [0.7, 0.2, 0.1]  # true_dist from Build It Step 2
Q = [0.1, 0.1, 0.8]  # bad_model from Build It Step 2

print(f"D_KL(P || Q) = {kl_divergence(P, Q):.4f} bits")
print(f"D_KL(Q || P) = {kl_divergence(Q, P):.4f} bits")
print()
print("Per-term contributions:")
print("  P||Q:", [f"{t:+.3f}" for t in kl_terms(P, Q)])
print("  Q||P:", [f"{t:+.3f}" for t in kl_terms(Q, P)])

# An extreme pair: Q says outcome 2 is impossible
A = [0.5, 0.5]
B = [1.0, 0.0]
print()
print(f"D_KL(A || B) = {kl_divergence(A, B)}")
print(f"D_KL(B || A) = {kl_divergence(B, A):.4f} bits")
