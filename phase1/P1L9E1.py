import math


def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float("inf") if p <= 0 else 0.0
    return -math.log(p) / math.log(base)


def entropy(probs, base=2):
    return sum(p * information_content(p, base) for p in probs if p > 0)


english_letter_probs = [1 / 26] * 26  # Uniform distribution for 26 letters
real_english_letter_probs = [
    0.08167,
    0.01492,
    0.02782,
    0.04253,
    0.12702,
    0.02228,  # a b c d e f
    0.02015,
    0.06094,
    0.06966,
    0.00153,
    0.00772,
    0.04025,  # g h i j k l
    0.02406,
    0.06749,
    0.07507,
    0.01929,
    0.00095,
    0.05987,  # m n o p q r
    0.06327,
    0.09056,
    0.02758,
    0.00978,
    0.02360,
    0.00150,  # s t u v w x
    0.01974,
    0.00074,  # y z
]

print(
    f"entropy of English letters(normal distribution): {entropy(english_letter_probs):.4f} bits"
)
print(
    f"entropy of English letters(real world): {entropy(real_english_letter_probs):.4f} bits"
)
