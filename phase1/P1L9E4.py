import math


def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def perplexity(predictions):
    """predictions: list of (true_token_index, predicted_logits) pairs."""
    losses = [-math.log(softmax(logits)[true_idx]) for true_idx, logits in predictions]
    avg_loss = sum(losses) / len(losses)  # average, not sum: per-token
    return math.exp(avg_loss)  # nats -> e^H


# Vocabulary: 5 words, each has an index
#   0      1      2      3     4
# "the"  "cat"  "sat"  "on"  "mat"
#
# Sentence to predict: "cat sat on"  ->  true indices 1, 2, 3
# Each logits list gives one score per word, in the order above.

# 1. Clueless model: same score for every word, every time
clueless = [
    (1, [0.0, 0.0, 0.0, 0.0, 0.0]),  # true word "cat"
    (2, [0.0, 0.0, 0.0, 0.0, 0.0]),  # true word "sat"
    (3, [0.0, 0.0, 0.0, 0.0, 0.0]),  # true word "on"
]

# 2. Good model: the true word gets the high score
good = [
    (1, [0.0, 10.0, 0.0, 0.0, 0.0]),  # high score on "cat"
    (2, [0.0, 0.0, 10.0, 0.0, 0.0]),  # high score on "sat"
    (3, [0.0, 0.0, 0.0, 10.0, 0.0]),  # high score on "on"
]

# 3. Confidently wrong model: a wrong word gets the high score
wrong = [
    (1, [10.0, 0.0, 0.0, 0.0, 0.0]),  # high score on "the", but truth is "cat"
    (2, [0.0, 10.0, 0.0, 0.0, 0.0]),  # high score on "cat", but truth is "sat"
    (3, [0.0, 0.0, 10.0, 0.0, 0.0]),  # high score on "sat", but truth is "on"
]

print(f"clueless: {perplexity(clueless):.2f}")
print(f"good:     {perplexity(good):.4f}")
print(f"wrong:    {perplexity(wrong):.2f}")
