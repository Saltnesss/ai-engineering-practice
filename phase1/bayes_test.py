"""Phase 1 · Lesson 7 — Bayes' Theorem. Build It steps + exercises."""

# ---- Step 1: Bayes' theorem as a function ---------------------------------
def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior


# ---- Step 2: Naive Bayes classifier ----------------------------------------
import math
from collections import defaultdict


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)                    # class -> #docs
        self.word_counts = defaultdict(lambda: defaultdict(int))  # class -> word -> count
        self.class_word_totals = defaultdict(int)               # class -> total words
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_class = None
        best_score = float("-inf")
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)          # log prior
            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size))
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class


# ---- Step 3: training data --------------------------------------------------
train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]
train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham",
]
test_messages = [
    "free money waiting for you",
    "meeting rescheduled to friday",
    "you won a free prize",
    "please review the attached report",
]


# ---- Step 4: inspect learned probabilities -----------------------------------
def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_totals[cls]
    probs = {}
    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)
        probs[word] = (count + classifier.smoothing) / (total + classifier.smoothing * vocab_size)
    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for word, prob in sorted_words[:n]:
        print(f"    {word}: {prob:.4f}")


if __name__ == "__main__":
    print("Step 1: medical test")
    result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
    print(f"  P(sick|positive) = {result:.4f}")

    print("\nStep 3: spam classifier")
    classifier = NaiveBayes()
    classifier.train(train_docs, train_labels)
    for msg in test_messages:
        print(f"  '{msg}' -> {classifier.predict(msg)}")

    print("\nStep 4: top words")
    print("  spam:")
    show_top_words(classifier, "spam")
    print("  ham:")
    show_top_words(classifier, "ham")

    # ---- Use It: scikit-learn ------------------------------------------------
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    print("\nUse It: sklearn MultinomialNB")
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_docs)
    clf = MultinomialNB()
    clf.fit(X_train, train_labels)

    X_test = vectorizer.transform(test_messages)
    predictions = clf.predict(X_test)
    for msg, pred in zip(test_messages, predictions):
        print(f"  '{msg}' -> {pred}")
    print(f"  X_train shape = {X_train.shape}  (docs x vocab)")

    # ---- Ship It: Bayesian A/B test via Monte Carlo -------------------------
    import numpy as np

    print("\nShip It: Bayesian A/B test")
    rng = np.random.default_rng(0)
    samples_A = rng.beta(1 + 50, 1 + 950, size=100_000)   # A: 50 clicks / 1000
    samples_B = rng.beta(1 + 65, 1 + 935, size=100_000)   # B: 65 clicks / 1000
    p_b_better = np.mean(samples_B > samples_A)
    print(f"  mean A = {samples_A.mean():.4f}, mean B = {samples_B.mean():.4f}")
    print(f"  P(B > A) = {p_b_better:.3f}")

    # ---- Exercise 1: two positive tests --------------------------------------
    print("\nExercise 1: two positive tests")
    first = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
    second = bayes(prior=first, likelihood=0.99, false_positive_rate=0.01)
    print(f"  after 1st positive: P(sick) = {first:.4f}")
    print(f"  after 2nd positive: P(sick) = {second:.4f}")
    evidence2 = 0.99 * first + 0.01 * (1 - first)
    print(f"  (2nd test: P(pos) = 0.99*{first:.4f} + 0.01*{1-first:.4f} = {evidence2:.5f})")
