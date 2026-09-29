import matplotlib.pyplot as plt


def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]


class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]


def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    losses = [func(params)]
    for _ in range(steps):
        try:
            grads = grad_func(params)
            params = optimizer.step(params, grads)
            losses.append(func(params))
        except OverflowError:
            return losses, True
    return losses, False


start = [-1.0, 1.0]
momentums = [0.0, 0.5, 0.9, 0.99]

for m in momentums:
    losses, diverged = optimize(SGDMomentum(lr=0.0001, momentum=m), rosenbrock, rosenbrock_gradient, start)

    # "overshoot" = the loss went UP on a step: we passed the bottom and climbed the other side
    rises = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
    # "fastest" = first step at which the loss drops below 0.01
    reached = next((i for i, loss in enumerate(losses) if loss < 0.01), None)

    status = "DIVERGED" if diverged else f"final loss {losses[-1]:.8f}"
    print(f"momentum={m:<5} -> {status}, loss<0.01 at step {reached}, loss rose on {rises} steps")
    plt.plot(losses, label=f"momentum={m}")

plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss (log scale)")
plt.title("Momentum on Rosenbrock (lr=0.0001)")
plt.legend()
plt.savefig("P1L8E2_momentum.png", dpi=110)
print("saved P1L8E2_momentum.png")
