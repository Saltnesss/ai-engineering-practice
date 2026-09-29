import matplotlib.pyplot as plt


def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]


class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]


def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    losses = [func(params)]
    for _ in range(steps):
        try:
            grads = grad_func(params)
            params = optimizer.step(params, grads)
            losses.append(func(params))
        except OverflowError:
            return losses, True          # diverged: numbers grew past float range
    return losses, False


start = [-1.0, 1.0]
lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]

fig, (ax_all, ax_start) = plt.subplots(1, 2, figsize=(13, 5))

for lr in lrs:
    losses, diverged = optimize(GradientDescent(lr=lr), rosenbrock, rosenbrock_gradient, start)
    if diverged:
        print(f"lr={lr:<7} -> DIVERGED after {len(losses) - 1} steps (last loss {losses[-1]:.2e})")
        ax_start.plot(losses, "x--", label=f"lr={lr} (diverged)")
    else:
        print(f"lr={lr:<7} -> final loss after 5000 steps = {losses[-1]:.8f}")
        ax_all.plot(losses, label=f"lr={lr}")
        ax_start.plot(losses[:11], "o-", label=f"lr={lr}")

ax_all.set_yscale("log")
ax_all.set_xlabel("step")
ax_all.set_ylabel("loss (log scale)")
ax_all.set_title("Converging learning rates, all 5000 steps")
ax_all.legend()

ax_start.set_yscale("log")
ax_start.set_xlabel("step")
ax_start.set_title("All five, first 10 steps")
ax_start.legend()

plt.tight_layout()
plt.savefig("P1L8E1_lr_sweep.png", dpi=110)
print("saved P1L8E1_lr_sweep.png")
