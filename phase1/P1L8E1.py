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
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

start = [-1.0, 1.0]

lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]

for lr in lrs:
    history = optimize(GradientDescent(lr=lr), rosenbrock, rosenbrock_gradient, start)
    final = history[-1]
    loss = rosenbrock(final)
    print(f"lr={lr:<7} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")