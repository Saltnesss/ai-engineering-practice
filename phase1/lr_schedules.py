import math
import matplotlib.pyplot as plt

T, lr0 = 1000, 0.001                       # 1000 training steps, starting lr 0.001
steps = range(T)

step_decay  = [lr0 * 0.5 ** (t // 250) for t in steps]        # halve every 250 steps
exp_decay   = [lr0 * 0.997 ** t for t in steps]               # multiply by 0.997 every step
cosine      = [0.5 * lr0 * (1 + math.cos(math.pi * t / T)) for t in steps]   # lr_min = 0
W = 100                                                       # warmup length
warm_cos    = [lr0 * t / W if t < W else
               0.5 * lr0 * (1 + math.cos(math.pi * (t - W) / (T - W))) for t in steps]

for ys, name in [(step_decay, "step decay"), (exp_decay, "exponential"),
                 (cosine, "cosine annealing"), (warm_cos, "warmup + cosine")]:
    plt.plot(steps, ys, label=name)
plt.xlabel("training step"); plt.ylabel("learning rate"); plt.legend()
plt.title("Learning rate schedules (lr0 = 0.001, T = 1000)")
plt.savefig("lr_schedules.png", dpi=120)
print("saved")
