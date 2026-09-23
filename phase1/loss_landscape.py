import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

torch.manual_seed(0)
X, y = make_moons(200, noise=0.2, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

model = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
loss_fn = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(2000):                                  # train to a minimum w*
    opt.zero_grad(); loss = loss_fn(model(X).squeeze(), y); loss.backward(); opt.step()

w_star = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
print("number of weights:", w_star.numel(), "  trained loss:", round(loss.item(), 4))

d1 = torch.randn_like(w_star); d1 /= d1.norm()        # two random directions, length 1
d2 = torch.randn_like(w_star); d2 /= d2.norm()

def loss_at(w):
    torch.nn.utils.vector_to_parameters(w, model.parameters())
    with torch.no_grad():
        return loss_fn(model(X).squeeze(), y).item()

a_vals = np.linspace(-10, 10, 81)
line = [loss_at(w_star + a * d1) for a in a_vals]                                  # 1-D slice
grid = [[loss_at(w_star + a * d1 + b * d2) for a in a_vals] for b in a_vals]      # 2-D slice

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(a_vals, line); ax1.axvline(0, color="r", ls="--", lw=0.8)
ax1.set_xlabel("a  (distance moved along d₁)"); ax1.set_ylabel("loss")
ax1.set_title("1-D slice: L(w* + a·d₁)")
cs = ax2.contourf(a_vals, a_vals, grid, levels=30, cmap="viridis"); fig.colorbar(cs, ax=ax2, label="loss")
ax2.plot(0, 0, "r*", ms=12)
ax2.set_xlabel("a  (along d₁)"); ax2.set_ylabel("b  (along d₂)")
ax2.set_title("2-D slice: L(w* + a·d₁ + b·d₂)")
plt.tight_layout(); plt.savefig("loss_landscape.png", dpi=110)
