import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(3):
    x = torch.randn(5, 4)
    target = torch.randn(5, 2)

    output = model(x)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()

    breakpoint()

    optimizer.step()
    print(f"step {step} done, loss={loss.item():.4f}")
