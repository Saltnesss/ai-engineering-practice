import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)

# 数据: 训练集少(30个点)，验证集多(200个点)，范围一致但不重叠
x_train = torch.linspace(0, 1, 15).unsqueeze(1)
y_train = torch.sin(2 * torch.pi * x_train) + torch.randn(15, 1) * 0.1

x_val = torch.linspace(0, 1, 200).unsqueeze(1)
y_val = torch.sin(2 * torch.pi * x_val)

# 模型: 相对30条数据来说，偏复杂
model = nn.Sequential(
    nn.Linear(1, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
writer_train = SummaryWriter("runs/overfit_demo/train")
writer_val = SummaryWriter("runs/overfit_demo/val")

for epoch in range(1000):
    optimizer.zero_grad()
    pred_train = model(x_train)
    loss_train = criterion(pred_train, y_train)
    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        pred_val = model(x_val)
        loss_val = criterion(pred_val, y_val)

    writer_train.add_scalar("loss", loss_train.item(), epoch)
    writer_val.add_scalar("loss", loss_val.item(), epoch)

writer_train.close()
writer_val.close()
print("训练完成，日志写进了 runs/overfit_demo")
