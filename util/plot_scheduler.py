import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim import SGD, AdamW
import matplotlib.pyplot as plt

STEPS = 100

lr = 2e-4
weight_decay = 0.005

optimizer = AdamW([torch.tensor(1)], lr=lr, weight_decay=weight_decay)
# Use a scheduler of your choice below.
# Great for debugging your own schedulers!
scheduler = StepLR(
    optimizer,
    step_size=10,
    gamma=0.8
)

# scheduler = CosineAnnealingLR(optimizer, STEPS)

lrs = []
for _ in range(500):
    optimizer.step()
    lrs.append(scheduler.get_last_lr())
    scheduler.step()

print(lrs)
plt.plot(lrs)
plt.show()
