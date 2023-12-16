import torch
import torch.nn.functional as F
torch.nn.BCEWithLogitsLoss

# Correction to the code
outputs = torch.tensor([1.0, 0.0, 1.0])
labels = torch.tensor([1.0, 0.0, 1.0])

criterion = torch.nn.BCELoss(reduction='none')

loss = criterion(outputs, labels)
print(loss.shape)
print(loss)

