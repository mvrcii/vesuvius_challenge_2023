import os

import torch
import wandb

from conf import CFG
from dataset import build_dataloader
from unetr_segformer import UNETR_Segformer
from torch.optim import AdamW


def main():
    wandb.init(project="Kaggle1stReimp", entity="mvrcii_")

    model = UNETR_Segformer(CFG)

    if torch.cuda.is_available():
        model = model.to('cuda')
    else:
        print('Cuda not available')

    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    loss_function = torch.nn.BCELoss()

    train_data_loader = build_dataloader(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)), dataset_type='train')

    # Training loop
    for epoch in range(CFG.epochs):
        model.train()
        total_loss = 0
        print("Starting Epoch", epoch)

        for batch_idx, (data, target) in enumerate(train_data_loader):
            print("Starting the batch loop")
            data, target = data.to(CFG.device), target.to(CFG.device)
            print("Moved input to CUDA")

            # Forward pass
            output = model(data)
            # output = output.squeeze(1)
            print("Performed Forward Pass")

            loss = loss_function(output, target.float())
            print("Calculated Loss")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Performed Backward")

            total_loss += loss.item()

            # Log to wandb
            wandb.log({"Epoch": epoch, "Batch Loss": loss.item()})

        # Log epoch loss
        average_loss = total_loss / len(train_data_loader)
        wandb.log({"Epoch": epoch, "Average Loss": average_loss})

        print(f"Epoch [{epoch+1}/{CFG.epochs}], Loss: {average_loss:.4f}")

    # Finish Weights & Biases run
    wandb.finish()


if __name__ == '__main__':
    main()