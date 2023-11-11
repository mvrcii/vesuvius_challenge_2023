import torch
import wandb

from conf import CFG
from dataset import build_dataloader
from unetr_segformer import UNETR_Segformer
from utils.create_dataset import create_dataset
from torch.optim import AdamW


def main():
    wandb.init(project="Kaggle1stReimp", entity="mvrcii_")

    model = UNETR_Segformer(CFG)
    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    loss_function = torch.nn.BCELoss()

    train_data_loader = build_dataloader(data_root_dir=CFG.data_root_dir, dataset_type='train')

    # Training loop
    for epoch in range(CFG.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(torch.device(CFG.device)), target.to(torch.device(CFG.device))

            # Forward pass
            output = model(data)
            loss = loss_function(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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