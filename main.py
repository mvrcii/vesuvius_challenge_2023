import os

import torch
from tqdm import tqdm

import wandb

from conf import CFG
from dataset import build_dataloader
from unetr_segformer import UNETR_Segformer
from torch.optim import AdamW
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR


def main():
    wandb.init(project="Kaggle1stReimp", entity="wuesuv")

    model = UNETR_Segformer(CFG)

    if torch.cuda.is_available():
        model = model.to('cuda')
    else:
        print('Cuda not available')

    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler_steplr = StepLR(optimizer, step_size=1, gamma=0.9)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_steplr)
    optimizer.zero_grad()
    optimizer.step()

    loss_function = torch.nn.BCELoss()

    train_data_loader = build_dataloader(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)), dataset_type='train')

    # Training loop
    for epoch in tqdm(range(CFG.epochs)):
        model.train()
        total_loss = 0

        scheduler_warmup.step(epoch)
        print("Epoch:", epoch, "LR:", optimizer.param_groups[0]['lr'])
        wandb.log({"LR": optimizer.param_groups[0]['lr']})

        print("Starting Epoch", epoch)

        for batch_idx, (data, target) in tqdm(enumerate(train_data_loader)):

            data, target = data.to(CFG.device), target.to(CFG.device)

            # Forward pass
            output = model(data)

            loss = loss_function(output, target.float())

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