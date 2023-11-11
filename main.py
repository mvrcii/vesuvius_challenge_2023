import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import wandb
from conf import CFG
from dataset import build_dataloader
from unetr_segformer import UNETR_Segformer


def main():
    wandb.init(project="Kaggle1stReimp", entity="wuesuv")

    model = UNETR_Segformer(CFG)

    if torch.cuda.is_available():
        model = model.to('cuda')
    else:
        print('Cuda not available')

    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler_steplr = StepLR(optimizer, step_size=1, gamma=0.9)

    loss_function = torch.nn.BCELoss()

    train_data_loader = build_dataloader(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
                                         dataset_type='train')

    # Training loop
    for epoch in tqdm(range(CFG.epochs), desc='Epochs'):
        model.train()
        total_loss = 0

        print("Epoch:", epoch, "LR:", optimizer.param_groups[0]['lr'])
        wandb.log({"LR": optimizer.param_groups[0]['lr']})

        print("Starting Epoch", epoch)

        with tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc='Batches', leave=False) as t:
            for batch_idx, (data, target) in t:
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

                t.set_postfix({'Batch Loss': loss.item()})

        scheduler_steplr.step()

        # Log epoch loss
        average_loss = total_loss / len(train_data_loader)
        wandb.log({"Epoch": epoch, "Average Loss": average_loss})

        print(f"Epoch [{epoch + 1}/{CFG.epochs}], Loss: {average_loss:.4f}")

    # Finish Weights & Biases run
    wandb.finish()


if __name__ == '__main__':
    main()
