import os

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torch.optim import AdamW
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from conf import CFG
from dataset import build_dataloader
from metrics import calculate_incremental_metrics, calculate_final_metrics


def main():
    wandb.init(project="Kaggle1stReimp", entity="wuesuv")

    # define model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                             num_labels=1,
                                                             num_channels=16,
                                                             ignore_mismatched_sizes=True,
                                                             )

    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        print('Cuda not available')
        exit()

    optimizer = AdamW(model.parameters(), lr=0.00001)

    train_data_loader = build_dataloader(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
                                         dataset_type='train')

    val_data_loader = build_dataloader(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
                                       dataset_type='val')

    model.train()

    for epoch in tqdm(range(20), desc='Epochs'):
        total_loss = 0

        optimizer.zero_grad()  # Reset gradients; do this once at the start

        with tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc='Batches', leave=False) as t:
            for batch_idx, (data, target) in t:
                data, target = data.to("cuda"), target.to("cuda")
                optimizer.zero_grad()
                # todo find out why data has extra dimension of 1
                data = torch.squeeze(data, 1)

                # Forward pass
                outputs = model(pixel_values=data, labels=target)
                loss, logits = outputs.loss, outputs.logits
                loss.backward()
                wandb.log({"loss": loss.item()})
                optimizer.step()

        # Validation step
        val_metrics = validate_model(epoch, model, val_data_loader, "cuda")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"model_{CFG.size}_{CFG.lr}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        wandb.log(
            {"Epoch": epoch,
             "Val mIoU": val_metrics[0],
             "Val mAP": val_metrics[1],
             "Val AUC": val_metrics[2],
             "Val F1-Score": val_metrics[3]
             }
        )
        print(f"Validation - mIoU: {val_metrics[0]:.4f}, mAP: {val_metrics[1]:.4f}, "
              f"AUC: {val_metrics[2]:.4f}, F1-Score: {val_metrics[3]:.4f}")

    # Finish Weights & Biases run
    wandb.finish()
    torch.save(model.state_dict(), f"model_{CFG.size}_{CFG.lr}_final.pth")


def validate_model(epoch, model, val_data_loader, device, threshold=0.5):
    model.eval()
    metric_accumulator = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'auc': 0, 'count': 0}

    with torch.no_grad():

        visualized = False

        for data, target in val_data_loader:
            data, target = data.to(device), target.to(device)
            # todo find out why data has extra dimension of 1
            data = torch.squeeze(data, 1)
            output = model(data)
            output = output.logits
            output = torch.sigmoid(output)  # Convert to probabilities

            if not visualized:
                visualize(epoch=epoch, val_idx=0, pred_label=output, target_label=target)
                visualized = True  # Set the flag to True after visualization

            calculate_incremental_metrics(metric_accumulator, target.cpu().numpy(), output.cpu().numpy(), threshold)

    return calculate_final_metrics(metric_accumulator)


def visualize(epoch, val_idx, pred_label, target_label):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    axes[0].imshow(target_label.detach().squeeze().cpu().numpy(), cmap='viridis')
    axes[1].imshow(pred_label.detach().squeeze().cpu().numpy(), cmap='viridis')
    plt.tight_layout()
    plt.savefig(f"vis/vis_epoch{epoch}_validx{val_idx}.png")

if __name__ == '__main__':
    main()
