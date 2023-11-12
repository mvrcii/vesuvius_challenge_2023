import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import wandb
from conf import CFG
from dataset import build_dataloader
from metrics import calculate_incremental_metrics, calculate_final_metrics
from unetr_segformer import UNETR_Segformer


def main():
    wandb.init(project="Kaggle1stReimp", entity="wuesuv")

    model = UNETR_Segformer(CFG)

    if torch.cuda.is_available():
        model = model.to(CFG.device)
    else:
        print('Cuda not available')

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.995)
    loss_function = torch.nn.BCELoss()
    # TODO: Implement and test Dice Loss
    # TODO: Add global seeding

    train_data_loader = build_dataloader(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
                                         dataset_type='train')

    val_data_loader = build_dataloader(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
                                       dataset_type='val')

    accumulation_steps = 4
    accumulated_loss = 0

    # Training loop
    for epoch in tqdm(range(CFG.epochs), desc='Epochs'):
        model.train()
        total_loss = 0

        optimizer.zero_grad()  # Reset gradients; do this once at the start

        with tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc='Batches', leave=False) as t:
            for batch_idx, (data, target) in t:
                data, target = data.to(CFG.device), target.to(CFG.device)

                # Forward pass
                output = model(data)
                loss = loss_function(output, target.float())
                loss = loss / accumulation_steps  # Normalize our loss (if averaged)

                loss.backward()  # Accumulate gradients

                accumulated_loss += loss.item()

                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()  # Perform a single optimization step
                    optimizer.zero_grad()  # Reset gradients after updating
                    scheduler.step()  # Update the learning rate

                    # Log the accumulated loss and then reset it
                    wandb.log({
                        "Epoch": epoch,
                        "Batch Loss": accumulated_loss,
                        "LR": optimizer.param_groups[0]['lr']
                    })

                    total_loss += accumulated_loss * accumulation_steps  # Adjust for the normalization
                    t.set_postfix({'Batch Loss': accumulated_loss * accumulation_steps})
                    accumulated_loss = 0  # Reset accumulated loss after logging

        # Log epoch loss
        average_loss = total_loss / (len(train_data_loader) / accumulation_steps)
        wandb.log({"Epoch": epoch, "Average Train Loss": average_loss})
        print(f"Epoch [{epoch + 1}/{CFG.epochs}], Train Loss: {average_loss:.4f}")

        # Validation step
        val_metrics = validate_model(epoch, model, val_data_loader, CFG.device)

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
            output = model(data)
            output = torch.sigmoid(output)  # Convert to probabilities

            if not visualized:
                visualize(epoch=epoch, val_idx=0, pred_label=output, target_label=target)
                visualized = True  # Set the flag to True after visualization

            calculate_incremental_metrics(metric_accumulator, target.cpu().numpy(), output.cpu().numpy(), threshold)

    return calculate_final_metrics(metric_accumulator)


def visualize(epoch, val_idx, val_total, pred_label, target_label):
    if CFG.show_predictions and torch.max(pred_label).item() > 0:
        print("Predicting something white!")

        pred_label_np = pred_label.cpu().numpy().squeeze(0) > 0.5  # Threshold predictions
        label_np = target_label.cpu().numpy().squeeze(0) > 0.5  # Threshold ground truth

        # Calculate the correct and wrong pixels
        correct = np.logical_and(pred_label_np == label_np, label_np == 1)
        wrong = pred_label_np != label_np

        overlay = np.zeros((*label_np.shape, 3), dtype=np.uint8)
        overlay[..., 1] = correct * 255  # Green for correct
        overlay[..., 0] = wrong * 255  # Red for incorrect

        # Create a figure with 1 row and 2 columns of subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        fig.suptitle(f'Epoch {epoch}, Validation Index {val_idx} of {val_total}', fontsize=16)

        # Plot the first tensor on the first axis
        ax1.imshow(label_np, cmap='gray')
        ax1.title.set_text('Ground Truth')
        ax1.axis('off')

        # Plot the second tensor on the second axis
        ax2.imshow(pred_label_np, cmap='gray')
        ax2.title.set_text('Prediction')
        ax2.axis('off')

        # Plot the overlay on the third axis
        ax3.imshow(label_np, cmap='gray')
        ax3.imshow(overlay, cmap=ListedColormap(['red', 'green']), alpha=0.5)
        ax3.title.set_text('Overlay')
        ax3.axis('off')

        # Save the plot to a file
        vis_path = 'vis'
        os.makedirs(vis_path, exist_ok=True)
        vis_path = os.path.join(vis_path, f'vis_epoch_{epoch}_idx_{val_idx}.png')
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure
        print("Saved image in", vis_path)
    else:
        print("Skip visualization, torch.max(pred_label).item() =", torch.max(pred_label).item())


if __name__ == '__main__':
    main()
