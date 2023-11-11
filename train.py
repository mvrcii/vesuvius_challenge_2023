import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import wandb
from conf import CFG
from dataset import build_dataloader
from metrics import calculate_incremental_metrics, calculate_final_metrics, log_predictions_to_wandb
from unetr_segformer import UNETR_Segformer


def main():
    wandb.init(project="Kaggle1stReimp", entity="wuesuv")

    model = UNETR_Segformer(CFG)

    if torch.cuda.is_available():
        model = model.to(CFG.device)
    else:
        print('Cuda not available')

    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.999)
    loss_function = torch.nn.BCELoss()
    # TODO: Implement and test Dice Loss
    # TODO: Add global seeding

    train_data_loader = build_dataloader(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
                                         dataset_type='train')

    val_data_loader = build_dataloader(data_root_dir=os.path.join(CFG.data_root_dir, str(CFG.size)),
                                       dataset_type='val')

    # Training loop
    for epoch in tqdm(range(CFG.epochs), desc='Epochs'):
        model.train()
        total_loss = 0

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

                # Update Scheduler
                scheduler.step()
                wandb.log({"LR": optimizer.param_groups[0]['lr']})

                # Log to wandb
                wandb.log({"Epoch": epoch, "Batch Loss": loss.item()})

                t.set_postfix({'Batch Loss': loss.item()})

        # Log epoch loss
        average_loss = total_loss / len(train_data_loader)
        wandb.log({"Epoch": epoch, "Average Train Loss": average_loss})

        print(f"Epoch [{epoch + 1}/{CFG.epochs}], Train Loss: {average_loss:.4f}")

        # Validation step
        val_metrics = validate_model(model, val_data_loader, CFG.device)

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


def validate_model(model, val_data_loader, device, threshold=0.5):
    model.eval()
    step = 0
    metric_accumulator = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'auc': 0, 'count': 0}

    with torch.no_grad():
        for data, target in val_data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.sigmoid(output)  # Convert to probabilities

            calculate_incremental_metrics(metric_accumulator, target.cpu().numpy(), output.cpu().numpy(), threshold)

            log_predictions_to_wandb(data, output, target, threshold, step)
            step += 1

    return calculate_final_metrics(metric_accumulator)


if __name__ == '__main__':
    main()
