import torch
from fvcore.nn import FlopCountAnalysis

from models.architectures.unet3d_segformer import get_device
from models.architectures.unetr_pp.vesuv.unetr_pp import UNETR_PP
from models.lightning_modules.abstract_module import AbstractLightningModule


def calculate_masked_metrics_batchwise(epsilon, outputs, labels, mask):
    # Ensure batch dimension is maintained during operations
    outputs = (outputs > 0.5).float()
    batch_size = outputs.size(0)

    # Flatten tensors except for the batch dimension
    outputs_flat = (outputs * mask).view(batch_size, -1)
    labels_flat = (labels * mask).view(batch_size, -1)

    # Calculate True Positives, False Positives, and False Negatives for each batch
    true_positives = (outputs_flat * labels_flat).sum(dim=1)
    false_positives = (outputs_flat * (1 - labels_flat)).sum(dim=1)
    false_negatives = ((1 - outputs_flat) * labels_flat).sum(dim=1)

    # Calculate metrics for each batch
    iou = true_positives / (
            true_positives + false_positives + false_negatives + epsilon)  # Added epsilon for numerical stability
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)  # Added epsilon for F1 calculation

    return iou.mean(), precision.mean(), recall.mean(), f1.mean()


class UNETR_PP_Module(AbstractLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.epsilon = cfg.epsilon

        in_channels = 16
        out_channels = 2

        self.model = UNETR_PP(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_rate=.2,
            feature_size=16,
            num_heads=[3, 6, 12, 24],
            depths=[2, 2, 2, 2],
            dims=[32, 64, 128, 256],
            do_ds=False,  # Use deep supervision to compute the loss.
        )

        if torch.cuda.is_available():
            self.network.cuda()

        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

        # Print the network parameters & Flops
        n_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        input_res = (4, 128, 128, 128)
        input = torch.ones(()).new_empty((1, *input_res), dtype=next(self.network.parameters()).dtype,
                                         device=next(self.network.parameters()).device)
        flops = FlopCountAnalysis(self.network, input)
        model_flops = flops.total()
        print(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
        print(f"MAdds: {round(model_flops * 1e-9, 2)} G")

    def training_step(self, batch, batch_idx):
        data, label = batch
        y_true = label[:, 0]
        y_mask = label[:, 1]

        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        total_loss, losses = self.calculate_masked_weighted_loss(logits, y_true, y_mask)
        self.log_losses_to_wandb(losses, 'train')

        self.update_unetr_training_metrics(total_loss)
        # self.train_step += 1

        # if batch_idx == 5 and self.trainer.is_global_zero:
        #     with torch.no_grad():
        #         combined = torch.cat([y_pred[0], y_true[0], y_mask[0]], dim=1)
        #         grid = make_grid(combined).detach().cpu()
        #         test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
        #         wandb.log({"Train Image": test_image})

        return total_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        y_true = label[:, 0]
        y_mask = label[:, 1]

        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        total_loss, losses = self.calculate_masked_weighted_loss(logits, y_true, y_mask)
        self.log_losses_to_wandb(losses, 'val')

        iou, precision, recall, f1 = calculate_masked_metrics_batchwise(self.epsilon, y_pred, y_true, y_mask)
        self.update_unetr_validation_metrics(total_loss, iou, precision, recall, f1)

        # if batch_idx % 20 == 0 and self.trainer.is_global_zero:
        #     with torch.no_grad():
        #         combined = torch.cat([y_pred[0], y_true[0], y_mask[0]], dim=1)
        #         grid = make_grid(combined).detach().cpu()
        #         test_image = wandb.Image(grid, caption="Train Step {}".format(self.train_step))
        #         wandb.log({"Validation Image": test_image})

    def calculate_masked_weighted_loss(self, logits, y_true, y_mask):
        losses = [(name, weight, loss_function(logits, y_true.float(), y_mask)) for (name, weight, loss_function) in
                  self.loss_functions]
        total_loss = sum([weight * value for (_, weight, value) in losses])
        losses.append(("total", 1.0, total_loss))

        return total_loss, losses

    def update_unetr_validation_metrics(self, loss, iou, precision, recall, f1):
        self.log(f'val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_precision', precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_recall', recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def update_unetr_training_metrics(self, loss):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log(f'train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


if __name__ == "__main__":
    # model = UNETR(input_dim=1, output_dim=32, img_shape=(16, 256, 256))
    print("test")

    model = UNETR_PP(
        in_channels=12,
        out_channels=1,
        dropout_rate=.2,
        feature_size=16,
        num_heads=4,
        depths=[3, 3, 3, 3],
        dims=[32, 64, 128, 256],
        do_ds=False,  # Use deep supervision to compute the loss.
    )

    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("CUDA is not available. The model will remain on the CPU.")

    # Input
    x = torch.randn(4, 1, 16, 128, 128)
    print(x.shape)

    # move x to cuda
    x = x.to(get_device(model))

    output = model(x)

    print(output.shape)
    print(torch.unique(output))
