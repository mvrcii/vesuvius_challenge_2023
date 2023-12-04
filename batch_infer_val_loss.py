import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

from config_handler import Config
from data_modules.segformer import SegFormerDataModule
from util.losses import BinaryDiceLoss
import albumentations as A

val_image_aug = [
    A.Normalize(mean=[0], std=[1]),
]


def find_pth_in_dir(path):
    path = os.path.join('checkpoints', path)
    for file in os.listdir(path):
        if file.endswith('.ckpt'):
            return os.path.join(path, file)
    return None


def plot(image, label, prediction, filename):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns

    # Plotting the images in the first row
    for i in range(4):
        img = image[i].squeeze()  # Remove any unnecessary dimensions
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].axis('off')

    # Plotting the label in the second row, first column
    lbl = label.squeeze() # Remove any unnecessary dimensions
    axs[1, 0].imshow(lbl, cmap='gray')
    axs[1, 0].axis('off')

    # Plotting the prediction in the second row, second column
    pred = prediction.squeeze()  # Remove any unnecessary dimensions for prediction
    axs[1, 1].imshow(pred, cmap='gray')
    axs[1, 1].axis('off')

    # Turn off axes for empty subplots in the second row
    for i in range(2, 4):
        axs[1, i].axis('off')

    # Set the title for the entire plot
    fig.suptitle(f"Filename: {filename}", fontsize=16)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Check for the minimum number of arguments
    if len(sys.argv) != 2:
        print("Usage: python batch_infer_layered.py <checkpoint_folder_name>")
        sys.exit(1)

    checkpoint_folder_name = sys.argv[1]

    # Determine the path to the configuration based on the checkpoint folder
    config_path = os.path.join('checkpoints', checkpoint_folder_name, 'config.py')

    # Find the checkpoint path
    checkpoint_path = find_pth_in_dir(checkpoint_folder_name)
    if checkpoint_path is None:
        print("No valid checkpoint file found")
        sys.exit(1)

    config = Config.load_from_file(config_path)
    channels = config.in_chans

    model = SegformerForSemanticSegmentation.from_pretrained(config.from_pretrained,
                                                             num_labels=1,
                                                             num_channels=config.in_chans,
                                                             ignore_mismatched_sizes=True)
    model = model.to("cuda").half()
    checkpoint = torch.load(checkpoint_path)
    state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    print("Loaded model", checkpoint_path)

    config.val_aug = val_image_aug
    config.num_workers = 2
    config.val_batch_size = 4
    config.dataset_fraction = 0.1

    data_module = SegFormerDataModule(cfg=config)
    val_loader = data_module.val_dataloader()

    bce_loss_fn = BCEWithLogitsLoss()
    dice_loss_fn = BinaryDiceLoss(from_logits=True)

    losses = {}

    for (img_batch, label_batch, img_paths, label_paths) in tqdm(val_loader):

        label_batch = label_batch.to('cuda', dtype=torch.float16)
        img_batch = img_batch.to('cuda', dtype=torch.float16)

        with torch.no_grad():
            outputs = model(img_batch).logits

        outputs = torch.squeeze(outputs, dim=1)

        # Loop over elements in batch
        for i in range(img_batch.size(0)):
            bce_loss = bce_loss_fn(outputs[i], label_batch[i]).item()
            dice_loss = dice_loss_fn(torch.sigmoid(outputs[i]), label_batch[i]).item()
            total_loss = bce_loss + dice_loss

            prediction = torch.sigmoid(outputs[i])

            losses[img_paths[i]] = {
                'bce_loss': bce_loss,
                'dice_loss': dice_loss,
                'total_loss': total_loss,
                'label_path': label_paths[i],
                'prediction': prediction.cpu().numpy()  # Detach and convert to numpy array
            }

    def foo(losses, loss_type):
        sorted_losses = sorted(losses.items(), key=lambda x: x[1][loss_type], reverse=True)
        return [filename for filename, loss in sorted_losses[:10]]

    top10_bce = foo(losses, loss_type='bce_loss')
    top10_dice = foo(losses, loss_type='dice_loss')
    top10_total = foo(losses, loss_type='total_loss')

    for loss_type in ['bce_loss', 'dice_loss', 'total_loss']:
        print("Loss Type:", loss_type)

        for img_path in foo(losses, loss_type):
            label_path = losses[img_path]['label_path']
            prediction = losses[img_path]['prediction']

            image = np.load(img_path)
            label = np.load(label_path)

            img_path += f"\n{loss_type.upper()}={losses[img_path][loss_type]}"

            plot(image, label, prediction, filename=img_path)
