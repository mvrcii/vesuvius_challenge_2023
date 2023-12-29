import random

import torch

from models.datasets.unet3dsf_dataset import UNET3D_SFDataset


class UNETR_SFDataset(UNET3D_SFDataset):
    def __init__(self, root_dir, images, transform, cfg, labels=None):
        super().__init__(cfg=cfg, root_dir=root_dir, images=images, transform=transform, labels=labels)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        # todo fix
        # so z modifications are easier below
        image = image.squeeze(0)

        # Z Axis Augmentation
        if random.random() < 0.3:
            image = torch.flip(image, dims=[0])

        # Random cut-and-paste augmentation
        if random.random() < 0.3:  # Adjust the probability as needed
            num_layers = image.shape[0]
            num_layers_to_cut = random.randint(1, 4)  # Randomly choose 1-4 layers to cut
            start_layer = random.randint(0, num_layers - num_layers_to_cut)  # Choose start layer for cut
            end_layer = start_layer + num_layers_to_cut  # End layer for cut

            # Extract the block of layers
            cut_block = image[start_layer:end_layer]

            # Choose a random position to paste the block
            paste_position = random.randint(0, num_layers - num_layers_to_cut)

            # Perform the cut and paste
            image = torch.cat((image[:paste_position], cut_block, image[paste_position + num_layers_to_cut:]), dim=0)

        # Probability of shifting layers
        if random.random() < 0.3:  # Adjust the probability as needed
            shift_amount = random.randint(1, 4)  # Random shift amount (1-4 layers)

            # Randomly decide to shift up or down
            shift_direction = random.choice(['up', 'down'])

            # Create a tensor of zeros for padding
            zero_padding = torch.zeros((shift_amount, *image.shape[1:]), dtype=image.dtype, device=image.device)

            if shift_direction == 'up':
                # Shift layers up and pad at the bottom
                image = torch.cat((image[shift_amount:], zero_padding), dim=0)
            else:
                # Shift layers down and pad at the top
                image = torch.cat((zero_padding, image[:-shift_amount]), dim=0)

        # todo fix, see above
        image = image.unsqueeze(0)
        # pad image to have 16 layers
        image = torch.cat([image, torch.zeros(1, 16 - image.shape[1], self.patch_size, self.patch_size)], dim=1)

        return image, label
