import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import os

from conf import CFG

mode = 'train'


def read_image(fragment_id):
    images = []

    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    for i in tqdm(idxs):
        # image = cv2.imread(CFG.comp_dataset_path + f"{mode}/{fragment_id}/surface_volume/{i:02}.tif", 0)
        img_path = os.path.join("A:\\", "projects_a", "Python", "vesuv", "data", "fragments", "fragment2", "slices", f"{i:05}.tif")
        image = cv2.imread(img_path, 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    return images


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = np.array(images)
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(self.images[idx])
        data = self.transform(image=image)
        image = data['image']
        return image[None, :, :, :]


def make_test_dataset(fragment_id):
    test_images = read_image(fragment_id)

    x1_list = list(range(0, test_images.shape[1] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, test_images.shape[0] - CFG.tile_size + 1, CFG.stride))

    test_images_list = []
    xyxys = []
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if test_images[y1:y2, x1:x2].max() != 0:
                if not os.path.exists(f"{x1}_{y1}_{x2}_{y2}.npy"):
                    np.save(f"{x1}_{y1}_{x2}_{y2}.npy", test_images[y1:y2, x1:x2])
                test_images_list.append(f"{x1}_{y1}_{x2}_{y2}.npy")
                xyxys.append((x1, y1, x2, y2))
    del test_images
    gc.collect()
    xyxys = np.stack(xyxys)

    test_dataset = CustomDataset(test_images_list, CFG, transform=get_transforms(data='valid', cfg=CFG))

    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    return test_loader, xyxys

test_loader, xyxys = make_test_dataset(2)
