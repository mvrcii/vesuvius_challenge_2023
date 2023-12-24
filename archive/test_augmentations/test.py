from models.data_modules import SegFormerDataModule
from utility.config_handler import Config


def main():
    config_path = "data/test_augmentations/config.py"
    config = Config.load_from_file(config_path)
    data_module = SegFormerDataModule(cfg=config)
    train_loader = data_module.train_dataloader()

    for (image, label) in train_loader:
        batch_idx = 0
        print(image)
        # plot(image[batch_idx], label[batch_idx])


if __name__ == '__main__':
    main()
