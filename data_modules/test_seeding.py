from config_handler import Config
from data_modules.utils import generate_dataset


def test_seeding():
    cfg = Config.load_from_file("configs/segformer/b2/512/config.py")
    cfg.seed = -1
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = generate_dataset(cfg=cfg)
    print(train_image_paths[:10])


if __name__ == '__main__':
    test_seeding()
