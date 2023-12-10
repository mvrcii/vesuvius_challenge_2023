# Workflow to create a dataset
## Preparation
   1. Git pull
   2. Binarize layered label files: `python util/binarize_label_layered.py`
   3. Download required fragments (if not done yet): `python util/batch_download_frags.py`
   4. Create config file in `CONFIG_PATH`: `configs/ARCHITECTURE/MODEL_TYPE/PATCH_SIZE/config.py`
         1) Patch Size: 512 (256, 64)
         2) Stride Factor: 2, 4 (Stride = Patch Size // Stride Factor)
         3) In Channels: 4 (8, 16, 32)

## Create the dataset
`python data/create_dataset.py <CONFIG_PATH>`

## Notes
- Image patches are only created once for a specific patch size to reduce overhead. 
- Existing image patches are recognized and will be skipped. 
- Non-existent image patches for new labels are  
- Label patches are cleaned up and recreated each time `create_dataset.py` is called up