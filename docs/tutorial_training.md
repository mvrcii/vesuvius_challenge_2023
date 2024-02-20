# Getting Started with UNETR_SF

### 1. Download layers files for fragment(s)
On Windows it is easier to call this command within a bash (e.g. git-bash)
`./scripts/utils/download_fragment.sh 20231012184423 "20 45"`

### 2. Preprocess layer files

Preprocess all layer files by enhancing the contrast. 
Moves the contrasted layer files to the correct directory for a follow-up dataset creation: 
`python ./convert/preprocess_images.py`

### 3. Create the dataset

Create the dataset for the segmentation task:
`python multilayer_approach/create_dataset_segmentation.py configs/segmentation/unetr_sf/64/config_stage2_starter.py`