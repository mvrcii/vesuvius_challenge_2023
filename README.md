# ReadMe
How to reproduce the images located in `images`

#### Requirements:
For inference 24 GB VRAM is enough (e.g. RTX 4090). The models were trained on 8xH100s (80GB VRAM each), but with smaller batch sizes, training on 24GB is possible.

When reproducing the training, the dataset creation can take up to 90GB for the 128x128 model, and another 90GB for the 512x512 model, so make sure ~ 200GB of free disk space is available (not counting the disk space already occupied by the raw fragment tif files).
#### 1. Set up the data locally

The code expects a `data` directory to contain a directory called `fragments`which in turn contains directories of the format
`fragment<id>` e.g. `fragment2023101284423`.
These fragment directories should contain the mask file (`mask.png`) and a subdirectory called `slices` which contains the .tif files numbered with 5 digits, e.g. `00032.tif`

    data/
    │
    └── fragments/
        │
        ├── fragment2023101284423/
        │   ├── mask.png
        │   └── slices/
        │       ├── 00001.tif
        │       ├── 00002.tif
        │       ├── 00003.tif
        │       ...
        │       └── [Other .tif files]
####  2. Download the app.tar docker image
####  3. Load it using
```
docker load -i myapp.tar
```
#### 4. Run it using

```
docker run -it --gpus all -v <your_local_data_directory>:/usr/src/app/data submission bash
```
#### Data Preprocessing
Once inside the interactive shell, your working directory should be `/usr/src/app`.
From here you can run 
````commandline
python convert/preprocess_images.py 
````
This will create a directory ``fragments_contrasted`` in your local data directory, load all fragment files from your original ``fragments`` directory, preprocess them (increase contrast) and save them to ``fragments_contrasted``.

#### 5. Running inference

````commandline
python multilayer_approach/infer_layered_segmentation_padto16.py olive-wind-1194-unetr-sf-b5-231231-064008 20231012184423
````

# Training 
### Dataset creation
Before a model can be trained, the dataset has to be created.
Make sure you completed the step `Data Preprocessing` mentioned above. To create the `fragments_contrasted` directory. Once this is done, call 
```commandline
python multilayer_approach/create_dataset_segmentation.py configs/submission/config_dataset.py
```

### Training
Any model can be trained by calling
```commandline
python train.py <config_path>
```
Where ``config_path`` points to a config_xxxxx.py file. Make sure to adjust the parameter ```train_batch_size``` according to your hardwar requirements, when training with less than 80 GB VRAM.

The 3 configs used for our ensemble submission are placed under `configs/submission` 
1. ``olive``



x





x




x
x



x




x










x








x








x









x











x



x
x
x
x
x




x







x




x









x












x








x












x





x












x








x












x












x












x












x












x












x












x












x












x









x












x



