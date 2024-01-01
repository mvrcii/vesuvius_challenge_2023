# ReadMe
How to reproduce the images located in `images`

## General approach
This is a rough description on how we arrived with these results (Running out of time due to deadline approaching), but we might submit this in more detail later if it is required. (All the technical details to reproduce the final results should be listed below.)

We started off by handlabelling crackles up to a precision of 4 layers, these labels can be found in 
`archive\labels\handmade_labels\four_layer_handmade`. We then trained multiple Segformer models with 4 in-channels on this data. 
The results of these models can be found in `data/labels/four_layer/binarized`
We then continued to train models on the predictions of the previous models which to our surprise produced better and better results, until we found
first letters, these can be seen for example in `data/labels/four_layer/binarized/lively-meadow-695-segformer-b2-231206-230820/20231005123336/inklabels_36_39.png` 

We then shifted from 4-layer models to a 12 layer model, specifically a UNETR + Segformer B5 (inspired by the kaggle competition winners.) And trained these models by overlaying all of our 4 layer labels corresponding to the "best" 12 layer block. (By having a model with 4 layer precision we could precisely tell which 12 layers were the best, to combat layer skipping)

Here we repeated our approach of retraining on previous predictions, however we included a crucial new detail: An "ignore" mask. Instead of giving the model a binary label (ink/no-ink), we added a third mask, to manually annotate areas where the predictions of the previous model were uncertain / obviously wrong. Instead of fixing them, e.g. filling a hole in a very obvious interrupted vertical line, we marked these spots with an ignore mask, which effectively removed the underlying pixels from the loss calculation. this way the previously wrong prediction can not propragate to the next model. And also we don't risk accidentally annotating ink where there is no ink. Surprisingly, these ignored areas then iteratively improved as we trained more and more models, 
allowing us to iteratively reduce the area that we ignore in the loss.
Our first set of 12 layer labels can be found in `data/labels/twelve_layer` 


## Requirements:
For inference 24 GB VRAM is enough (e.g. RTX 4090). The models were trained on 8xH100s (80GB VRAM each), but with smaller batch sizes, training on 24GB is possible.

When reproducing the training, the dataset creation can take up to 90GB for the 128x128 model, and another 90GB for the 512x512 model, so make sure ~ 200GB of free disk space is available (not counting the disk space already occupied by the raw fragment tif files).
#### Note
To make communication about the different fragments easier during development, we assigned an alias name to each fragment e.g. `SUNSTREAKER` == `20231031143852`. The full mapping of fragment aliases to their IDs can be seen in `utility/fragments.json` inside the docker image.
### 1. Set up the data locally

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
###  2. Download the app.tar docker image from Google Drive (link provided in submission mail)
###  3. Load the docker image
```
docker load -i myapp.tar
```
### 4. Run the docker image

```
docker run -it --gpus all -v <your_local_data_directory>:/usr/src/app/data submission bash
```
### 5. Data Preprocessing
Once inside the interactive shell, your working directory should be `/usr/src/app`.
From here you can run 
````commandline
python convert/preprocess_images.py 
````
This will create a directory ``fragments_contrasted`` in your local data directory, load all fragment files from your original ``fragments`` directory, preprocess them (increase contrast) and save them to ``fragments_contrasted``.

### 6. Running inference

````commandline
python inference_submission.py
````
Note: This can take very long due to high TTA, the script has a variable ``tta=True`` which can be set to false, this will speed up inference by ~ 8x. (but produce slightly noisier images)

### 7. Ensemble and create images
Once Inference is done, run

````commandline
python ensemble_submission.py
````
This will combine the previously created predictions, and save the resulting images to your local `data/ensemble_results`


# Training 
### Dataset creation
Before a model can be trained, the dataset has to be created.
Make sure you completed the step `Data Preprocessing` mentioned above, then run: 
```commandline
python multilayer_approach/create_dataset_segmentation.py configs/submission/config_dataset.py
```

### Training
Any model can be trained by calling
```commandline
python train.py <config_path>
```
Where ``config_path`` points to a ``config_xxxxx.py`` file. Make sure to adjust the parameter ```train_batch_size``` in the corresponding config, according to your hardwar requirements, when training with less than 80 GB VRAM.

The 3 configs used for our ensemble submission are placed under `configs/submission` 
1. ``olive_wind.py`` (128x128)
2. ``curious_rain.py``(128x128)
3. ``wise_enegry.py``(512x512)

Since our approach was to iteratively train models on their own predictions (e.g. train on 4422) => infer on 4422 => repeat, we also included configs for models that did not get any labels for the fragments they are inferring on:

These are placed in `configs/segmentation/unetr_sf/one_out`.
Here the naming scheme indicates the fragment left out of the training set, e.g. `config_sunstreaker.py` does not include the `SUNSTREAKER` fragment in its training set. 
## Running inference with a trained model
Inference can be run via
```
python multilayer_approach/infer_layered_segmentation_padto16.py <checkpoint_folder> <fragment_id>
```
Where checkpoint_folder points to the folder (named with a wandb name, e.g. icy-disco-1199-unetr-sf-b5-231231-223530)
This name will be automatically generated when the train run is started.



### Hallucination mitigation
We are very certain that our model is not prone to hallucination. Mostly due to the fact that a patch size of 128x128 is not nearly enough to cover a notable part of a letter, additionally our predictions match very closely to those that have been shared publicly on the discord of other users. The 512x512 model's patch size could in theory be big enough to hallucinate notable parts of letters, but its predictions very closely match those of the 128x128 models.




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



