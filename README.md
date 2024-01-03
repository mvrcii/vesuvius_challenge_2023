# MT3: Unearthing History with a Transformer Triad
Leveraging the synergy of UNETR, SegFormer, and Transformer Fragment IDs, MT3 employs an innovative masking technique to discern ink on ancient scrolls buried by Vesuvius' ash. This method carefully omits uncertain areas from loss calculations during training, ensuring precise and accurate machine learning interpretations without negative influence from ambiguous data.

## Our approach

We started off by hand-labelling crackles up to a precision of 4 layers, these labels can be found in 
`archive\labels\handmade_labels\four_layer_handmade`.

To get started, we trained multiple SegFormer models with 4 input layers on the handlabelled data by passing 4 stacked data layers into the
input layers of the model, discarding the 3D information.


The results of these models can be found in `data/labels/four_layer/binarized`.

We then continued to train models on the predictions of the previous models - which to our surprise produced better and 
better results - until we found our first letter. These can be seen in:
`data/labels/four_layer/binarized/lively-meadow-695-segformer-b2-231206-230820/20231005123336/inklabels_36_39.png`. 

We then shifted our approach from 4-layer models to a 12-layer model, specifically a UNETR as encoder combined with a Segformer B5 as decoder
(similar to the kaggle competition winners). We found that, for most fragments, the use of more than 12 layers would result in incorrect colour signals from higher and lower layers due to layer skipping during segmentation. Since the UNETR architecture requires 16 input layers we had to zero-pad our input.

We then trained these models by overlaying all of our 4-layer labels corresponding to the "best" 12 layer block. By having a model with 4-layer precision we could precisely tell which 12 layers were the best to combat layer skipping, which would otherwise introduce noise and false letters.

From there on we repeated our approach of retraining on previous predictions. However, we included a crucial new detail: 
an "ignore" mask. Instead of giving the model a binary label (ink/no-ink), we added a third mask to manually
annotate areas where the predictions of the previous model were uncertain or obviously wrong. Instead of fixing
them, e.g. filling a hole in a very obvious interrupted vertical line, we marked these spots with an ignore mask, 
which effectively removed the underlying pixels from the loss calculation. This way the previously wrong prediction 
can not propagate to the next model iteration. With this method we also achieved to reduce the risk of accidentally manually annotating ink
where there is no ink. By following this approach, we were able to iteratively improve the areas that were previously ignored as we trained more models. 
This allowed us to gradually reduce the area that we ignore in the loss function.

Our first set of 12-layer labels can be found in `data/labels/twelve_layer` 

A significant improvement in performance was achieved by **reducing** the patch size from 512x512 to 128x128.
This increased the precision of our model and boosted our confidence in our predictions by reducing the risk of hallucinations.

The final result is a combination of an ensemble of 128x128 models with a 512x512 model, which has the best features of both models. 
features of both models. The 128 model contributed to sharper edges, while the 512 model helped to reduce overall noise.

## Requirements:
For inference 24 GB VRAM is enough (e.g. RTX 4090). The models were trained on 8xH100s (80GB VRAM each),
but with smaller batch sizes, training on 24GB is possible.

When reproducing the training, the dataset creation can take up to 90GB for the 128x128 model, 
and another 90GB for the 512x512 model, so make sure ~ 200GB of free disk space is available 
(not counting the disk space already occupied by the raw fragment tif files).
#### Note
To make communication about the different fragments easier during development, we assigned an alias name to each 
fragment e.g. `SUNSTREAKER` == `20231031143852`. The full mapping of fragment aliases to their IDs can be seen 
in `utility/fragments.json` inside the docker image. References to these aliases
can be found through the code, configs etc.

### 1. Setting up the data locally
When running the docker container, you need to mount a local data directory to the data directory in the docker container.
The code expects a directory on your local machine called `fragments` which contains subdirectories of the format `fragment<id>` e.g. `fragment2023101284423`.
These fragment directories must contain the mask file (`mask.png`) and a subdirectory called `layers` 
which contains the .tif files numbered with 5 digits, e.g. `00032.tif`

    fragments/
    │
    ├── fragment2023101284423/
    │   ├── mask.png
    │   └── layers/
    │       ├── 00001.tif
    │       ├── 00002.tif
    │       ...
    │       └── 00064.tif
###  2. Download the app.tar docker image from Google Drive (link provided in submission mail)
###  3. Load the docker image
```
docker load -i app.tar
```
### 4. Running the docker image

```
docker run -it --gpus all -v <your_local_data_directory>:/usr/src/app/fragments submission bash
```
### 5. Data Preprocessing
Once inside the interactive shell, your working directory should be `/usr/src/app`.
From here you can run 
````commandline
python convert/preprocess_images.py 
````
This will create a directory ``fragments_contrasted`` in your local data directory, load all fragment 
files from your original ``fragments`` directory, preprocess them (increase contrast) and save them to 
``fragments_contrasted``.

### 6. Running the inference

````commandline
python inference_submission.py
````
Note: This can take very long due to high TTA, the script has a variable ``tta=True`` which can be 
set to false, this will speed up inference by ~ 8x. (but produce slightly noisier images)

### 7. Ensemble and creating images
Once Inference is done, run

````commandline
python ensemble_submission.py
````
This will combine the previously created predictions, and save the resulting images to your local
`data/ensemble_results`


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
Where ``config_path`` points to a ``config_xxxxx.py`` file. Make sure to adjust the parameter ```train_batch_size```
in the corresponding config according to your hardwar requirements, when training with less than 80 GB VRAM.

The 4 configs used for our ensemble submission are placed under `configs/submission` 
1. ``olive-wind.py`` (128x128)
2. ``curious-rain.py``(128x128)
3. ``desert-sea.py`` (128x128)
4. ``wise-energy.py``(512x512)

We provide trained checkpoints of all of these models.

Since our approach was to iteratively train models on their own predictions (e.g. train on 4422) => infer on 4422 => 
repeat, we also included configs for models that do not get any labels for the fragments they are inferring on:

These are placed in `configs/segmentation/one_out`.
Here the naming scheme indicates the fragment left out of the training set, e.g. `config_sunstreaker.py` does not
include the `SUNSTREAKER` fragment in its training set.  (Note, check ``utils/fragments.json`` for alias-to-ID mapping.)

We provide trained checkpoints for the following fragments 

*FragmentID left out of training : checkpoint path*

1.``20231016151002`` : ``dazzling-haze-1197-unetr-sf-b5-231231-223336``

2.``20230702185753`` : ``lucky-field-1198-unetr-sf-b5-231231-223422``

3.``20231210121321`` : ``icy-disco-1199-unetr-sf-b5-231231-223530``

The ``supplementary`` directory in google drive contains their raw predictions, proving the generalizational 
capabilities, and confirming the correctness of our ensemble predictions which were partially trained on previous 
predictions on the same fragment.

## Running inference with a trained model
Inference can be run via
```
python multilayer_approach/infer_layered_segmentation_padto16.py <checkpoint_folder> <fragment_id>
```
Where checkpoint_folder points to the folder (named with a wandb name, e.g. icy-disco-1199-unetr-sf-b5-231231-223530)
This name will be automatically generated when the train run is started and printed to the console when the training starts.

The resulting npy files will be stored in `inference/results/fragment_<id>/<checkpiont_folder>`

You can quickly extract the results to your local data directory as an image using
````commandline
python get_single_results.py <fragment_id> <checkpoint_name>
````
E.g. `python get_single_results.py 20231016151002 dazzling-haze`

Note: Any uniquely identifying substring of the checkpoint works here, so the full name is not required.


### Hallucination mitigation
We are confident that our model is not prone to hallucination because a patch size of 128x128 is not sufficient to cover a significant portion of a letter.
Additionally, our predictions closely match those shared publicly on Discord by other users. 
While the 512x512 model's patch size could theoretically lead to hallucination of notable parts of letters, its predictions also closely match those of the 128x128 models.
A 64x64 resolution was also trained and confirmed the results of the other models, but underperformed slightly due to higher noise.he results found by the other models, 
but slightly underperforming them due to higher noise content.
