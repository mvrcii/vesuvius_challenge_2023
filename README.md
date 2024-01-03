# MT3: Unearthing History with a Transformer Triad
Leveraging the synergy of UNETR, SegFormer, and Transformer Fragment IDs, MT3 employs an innovative masking technique to discern ink on ancient scrolls buried by Vesuvius' ash. This method carefully omits uncertain areas from loss calculations during training, ensuring precise and accurate machine learning interpretations without negative influence from ambiguous data.

## Our approach

We started off by handlabelling crackles up to a precision of 4 layers, these labels can be found in 
`archive\labels\handmade_labels\four_layer_handmade`.

We then trained multiple Segformer models with 4 in-channels on this data (passing 4 layers into the 4
in-channels of the model, discarding the 3D information).


The results of these models can be found in `data/labels/four_layer/binarized`.

We then continued to train models on the predictions of the previous models - which to our surprise produced better and 
better results - until we found our first letters. These can be seen in:
`data/labels/four_layer/binarized/lively-meadow-695-segformer-b2-231206-230820/20231005123336/inklabels_36_39.png`. 

We then shifted our approach from 4-layer models to a 12 layer model, specifically a UNETR + Segformer B5 
(similar to the kaggle competition winners). Since UNETR needs 16 layers, we zero-padded our input, due to the fact that in many 
fragments taking more than 12 layers would include faulty ink signals from higher / lower layers due to layer skipping
during segmentation.

We then trained these models by overlaying all of our 4 layer labels corresponding to the "best" 12 layer block. 
By having a model with 4 layer precision we could precisely tell which 12 layers were the best to combat layer skipping, which would otherwise introduce
noise and false letters.

From there on we repeated our approach of retraining on previous predictions. However, we included a crucial new detail: 
an "ignore" mask. Instead of giving the model a binary label (ink/no-ink), we added a third mask to manually
annotate areas where the predictions of the previous model were uncertain or obviously wrong. Instead of fixing
them, e.g. filling a hole in a very obvious interrupted vertical line, we marked these spots with an ignore mask, 
which effectively removed the underlying pixels from the loss calculation. This way the previously wrong prediction 
can not propagate to the next model iteration. With this method we also achieved to reduce the risk of accidentally manually annotating ink
where there is no ink. Surprisingly, following this approach, these ignored areas then iteratively improved as we trained more and more models, 
allowing us to iteratively reduce the area that we ignore in the loss.

Our first set of 12 layer labels can be found in `data/labels/twelve_layer` 

 A final big leap in performance was achieved by **decreasing** our patch size from 512x512 down to 128x128.
This made our model way more precise and also made us more confident in our predictions as it reduced the risk
for hallucinations.

The final results were created using an ensemble of 128x128 models and one 512x512 model, which improved the best 
aspects of both models. The 128 model contributed to sharper edges while the 512 model helped to reduce overall noise.

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

### 1. Set up the data locally
When running the docker container, you need to mount a local data directory to the data directory in the docker container.
The code expects a `data` directory on your local machine to contain a directory called `fragments`which in turn 
contains directories of the format
`fragment<id>` e.g. `fragment2023101284423`.
These fragment directories should contain the mask file (`mask.png`) and a subdirectory called `slices` 
which contains the .tif files numbered with 5 digits, e.g. `00032.tif`

    data/
    │
    └── fragments/
        │
        ├── fragment2023101284423/
        │   ├── mask.png
        │   └── slices/
        │       ├── 00001.tif
        │       ├── 00002.tif
        │       ...
        │       └── 00064.tif
###  2. Download the app.tar docker image from Google Drive (link provided in submission mail)
###  3. Load the docker image
```
docker load -i app.tar
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
This will create a directory ``fragments_contrasted`` in your local data directory, load all fragment 
files from your original ``fragments`` directory, preprocess them (increase contrast) and save them to 
``fragments_contrasted``.

### 6. Running inference

````commandline
python inference_submission.py
````
Note: This can take very long due to high TTA, the script has a variable ``tta=True`` which can be 
set to false, this will speed up inference by ~ 8x. (but produce slightly noisier images)

### 7. Ensemble and create images
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
We are quite certain that our model is not prone to hallucination, due to the fact that a patch size
of 128x128 is not nearly enough to cover a notable part of a letter. Furthermore our predictions match very
closely to those that have been shared publicly on the discord by other users. The 512x512 model's patch size 
could in theory be big enough to hallucinate notable parts of letters, but its predictions very closely match 
those of the 128x128 models.
We also trained models with a 64x64 resolution, which also confirmed the results found by the other models, 
but slightly underperforming them due to higher noise content.
