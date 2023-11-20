# How to use the Config system

This documentation gives a brief overview on how to use the established config system within this repository effectively and without running into problems.

## 1. User Dependent Variables
If you want to use this repository on your local device, you must create a `conf_local.py` file and set the `work_dir`
path to the absolute path of your repository:

```python
# config_local.py for Developer XY
work_dir = r"/Users/XY/.../kaggle1stReimp"
```

## 2. Configs
The `configs` directory in the root repository contains all relevant configuration files to build a dataset and successfully start a training run.
The core idea is, to always use model specific configuration files which are located within each specific model directory, such as `segformer`. 
**Important: Do not try to use a sub-config file (e.g. a dataset sub-config) directly for any python script!**

### 2.1 Model Configs
May be used to:
- train a new model
- perform inference with a trained model
- create a dataset

Model configuration files always contain a `_base_` list. This list may contain relative paths starting from the projects root directory to 
sub-configuration files. The sub-configs are merged in a top-down scheme. In the following example code, first the datasets sub-config is loaded,
subsequently the schedule is imported and thereby adds new variables to the config. Already existent variables are overwritten.
Finally, the variables of the actual model config are being parsed and also updated / overwritten. The highest priority is given by the 
local config file, so be careful what it contains.

Priority Order (1=highest):
1. Local Config
2. Model Config
3. Last Sub-Config
4. ...
5. First Sub-Config

b0_512px_64ch.py:
```python
_base = [
    "configs/datasets/single/optimus_bumblebee_512px_64ch.py",
    "configs/schedules/adamw_cosine_lr.py",
]

# further variables
```

### 2.2 Dataset Configs
This sub-config file should contain dataset dependent variables and is split into `k_fold` and `single` configs.
Both modes make use of the list variables `train_frag_ids` and `val_frag_ids`. Furthermore, this config should contain the following variables:

- augmentations: The augmentations for the training and validation dataset
- calc_mean_std: Whether to calculate the mean and std for each specific channel across all training fragments
- patch_size: The patch size for the dataset generation
- stride: The stride for the dataset generation
- dataset_in_chans: The amount of fragment dimensions/channels that are used to generate a dataset
- ink_ratio: A threshold that determines how much ink percentage a patch must contain to be used during dataset generation
- ...

#### k_fold
This mode takes all training fragment ids to generate the train dataset, and all validation fragment ids to respectively generate the validation dataset.

```python
# datasets/k_fold/optimus_bumblebee_512px_64ch.py
from constants import OPTIMUS_FRAG_ID, BUMBLEBEE_FRAG_ID, ULTRA_MAGNUS_FRAG_ID

k_fold = True
train_frag_ids = [OPTIMUS_FRAG_ID, BUMBLEBEE_FRAG_ID]
val_frag_ids = [ULTRA_MAGNUS_FRAG_ID]
```

#### single
This mode takes all training fragment ids, generates the dataset and subsequently creates the validation dataset based
on the created dataset with a given `train_split`.

```python
# datasets/single/optimus_bumblebee_512px_64ch.py
from constants import OPTIMUS_FRAG_ID, BUMBLEBEE_FRAG_ID, ULTRA_MAGNUS_FRAG_ID

k_fold = False
train_frag_ids = [OPTIMUS_FRAG_ID, BUMBLEBEE_FRAG_ID, ULTRA_MAGNUS_FRAG_ID]
val_frag_ids = []
train_split = 0.8
```

### 2.3 Schedule Configs
This sub-config file should contain schedule dependent variables.

```python
# schedules/adamw_cosine_lr.py 
weight_decay = 0.01
lr = 2e-4
```

