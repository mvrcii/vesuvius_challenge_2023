# Scroll branch

# Source
https://www.kaggle.com/code/ryches/1st-place-solution#dataset


https://smp.readthedocs.io/en/latest/index.html

# Configs
By default, the git-tracked main config `conf.py` is being used. In the case that a local config file with the correct 
naming scheme is present, variables within the main config will be overwritten. If no local config is present, the main 
config is being used by default.

## Local Dev
Create a `conf_local.py` file within the root directory. Parameters set in this config receive a higher priority than
the main config.

## Cluster Dev
The cluster automatically uses the git-tracked main config `conf.py`.

## (Conda) Virtual Environment
Same ``env`` as mmsegmentation experiment.

Then only additional dependency should be albumentations:

## Dataset Creation
The dataset can be created with `create_dataset.py`.