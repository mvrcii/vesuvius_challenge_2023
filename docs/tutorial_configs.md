# How to use the Config system

This documentation gives a brief overview on how to use the established config system within this repository effectively
and without running into problems.

## 1. User Dependent Variables

If you want to use this repository on your local device, you must create a `conf_local.py` file and set the `work_dir`
path to the absolute path of your repository:

```python
# config_local.py for Developer XY
work_dir = r"/Users/XY/.../MT3"
```

## 2. Configs

The `configs` directory in the root repository contains all relevant configuration files to build a dataset and
successfully start a training run.
The core idea is, to always use model specific configuration files which are located within each specific model
directory, such as `segmentation/unetr_sf`.
**Important: Do not try to use a sub-config file (e.g. a dataset sub-config) directly for any python script!**

### 2.1 Model Configs

May be used to:

- train a new model
- perform inference with a trained model
- create a dataset
- download all fragments contained in fragment_ids

Model configuration files always contain a `_base_` list, containing relative paths starting from the projects root
directory to
sub-config files. Sub-configs are merged in a top-down scheme.

In the following example code, first the datasets sub-config is imported and subsequently the schedule. Existent
variables will be overwritten. Finally, the variables of the actual model config are parsed and also overwritten. The highest priority is given by the local config file, so be careful with what it contains.

Priority Order (1=highest):

1. Local Config
2. Model Config
3. Last Sub-Config
4. ...
5. First Sub-Config

`unetr_sf/64/config_stage2_starter.py`

```python
_base_ = [
    "configs/schedules/adamw_cosine_lr.py",
]

# further variables
```

### 2.2 Schedule Configs

This sub-config file should contain schedule dependent variables.

```python
# schedules/adamw_cosine_lr.py 
weight_decay = 0.01
lr = 2e-4
```

