All scripts skip a patch if it has an invalid shape or is fully masked.

| Script                              | Ignore                   | Ink                                                    |
|-------------------------------------|--------------------------|--------------------------------------------------------|
| 1: create_dataset_segmentation.py   | Skip if ``> 95`` ignored | Creates all patches independent of ink                 |
| 2: create_dataset_classification.py | Skip if ``> 0`` ignored  | Create all ``ink_p==0`` and ``ink_p >= cfg.ink_ratio`` |


## Scripts building on 1: create_dataset_segmentation.py
unet3dsf_datamodule.py // unetrsf_datamodule.py // unet3d_datamodule.py (used by vit3d classification):

1. Skips patches with ``ignore_p > cfg.max_ignore_threshold`` (if config has this attr, otherwise skips nothing here)
2. If ```cfg.take_full_dataset == True``` => doesn't do any ink balancing
3. If ```cfg.take_full_dataset == False```:
   - Takes all the ``ink_p >=  cfg.ink_ratio`` samples (ink patches) => counts them to variable `ink_count`
   - Takes ```cfg.no_ink_sample_percentage * ink_count``` number of ``ink_p==0`` samples (full black patches)

