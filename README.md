## Source
https://www.kaggle.com/code/ryches/1st-place-solution#dataset
## Env 
Same ``env`` as mmsegmentation experiment.

Then only additional dependency should be albumentations:

## Dataset Creation
```bash
create_dataset(data_root_dir=CFG.data_root_dir, dataset_type='train')
```

```commandline
pip install albumentations
```

# Todo: Train Loop (not in source)
However we have train config and example of test dataloader, and model is a nn.Module, so should be quite straight forward.

### (And later inference / ensemble, see source for more code)