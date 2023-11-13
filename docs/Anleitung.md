# Todos
- Repo aufräumen
- Train so anpassen, dass checkpoints in pro run getrennte work_dirs gepackt werden
- infer_full_image so anpassen dass checkpoint als param übergeben werden kann (inference easy auf cluster ohne script anpassung)
- (maybe direkt nur pngs speichern, siehe generate_pngs)
- Validation in train loop einbauen, use following code:

```python
from datasets import load_metric

metric = load_metric("mean_iou")

# ...
# in train oop
with torch.no_grad():
    prediction = ..
    label = ..
    
    # note that the metric expects predictions + labels as numpy arrays
    
    # todo, binarize predictions here??
    metric.add_batch(predictions=prediction.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

# let's print loss and metrics every 100 batches
if idx % 100 == 0:
    metrics = metric._compute(num_labels=len(id2label),
                              ignore_index=255,
                              reduce_labels=False,  # we've already reduced the labels before)
                              )
   print("Loss:", loss.item())
          print("Mean_iou:", metrics["mean_iou"])
          print("Mean accuracy:", metrics["mean_accuracy"])
```
- Augmentation einbauen, how to augment (16, 512, 512)? Does this even work properly? Upscale labels to transform them (see create_dataset on how to resize)
- Augmentation of training set, shift the 16layer block up and down / scramble the 16 layers
- Try out different segformer versions with different batch sizes
- Try out segformers trained on 1024 with patch sizes 1024
- Try out different learning rate schedules (however current one performs quite well)
- Adjust infer_full_image_overlap to work with new model (look at infer_full_image on how to work with new model) 
- Try inserting more layers? 64 channel segformer? Inspect segformer params, how many params does it add to increase input channels?
- Download all remaining fragments, do k-fold evaluation, adjust scripts like create_dataset to work with multiple fragments