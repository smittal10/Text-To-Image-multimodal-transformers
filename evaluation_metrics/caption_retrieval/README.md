### Evaluation metrics for perturbations in caption


#### Generate images on original (coco/annotations/captions_val2017.json) and pertubed captions (coco/annotations/captions_val2017_bt.json) from the model which is to be evaluated, set the model path here, "dalle/models/__init__.py"

##### Samples with clip re-ranking
```shell
python sampling_batched_together2.py
```
##### Samples without clip re-ranking
```shell
python sampling_together_no_clip.py
```

#### Evaluate
caption_retrieval.py evaluates generated images and outputs Recall@1, Recall@5, REcall@10. Set 'data_root' to the diectory containing the images.

##### To evaluate on images generated from original captions, change the following variables as:

valid_dir = os.path.join(data_root, 'orig')
imgs_path = os.path.join(data_root, 'orig_map.tsv')

##### To evaluate on images generated from pertubed captions, change the following variables as:

valid_dir = os.path.join(data_root, 'bt')
imgs_path = os.path.join(data_root, 'bt_map.tsv')


