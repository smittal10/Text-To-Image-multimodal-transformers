### Evaluation metrics for different attacks

#### Using Object detection model to test negated capions

Follow the steps [here](https://gilberttanner.com/blog/detectron-2-object-detection-with-pytorch) to setup detectron2 repo to use this evaluation metric.

#### Generate images on original (negations/original_captions_filtered.txt) and negated captions (negations/negated_coco_filtered.txt) from the model which is to be evaluated, set the model path here, "dalle/models/__init__.py"

```shell
python sampling_batched_together.py
```

detections_metric.py evaluates generated images and score it as 0 if a negated object was generated otherwise 1. A score of -1 is given to a generation where the detectron model fails to detect any objects with confidence.


