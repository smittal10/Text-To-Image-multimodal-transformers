### Evaluation metrics for different attacks


```shell
Follow the steps [here](https://gilberttanner.com/blog/detectron-2-object-detection-with-pytorch#:~:text=on%20my%20Github.-,Inference%20with%20pre%2Dtrained%20model,-Using%20a%20pre) to setup detectron2 repo to use that evaluation metric.


detection.py evaluates generated images and score it as 0 if a negated object was generated otherwise 1. A score of -1 is given to a generation where the detectron model fails to detect objects with any confidence.
```
---