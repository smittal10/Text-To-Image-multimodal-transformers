# import some common detectron2 utilities
from charset_normalizer import detect
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os
def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            # labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
            labels = labels
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels

coco = set()
with open("examples/data/coco-labels-2014_2017.txt","r") as f:
    for line in f.readlines():
        coco.add(line.strip())

root_dir = "negated_coco_figures"
images = os.listdir(root_dir)
images.sort()
# images = images[:10]

detections_dir = './detections_negated_coco'
if not os.path.exists(detections_dir):
    os.makedirs(detections_dir)
fp = open(os.path.join(detections_dir,'detection_results2.tsv'),'w')
# Create config
cfg = get_cfg()
cfg.merge_from_file("../detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

# Create predictor
predictor = DefaultPredictor(cfg)

for ind in range(0,len(images),5):
    ########################
    #get negated object list for this caption
    tokens = images[ind].split('_')[0].split()
    i=0
    negations = []
    while(tokens[i]!="no" and tokens[i]!="without" and i<len(tokens)-1):
        i+=1
    fg = -1
    for i in range(i+1,min(i+7,len(tokens))):
        if tokens[i].lower() in coco:
            fg = 0
        negations.append(tokens[i].lower())
    fp.write(images[ind]+"\t"+str(fg))
    #####################
    j=0
    while (j < 5):
        # get image
        im = cv2.imread(os.path.join(root_dir,images[ind+j]))
        # Make prediction
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        scores = outputs["instances"].to("cpu").scores if outputs["instances"].to("cpu").has("scores") else None
        classes = outputs["instances"].to("cpu").pred_classes.tolist() if outputs["instances"].to("cpu").has("pred_classes") else None
        labels = _create_text_labels(classes, scores, v.metadata.get("thing_classes", None))
        # print(labels)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imwrite(os.path.join(detections_dir,images[ind+j]), v.get_image()[:, :, ::-1])
        # cv2.imshow(v.get_image()[:, :, ::-1])
        reward = 1
        if len(labels) == 0:
            reward = -1
        for label in labels:
            if label in negations:
                reward = 0
                break
        fp.write("\t"+str(reward))
        j+=1
    fp.write("\n")
fp.close()
