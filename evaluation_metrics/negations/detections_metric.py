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

coco = []
with open("examples/data/coco-labels-2014_2017.txt","r") as f:
    for line in f.readlines():
        coco.append(line.strip())

# Create config
cfg = get_cfg()
cfg.merge_from_file("../detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.DEVICE='cpu'
# Create predictor
predictor = DefaultPredictor(cfg)
#path to generated images
img_dir = "negations_finetuned"
original_file = img_dir+"/orig_map.tsv"
negated_file = img_dir+"/bt_map.tsv"

# detections_dir = img_dir +'/detections_negated_coco'
# if not os.path.exists(detections_dir):
#    os.makedirs(detections_dir)
fp = open(os.path.join(img_dir,'detection_results.tsv'),'w')


with open(original_file,"r",encoding='utf8') as orig, open(negated_file,'r',encoding='utf8') as neg:
    j=0
    for o,n in zip(orig.readlines(),neg.readlines()):
        # j+=1
        # if j==10:
        #     break
        n = n.strip().split('\t')
        o = o.strip().split('\t')
        ########################
        #get negated object list for this caption
        negated1 = o[1].split(" with ")[1]
        negations = []
        for object in coco:
            if object in negated1:
                negations.append(object.lower())
        #####################
        orig_img_path = os.path.join(img_dir,'orig',o[0]+'.png')
        n_img_path = os.path.join(img_dir,'bt',n[0]+'.png')
        ####################
        # get  original image and pass thru detectron2
        im = cv2.imread(orig_img_path)
        # Make prediction
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        scores = outputs["instances"].to("cpu").scores if outputs["instances"].to("cpu").has("scores") else None
        classes = outputs["instances"].to("cpu").pred_classes.tolist() if outputs["instances"].to("cpu").has("pred_classes") else None
        labels = _create_text_labels(classes, scores, v.metadata.get("thing_classes", None))
        # print(labels)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imwrite(os.path.join(detections_dir,o[0]+".png"), v.get_image()[:, :, ::-1])
        # cv2.imshow(v.get_image()[:, :, ::-1])
        reward = 0
        if len(labels) == 0:
            reward = -1
        for label in labels:
            if label in negations:
                reward = 1
                break
        fp.write(o[1] + "\t" +str(reward) + "\t" )
        #####################################
        #####################################
        # get  negation image and pass thru detectron2
        im = cv2.imread(n_img_path)
        # Make prediction
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        scores = outputs["instances"].to("cpu").scores if outputs["instances"].to("cpu").has("scores") else None
        classes = outputs["instances"].to("cpu").pred_classes.tolist() if outputs["instances"].to("cpu").has("pred_classes") else None
        labels = _create_text_labels(classes, scores, v.metadata.get("thing_classes", None))
        # print(labels)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imwrite(os.path.join(detections_dir,n[0]+".png"), v.get_image()[:, :, ::-1])
        # cv2.imshow(v.get_image()[:, :, ::-1])
        reward = 0
        if len(labels) == 0:
            reward = -1
        for label in labels:
            if label in negations:
                reward = 1
                break
        fp.write(n[1] + "\t" +str(reward) + "\n" )
        #####################################
fp.close()
