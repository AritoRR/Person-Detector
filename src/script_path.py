import os


def jpath(*paths):
    return os.path.join(*paths)


TRAIN_PATH = '../yolov5/train.py'
VAL_PATH = '../yolov5/val.py'
DETECT_PATH = '../yolov5/detect.py'
WEIGHTS_PATH = '../runs/train/exp_person/weights/best.pt'
CONFIG_PATH = '../configs/coco_person.yaml'
MODEL_PATH = '../yolov5/models/yolov5s.yaml'
ANNOTATIONS_PATH = "../datasets/coco/annotations/"
ANNOTATIONS_OUTPUT = "../datasets/coco/"
LABELS_PATH = "../datasets/coco/labels/"
LABELS_OUTPUT = "../datasets/coco/labels_person/"
RUNS_PATH = '../runs'
MILITARY_PATH = "../datasets/military"
CONFIGS_PATH = "../configs"
HYP_SLOW_PATH = '../configs/hyps/hyp.scratch-low.yaml'
