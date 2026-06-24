import os

TRAIN_PATH = os.path.join('..', 'yolov5', 'train.py')
VAL_PATH = os.path.join('..', 'yolov5', 'val.py')
WEIGHTS_PATH = '../runs/train/exp_person/weights/best.pt'
CONFIG_PATH = '../configs/coco_person.yaml'
MODEL_PATH = '../yolov5/models/yolov5s.yaml'
ANNOTATIONS_PATH = "../datasets/coco/annotations/"
ANNOTATIONS_OUTPUT = "../datasets/coco/"
LABELS_PATH = "../datasets/coco/labels/"
LABELS_OUTPUT = "../datasets/coco/labels_person/"
