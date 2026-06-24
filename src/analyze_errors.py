import os
import torch
from src.script_path import WEIGHTS_PATH


def analyze_predictions(weights_path=WEIGHTS_PATH, data_path='./coco_person_final.yaml'):
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found")
        return

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

    results = model.val(
        data=data_path,
        batch_size=16,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
    )

    print("\n--- АНАЛИЗ ОШИБОК ---")
    print(f"TP (True Positives):   {results.tp}")
    print(f"FP (False Positives):  {results.fp}")
    print(f"FN (False Negatives):  {results.fn}")
    print(f"Precision:             {results.p:.4f}")
    print(f"Recall:                {results.r:.4f}")
    print(f"mAP@0.5:               {results.map50:.4f}")
    print(f"mAP@0.5:0.95:          {results.map:.4f}")

    return results


def main():
    analyze_predictions()


if __name__ == '__main__':
    main()
