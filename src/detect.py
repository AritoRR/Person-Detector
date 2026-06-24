import os
import sys

import cv2
import torch
import argparse

from script_path import WEIGHTS_PATH


def detect_image(image_path, weights_path=WEIGHTS_PATH, output_dir='outputs'):
    if not os.path.exists(weights_path):
        print(f"ERROR: {weights_path} not found")
        return

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    model.conf = 0.25

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: can't load {image_path}")
        return

    results = model(img)

    os.makedirs(output_dir, exist_ok=True)
    results.render()
    output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, results.imgs[0])

    print(f"Обнаружено {len(results.xyxy[0])} человек(а)")
    print(f"Результат сохранен в {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Детектор людей на изображениях')
    parser.add_argument('--image', type=str, help='Путь к изображению')
    parser.add_argument('--weights', type=str, default=WEIGHTS_PATH, help='Путь к весам')
    parser.add_argument('--output', type=str, default='outputs', help='Папка для сохранения')

    args = parser.parse_args()

    if not args.image:
        print("ERROR: укажите --image /path/to/image.jpg")
        sys.exit(1)

    detect_image(args.image, args.weights, args.output)


if __name__ == '__main__':
    main()
