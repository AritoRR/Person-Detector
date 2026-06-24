# src/make_person_dataset.py
import json
import os
import shutil
from pathlib import Path
import random

from script_path import ANNOTATIONS_PATH, ANNOTATIONS_OUTPUT, LABELS_PATH, LABELS_OUTPUT


def filter_labels_in_file(src_path, dst_path):
    if not os.path.exists(src_path):
        return False

    with open(src_path, 'r') as f:
        lines = f.readlines()

    # Оставляем только строки, начинающиеся с "0 " (класс person)
    filtered_lines = [line for line in lines if line.strip().startswith('0 ')]

    # Записываем результат
    with open(dst_path, 'w') as f:
        f.writelines(filtered_lines)

    return True

def filter(split):
    print("1. Загружаю аннотации COCO...")
    with open(os.path.join(ANNOTATIONS_PATH, f"instances_{split}2017.json"), 'r') as f:
        data = json.load(f)

    person_image_ids = set()
    for ann in data['annotations']:
        if ann['category_id'] == 1:  # человек
            person_image_ids.add(ann['image_id'])

    print(f"Найдено {len(person_image_ids)} изображений с людьми.")

    id_to_filename = {}
    for img in data['images']:
        if img['id'] in person_image_ids:
            id_to_filename[img['id']] = img['file_name']

    with open(os.path.join(ANNOTATIONS_OUTPUT, f"{split}_person.txt"), 'w') as f:
        for img_id in sorted(person_image_ids):
            f.write(f"./images/{split}2017/" + id_to_filename[img_id] + '\n')
    text = os.path.join(ANNOTATIONS_OUTPUT, f"{split}_person.txt")
    print(f"Список сохранен в {text}")

    if os.path.exists(os.path.join(LABELS_PATH, f"{split}2017")):
        print("2. Фильтрую YOLO-лейблы...")
        os.makedirs(os.path.join(LABELS_OUTPUT, f"{split}2017"), exist_ok=True)

        person_filenames = {Path(name).stem for name in id_to_filename.values()}

        copied = 0
        for label_file in os.listdir(os.path.join(LABELS_PATH, f"{split}2017")):
            if label_file.endswith('.txt'):
                stem = Path(label_file).stem
                if stem in person_filenames:
                    src = os.path.join(os.path.join(LABELS_PATH, f"{split}2017"), label_file)
                    dst = os.path.join(os.path.join(LABELS_OUTPUT, f"{split}2017"), label_file)
                    filter_labels_in_file(src, dst)
                    copied += 1
        text = os.path.join(LABELS_OUTPUT, f"{split}2017")
        print(f"Скопировано {copied} файлов разметки в {text}")
    else:
        print("Папка с лейблами не найдена. Пропускаю копирование.")

def main():
    filter('train')
    with open(os.path.join(ANNOTATIONS_OUTPUT, f"train_person.txt"), 'r') as f:
        all_images = [line.strip() for line in f if line.strip()]


    random.seed(42)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * 0.8)

    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    # Сохраняем
    with open(os.path.join(ANNOTATIONS_OUTPUT, f"train_person_split.txt"), 'w') as f:
        f.write('\n'.join(train_images))

    with open(os.path.join(ANNOTATIONS_OUTPUT, f"test_person.txt"), 'w') as f:
        f.write('\n'.join(test_images))
    print(f"\nTrain: {len(train_images)} изображений")
    print(f"Test:  {len(test_images)} изображений\n")

    filter('val')

if __name__ == '__main__':
    main()