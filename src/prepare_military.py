# src/prepare_military_dataset.py
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Импортируем jpath из твоего script_path
from script_path import jpath, MILITARY_PATH


def mask_to_bbox(mask_path, image_path):
    """
    Преобразует маску в bbox в формате YOLO.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    pts = np.argwhere(mask > 0)
    if len(pts) == 0:
        return None

    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w, _ = img.shape

    x_center = (x_min + x_max) / 2 / w
    y_center = (y_min + y_max) / 2 / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

def prepare_dataset(img_dir, gt_dir, output_dir, test_size=0.2):
    """
    Основная функция подготовки датасета.
    """
    # Создаем папки для YOLO
    images_train = Path(output_dir) / 'images' / 'train'
    images_val = Path(output_dir) / 'images' / 'val'
    labels_train = Path(output_dir) / 'labels' / 'train'
    labels_val = Path(output_dir) / 'labels' / 'val'

    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    # Ищем файлы
    img_paths = sorted(Path(img_dir).glob('*.*'))
    gt_paths = sorted(Path(gt_dir).glob('*.*'))

    print(f"Найдено изображений: {len(img_paths)}")
    print(f"Найдено масок: {len(gt_paths)}")

    # Сопоставляем по имени файла
    paired = []
    for img_path in img_paths:
        stem = img_path.stem
        mask_candidates = [p for p in gt_paths if p.stem == stem]
        if mask_candidates:
            paired.append((img_path, mask_candidates[0]))

    if not paired:
        print("ОШИБКА: не найдено ни одной пары изображение-маска!")
        print(f"Проверьте папки: {img_dir} и {gt_dir}")
        return

    print(f"Найдено {len(paired)} пар изображение-маска")

    # Разбиваем на train/val
    train_pair, val_pair = train_test_split(paired, test_size=test_size, random_state=42)

    # Копируем train
    for img_path, mask_path in train_pair:
        shutil.copy2(img_path, images_train / img_path.name)
        label = mask_to_bbox(mask_path, img_path)
        if label:
            (labels_train / f"{img_path.stem}.txt").write_text(label)
        else:
            print(f"⚠️ Не удалось создать лейбл для {img_path.name}")

    # Копируем val
    for img_path, mask_path in val_pair:
        shutil.copy2(img_path, images_val / img_path.name)
        label = mask_to_bbox(mask_path, img_path)
        if label:
            (labels_val / f"{img_path.stem}.txt").write_text(label)
        else:
            print(f"⚠️ Не удалось создать лейбл для {img_path.name}")

    print(f"Train: {len(train_pair)} пар")
    print(f"Val:   {len(val_pair)} пар")


if __name__ == "__main__":
    # Используем jpath из script_path
    # Предполагаем, что MILITARY_PATH указывает на папку с img и gt
    # Например, MILITARY_PATH = "../datasets/military"
    prepare_dataset(
        jpath(MILITARY_PATH, 'img'),
        jpath(MILITARY_PATH, 'gt'),
        jpath(MILITARY_PATH, '..', 'military_yolo')
    )