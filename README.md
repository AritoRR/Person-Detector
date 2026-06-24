# Person-Detector
Детектор людей на основе архитектуры YOLO. Обучение с нуля на датасете COCO (класс "person").

## Быстрый старт
1. Клонировать репозиторий
2. Установить зависимости: `pip install -r requirements.txt`
3. Скачать COCO: `python scripts/download_coco.py`
4. Запустить обучение: `python src/train.py --data configs/coco_person.yaml --weights '' --epochs 100`

## Результаты
pass

## Структура проекта
- `src/` — код обучения
- `configs/` — конфиги датасетов
- `notebooks/` — анализ