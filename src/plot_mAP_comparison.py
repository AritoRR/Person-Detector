import pandas as pd
import matplotlib.pyplot as plt
import os

# Пути к файлам (убедись, что они лежат в папке datasets/csv или где у тебя)
csv_files = {
    'Базовый (Baseline)': 'базовыый.csv',
    'Уменьшенный lr0': 'только lr0.csv',
    'Аугментация (deg+flip+hsv)': 'все аугментации тут.csv',
    'Только поворот (degrees)': 'только поворот.csv',
    'Дообучение на Military': 'дообученый на милитари.csv'
}

# Папка с CSV (если файлы в другой папке — поменяй)
CSV_DIR = '.'  # <-- если файлы лежат в другой папке, укажи здесь

# Цвета и стили для разных экспериментов
files_map = {
    'базовыый.csv': 'Базовый (Baseline)',
    'только lr0.csv': 'Уменьшенный lr0',
    'все аугментации тут.csv': 'Аугментация (deg+flip+hsv)',
    'только поворот.csv': 'Только поворот (degrees)',
    'дообученый на милитари.csv': 'Дообучение на Military'
}

colors = {
    'Базовый (Baseline)': 'blue',
    'Уменьшенный lr0': 'green',
    'Аугментация (deg+flip+hsv)': 'orange',
    'Только поворот (degrees)': 'purple',
    'Дообучение на Military': 'red'
}

linestyles = {
    'Базовый (Baseline)': '-',
    'Уменьшенный lr0': '--',
    'Аугментация (deg+flip+hsv)': '-.',
    'Только поворот (degrees)': ':',
    'Дообучение на Military': '-'
}


def find_column(df, possible_names):
    """Ищет колонку по списку возможных названий (без учёта пробелов)"""
    for col in df.columns:
        col_clean = col.strip().replace(' ', '')
        for name in possible_names:
            if col_clean == name.replace(' ', ''):
                return col
    return None


plt.figure(figsize=(14, 8))
found_any = False

for filename, label in files_map.items():
    # Пробуем найти файл
    filepath = None
    for root, dirs, files in os.walk('.'):
        if filename in files:
            filepath = os.path.join(root, filename)
            break

    if filepath is None:
        print(f"❌ Файл {filename} не найден. Пропускаю...")
        continue

    print(f"✅ Найден файл: {filepath}")

    try:
        df = pd.read_csv(filepath)

        # Находим колонку с эпохами
        epoch_col = find_column(df, ['epoch', 'Epoch', 'epochs'])
        if epoch_col is None:
            print(f"⚠️ В {filename} нет колонки с эпохами. Доступные колонки: {df.columns.tolist()}")
            continue

        # Находим колонку с mAP
        mAP_col = find_column(df, ['metrics/mAP_0.5', 'mAP_0.5', 'mAP@0.5'])
        if mAP_col is None:
            print(f"⚠️ В {filename} нет колонки с mAP. Доступные колонки: {df.columns.tolist()}")
            continue

        print(f"   Использую: эпохи = '{epoch_col}', mAP = '{mAP_col}'")

        # Берём ТОЛЬКО ПЕРВЫЕ 20 ЭПОХ
        epochs = df[epoch_col][:20]
        mAP = df[mAP_col][:20]

        plt.plot(
            epochs,
            mAP,
            label=label,
            color=colors.get(label, 'black'),
            linestyle=linestyles.get(label, '-'),
            linewidth=2.5,
            marker='o',
            markersize=5,
        )
        found_any = True

    except Exception as e:
        print(f"⚠️ Ошибка при чтении {filename}: {e}")

if not found_any:
    print("❌ Нет ни одного файла для построения графика!")
    exit()

# Настройки графика
plt.xlabel('Эпоха', fontsize=14, fontweight='bold')
plt.ylabel('mAP@0.5', fontsize=14, fontweight='bold')
plt.title('Сравнение динамики обучения mAP@0.5 (первые 20 эпох)', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.ylim(0.3, 1.0)  # Ограничиваем снизу, чтобы видеть разницу
plt.xlim(0, 20)

plt.tight_layout()
plt.savefig('./mAP_comparison_by_epoch.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ График сохранен в runs/mAP_comparison_by_epoch.png")