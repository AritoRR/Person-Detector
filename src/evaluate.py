import os
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(results_csv_path='runs/train/exp_person/results.csv'):
    if not os.path.exists(results_csv_path):
        print(f"ERROR: {results_csv_path} not found")
        return None

    df = pd.read_csv(results_csv_path)
    print(f"Loaded {len(df)} epochs")
    return df


def plot_metrics(df, output_dir='runs/eval'):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(df['epoch'], df['box_loss'], label='Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].grid(True)

    axes[0, 1].plot(df['epoch'], df['obj_loss'], label='Object Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Object Loss')
    axes[0, 1].grid(True)

    axes[1, 0].plot(df['epoch'], df['mAP@0.5'], label='mAP@0.5', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].set_title('mAP@0.5')
    axes[1, 0].grid(True)

    axes[1, 1].plot(df['epoch'], df['mAP@0.5:0.95'], label='mAP@0.5:0.95', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mAP')
    axes[1, 1].set_title('mAP@0.5:0.95')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    print(f"Графики сохранены в {os.path.join(output_dir, 'training_metrics.png')}")
    plt.show()


def main():
    print("=" * 60)
    print("СБОР МЕТРИК ОБУЧЕНИЯ")
    print("=" * 60)

    df = load_metrics()

    if df is not None and len(df) > 0:
        print("\nПоследние метрики:")
        print(df.tail())

        # Строим графики
        plot_metrics(df)

        print("\n--- ФИНАЛЬНЫЕ МЕТРИКИ ---")
        last_row = df.iloc[-1]
        print(f"mAP@0.5:     {last_row['mAP@0.5']:.4f}")
        print(f"mAP@0.5:0.95: {last_row['mAP@0.5:0.95']:.4f}")
        print(f"Box Loss:    {last_row['box_loss']:.4f}")
        print(f"Object Loss: {last_row['obj_loss']:.4f}")
    else:
        print("Файл с метриками не найден или пуст. Запустите обучение или валидацию.")


if __name__ == '__main__':
    main()
