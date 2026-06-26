import subprocess
import sys
import os
import argparse

from script_path import VAL_PATH, WEIGHTS_PATH, jpath, RUNS_PATH, CONFIGS_PATH


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='exp_person', help='Имя эксперимента')
    parser.add_argument('--weights', type=str, default=WEIGHTS_PATH, help='Путь к весам')
    parser.add_argument('--data', type=str, default='coco', help='Имя папки с датасетом')

    args = parser.parse_args()

    if not os.path.exists(VAL_PATH):
        print(f"ERROR: {VAL_PATH} dont exist.")
        sys.exit(1)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: {WEIGHTS_PATH} dont exist.")
        sys.exit(1)

    cmd = [
        sys.executable,
        VAL_PATH,
        '--data', jpath(CONFIGS_PATH, args.data),
        '--weights', args.weights,
        '--img', '640',
        '--batch', '16',
        '--project', jpath(RUNS_PATH, 'test'),
        '--name', args.name,
        '--task', 'test'
    ]

    print("=" * 60)
    print("Model test")
    print(' '.join(cmd))
    print("=" * 60)

    subprocess.run(cmd)


if __name__ == '__main__':
    main()
