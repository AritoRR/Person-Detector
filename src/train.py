import subprocess
import sys
import os
import argparse

from script_path import TRAIN_PATH, MODEL_PATH, RUNS_PATH, jpath, WEIGHTS_PATH, HYP_SLOW_PATH, CONFIGS_PATH


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', action='store_true', help='Продолжить обучение с last.pt')

    parser.add_argument('--name', type=str, default='exp_new', help='Имя эксперимента')
    parser.add_argument('--weights', type=str, default=WEIGHTS_PATH, help='Путь к весам')
    parser.add_argument('--hyp', type=str, default='hyp.scratch-low.yaml', help='Путь к файлу гиперпараметров')
    parser.add_argument('--data', type=str, default='coco_person.yaml', help='Имя файла конфига')
    parser.add_argument('--epochs', type=str, default='100', help='Число эпох')

    args = parser.parse_args()
    if not os.path.exists(TRAIN_PATH):
        print(f"ERROR: {TRAIN_PATH} dont exist.")
        sys.exit(1)

    if args.resume:
        last_pt = f'../runs/train/{args.name}/weights/last.pt'
        if not os.path.exists(last_pt):
            print(f"ERROR: {last_pt} not found. Can't resume.")
            sys.exit(1)

        cmd = [
            sys.executable,
            TRAIN_PATH,
            '--resume', last_pt
        ]
    else:
        cmd = [
            sys.executable,
            TRAIN_PATH,
            '--data', jpath(CONFIGS_PATH, args.data),
            '--cfg', MODEL_PATH,
            '--weights', args.weights,
            '--hyp', jpath(CONFIGS_PATH, 'hyps', args.hyp),
            '--img', '640',
            '--batch', '16',
            '--epochs', args.epochs,
            '--project', jpath(RUNS_PATH, 'train'),
            '--name', args.name,
            '--exist-ok'
        ]

    print("=" * 60)
    print("start of YOLOv5 training:")
    print(' '.join(cmd))
    print("=" * 60)
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
