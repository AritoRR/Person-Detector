import subprocess
import sys
import os

from script_path import TRAIN_PATH, CONFIG_PATH, MODEL_PATH, RUNS_PATH, jpath


def main():
    resume = False
    if len(sys.argv) > 1 and sys.argv[1] == '--resume':
        resume = True

    if not os.path.exists(TRAIN_PATH):
        print(f"ERROR: {TRAIN_PATH} dont exist.")
        sys.exit(1)

    if resume:
        last_pt = '../runs/train/exp_person/weights/last.pt'
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
            '--data', CONFIG_PATH,
            '--cfg', MODEL_PATH,
            '--weights', '',
            '--img', '640',
            '--batch', '16',
            '--epochs', '100',
            '--project', jpath(RUNS_PATH, 'train'),
            '--name', 'exp_person',
            '--exist-ok'
        ]

    print("=" * 60)
    print("start of YOLOv5 training:")
    print(' '.join(cmd))
    print("=" * 60)
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
