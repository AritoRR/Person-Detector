import subprocess
import sys
import os

from src.script_path import VAL_PATH, WEIGHTS_PATH


def main():
    if not os.path.exists(VAL_PATH):
        print(f"ERROR: {VAL_PATH} dont exist.")
        sys.exit(1)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: {WEIGHTS_PATH} dont exist.")
        sys.exit(1)

    cmd = [
        sys.executable,
        VAL_PATH,
        '--data', 'configs/coco_person.yaml',
        '--weights', WEIGHTS_PATH,
        '--img', '640',
        '--batch', '16',
        '--project', 'runs/val',
        '--name', 'exp_person'
    ]

    print("=" * 60)
    print("Model validation starts")
    print(f"WEIGHTS: {WEIGHTS_PATH}")
    print("=" * 60)

    subprocess.run(cmd)


if __name__ == '__main__':
    main()
