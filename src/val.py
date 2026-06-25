import subprocess
import sys
import os

from script_path import VAL_PATH, WEIGHTS_PATH, CONFIG_PATH, jpath, RUNS_PATH


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
        '--data', CONFIG_PATH,
        '--weights', WEIGHTS_PATH,
        '--img', '640',
        '--batch', '16',
        '--project', jpath(RUNS_PATH, '/val'),
        '--name', 'exp_person'
    ]

    print("=" * 60)
    print("Model validation starts")
    print(f"WEIGHTS: {WEIGHTS_PATH}")
    print("=" * 60)

    subprocess.run(cmd)


if __name__ == '__main__':
    main()
