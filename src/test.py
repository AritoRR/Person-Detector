import subprocess
import sys
import os

from script_path import VAL_PATH, WEIGHTS_PATH, jpath, RUNS_PATH, CONFIGS_PATH


def main():
    data = 'coco'
    if len(sys.argv) > 1:
        data = sys.argv[1]

    if not os.path.exists(VAL_PATH):
        print(f"ERROR: {VAL_PATH} dont exist.")
        sys.exit(1)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: {WEIGHTS_PATH} dont exist.")
        sys.exit(1)

    cmd = [
        sys.executable,
        VAL_PATH,
        '--data', jpath(CONFIGS_PATH, data),
        '--weights', WEIGHTS_PATH,
        '--img', '640',
        '--batch', '16',
        '--project', jpath(RUNS_PATH, 'test'),
        '--name', 'exp_person',
        '--task', 'test'
    ]

    print("=" * 60)
    print("Model test")
    print(' '.join(cmd))
    print("=" * 60)

    subprocess.run(cmd)


if __name__ == '__main__':
    main()
