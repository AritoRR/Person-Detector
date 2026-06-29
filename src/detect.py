import os
import subprocess
import sys
import argparse

from script_path import WEIGHTS_PATH, RUNS_PATH, DETECT_PATH
from src.script_path import jpath


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='exp_person', help='Имя эксперимента')
    parser.add_argument('--image', type=str, default='exp_person', help='Путь к картинке')

    args = parser.parse_args()
    if not os.path.exists(DETECT_PATH):
        print(f"ERROR: {DETECT_PATH} dont exist.")
        sys.exit(1)

    cmd = [
        sys.executable,
        DETECT_PATH,
        '--weights', WEIGHTS_PATH,
        '--source', args.image,
        '--img', '640',
        '--conf', '0.25',
        '--project', jpath(RUNS_PATH, 'detection'),
        '--name', args.name,
        '--exist-ok'
    ]

    print("=" * 60)
    print("Person Detection:")
    print(' '.join(cmd))
    print("=" * 60)
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
