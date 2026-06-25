import os
import subprocess
import sys

from script_path import WEIGHTS_PATH, RUNS_PATH, DETECT_PATH
from src.script_path import jpath


def main():
    if len(sys.argv) < 3:
        print("ERROR: enter --image /path/to/image.jpg")
        sys.exit(1)

    if not os.path.exists(DETECT_PATH):
        print(f"ERROR: {DETECT_PATH} dont exist.")
        sys.exit(1)

    cmd = [
        sys.executable,
        DETECT_PATH,
        '--weights', WEIGHTS_PATH,
        '--source', sys.argv[2],
        '--img', '640',
        '--conf', '0.25',
        '--project', jpath(RUNS_PATH, 'detection'),
        '--name', 'exp_person',
        '--exist-ok'
    ]

    print("=" * 60)
    print("Person Detection:")
    print(' '.join(cmd))
    print("=" * 60)
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
