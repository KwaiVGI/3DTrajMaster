import argparse
from pathlib import Path
from tqdm import tqdm
from hmr4d.utils.pylogger import Log
import subprocess
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument("-d", "--output_root", type=str, default=None)
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    args = parser.parse_args()

    output_root = args.output_root

    sub_folders = os.listdir(args.folder)
    mp4_paths = []
    for sub_folder in sub_folders:
        files = os.listdir(os.path.join(args.folder, sub_folder))
        for file in files:
            if file.endswith('.mp4'):
                mp4_path = os.path.join(args.folder, sub_folder, file)
                mp4_paths.append(mp4_path)

    # Run demo.py for each .mp4 file
    Log.info(f"Found {len(mp4_paths)} .mp4 files in {args.folder}")
    for mp4_path in tqdm(mp4_paths):
        try:
            command = ["python", "tools/demo/demo.py", "--video", str(mp4_path)]
            if output_root is not None:
                command += ["--output_root", output_root]
            if args.static_cam:
                command += ["-s"]
            Log.info(f"Running: {' '.join(command)}")
            subprocess.run(command, env=dict(os.environ), check=True)
        except:
            continue