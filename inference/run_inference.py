"""
Call CogVideo's cli_demo.py once for every (prompt, image) pair.
Keeps ordering deterministic and makes it easy to swap sort logic.
"""

import argparse
import subprocess
from pathlib import Path

try:
    # natural-sort is handy if filenames contain numbers (frame1, frame2, frame10 …)
    from natsort import natsorted                         # pip install natsort
except ImportError:
    natsorted = sorted

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--script_dir",     required=True)     # directory that has cli_demo.py
    p.add_argument("--prompt_file",    required=True)
    p.add_argument("--frame_dir",      required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--model_path",     required=True)
    p.add_argument("--lora_path",      required=True)
    p.add_argument("--num_frames",     type=int, default=49)
    p.add_argument("--width",          type=int, default=720)
    p.add_argument("--height",         type=int, default=480)
    return p.parse_args()

def main() -> None:
    args = parse_args()

    script      = Path(args.script_dir, "cli_demo.py")
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = [ln.strip() for ln in Path(args.prompt_file).read_text().splitlines() if ln.strip()]
    images  = natsorted([p for p in Path(args.frame_dir).glob("*") if p.is_file()], key=lambda p: p.name)

    for idx, (prompt, img) in enumerate(zip(prompts, images), start=0):
        print("Prompt: ", prompt)
        print("Image: ", img)
        out_file = output_dir / f"{idx:04d}.mp4"

        cmd = [
            "python", str(script),
            "--prompt",             prompt,
            "--image_or_video_path", str(img),
            "--model_path",         args.model_path,
            "--lora_path",          args.lora_path,
            "--output_path",        str(out_file),
            "--num_frames",         str(args.num_frames),
            "--width",              str(args.width),
            "--height",             str(args.height),
            "--generate_type",      "i2v",
        ]
        subprocess.run(cmd, check=True)

    print(f"All videos saved in {output_dir}")

if __name__ == "__main__":
    main()
