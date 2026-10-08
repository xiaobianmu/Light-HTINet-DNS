import subprocess
import sys
import os
from pathlib import Path

seeds = [1, 42, 2024]

out_root = Path("./outputs")
out_root.mkdir(exist_ok=True)

for seed in seeds:
    print(f"\n{'='*60}")
    print(f"Running experiment with SEED = {seed}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)

    seed_out_dir = out_root / f"seed_{seed}"
    seed_out_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        "train.py",
        "--seed", str(seed),
        "--output_dir", str(seed_out_dir)
    ]

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"Seed {seed} failed, return code: {result.returncode}")
    else:
        print(f"Seed {seed} finished")

print("\nAll seed experiments finished!")