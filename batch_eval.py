import subprocess
import sys
import os
from pathlib import Path
import re
import csv
import statistics
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.resolve()

outputs_dir = SCRIPT_DIR / "outputs"
if not outputs_dir.exists():
    print("No outputs/ directory found.")
    exit()

# Find all seed_* folders under outputs/
seed_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])

print(f"Found {len(seed_dirs)} seed directories:")
for d in seed_dirs:
    print(f"  - {d}")

all_results = []

for seed_dir in seed_dirs:
    seed = int(seed_dir.name.split("_")[1])
    model_dir = seed_dir / f"s_seed{seed}"

    if not model_dir.exists() or not list(model_dir.glob("epoch*.pt")):
        print(f"\nSeed {seed}: model directory not found at {model_dir}")
        continue

    print(f"\n{'='*70}")
    print(f"  Seed {seed}: {model_dir}")
    print(f"{'='*70}")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "test.py"),
        "--seed", str(seed),
        "--model_dir", str(model_dir)
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR),
                            capture_output=True, text=True,
                            encoding='utf-8', errors='replace', env=env)

    if result.returncode != 0:
        print(f"Seed {seed} failed:")
        print(result.stderr[-500:])
        continue

    output = result.stdout

    # Only parse the part AFTER "Best Epoch" to avoid picking up validation metrics
    parts = output.split("Best Epoch")
    if len(parts) < 2:
        print(f"Seed {seed}: could not find Best Epoch in output")
        continue

    test_output = parts[1]

    # Parse best epoch
    m = re.search(r"=\s*(\d+)", test_output)
    best_epoch = int(m.group(1)) if m else None

    # Parse only test metrics (lines after Best Epoch)
    hr = {}
    ndcg = {}
    for line in test_output.splitlines():
        hm = re.search(r"HR@(\d+)\s*:\s*([\d.]+)", line)
        nm = re.search(r"NDCG@(\d+)\s*:\s*([\d.]+)", line)
        if hm:
            hr[int(hm.group(1))] = float(hm.group(2))
        if nm:
            ndcg[int(nm.group(1))] = float(nm.group(2))

    print(f" Best epoch={best_epoch}, HR@10={hr.get(10, 'N/A')}, NDCG@10={ndcg.get(10, 'N/A')}")

    all_results.append({
        'seed': seed,
        'best_epoch': best_epoch,
        'HR@1': hr.get(1),
        'HR@3': hr.get(3),
        'HR@5': hr.get(5),
        'HR@10': hr.get(10),
        'NDCG@1': ndcg.get(1),
        'NDCG@3': ndcg.get(3),
        'NDCG@5': ndcg.get(5),
        'NDCG@10': ndcg.get(10),
    })

# Save CSV
if all_results:
    output_csv = SCRIPT_DIR / "batch_eval_results.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {output_csv}")

    # Summary
    print(f"\n{'='*70}")
    print("  Mean ± Std across 3 seeds")
    print(f"{'='*70}")

    for metric in ['HR@10', 'NDCG@10']:
        vals = [r[metric] for r in all_results if r[metric] is not None]
        if len(vals) >= 2:
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            print(f"  {metric:8s} = {mean:.4f} ± {std:.4f}")
