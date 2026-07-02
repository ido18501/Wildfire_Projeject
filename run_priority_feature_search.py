import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

REPO = Path.cwd()
DATA_ROOT = str(Path.home() / "sim2real_fire_data/simulation_data")
SCENES = "0014_00426,0005_00725,0004_01191"
EPOCHS = "100"
MAX_FIRES = "480"
GPUS = "0"
SEED = "42"

GROUPS = [
    "physics_plus",
    "baseline",
    "all_available",

    "no_wind",
    "no_dryness",
    "no_weather",
    "no_fuel_type",
    "no_canopy",
    "no_vegetation",
    "no_topography",

    "weather_only",
    "topography_only",
    "fuel_veg_only",
    "omitted_only",

    "topo_weather",
    "weather_fuel",
    "topo_fuel",
    "weather_omitted",
    "fuel_omitted",
    "topo_omitted",
    "topo_weather_omitted",
    "topo_fuel_omitted",
    "weather_fuel_omitted",

    "plus_fbfm40",
    "plus_canopy_base_height",
    "plus_fuel_disturbance",
    "plus_fuel_vegetation_cover",
    "plus_fuel_vegetation_height",
    "plus_fuel_vegetation_type",
    "plus_existing_vegetation_height",
    "plus_succession_classes",
    "plus_vegetation_condition_class",
    "plus_vegetation_departure",
]

def run_cmd(cmd, log_path=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    with subprocess.Popen(
        cmd,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        with open(log_path, "w") if log_path else open(os.devnull, "w") as f:
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
        return proc.wait()

def parse_log(log_path):
    text = Path(log_path).read_text(errors="ignore")
    avg_corr = re.findall(r"Average correlation:\s*([0-9.\-]+)", text)
    avg_rmse = re.findall(r"Average relative RMSE:\s*([0-9.\-]+)", text)
    scenes = re.findall(r"\s*(\d{4}_\d{5}): corr=([0-9.\-]+), rel_rmse=([0-9.\-]+)", text)
    return {
        "avg_corr": avg_corr[-1] if avg_corr else "",
        "avg_rel_rmse": avg_rmse[-1] if avg_rmse else "",
        "scene_results": "; ".join(f"{s}:corr={c},rmse={r}" for s, c, r in scenes),
    }

run_root = REPO / f"priority_feature_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logs_dir = run_root / "logs"
logs_dir.mkdir(parents=True)

master = run_root / "MASTER_LOG.txt"

with master.open("w") as m:
    m.write(f"RUN_ROOT={run_root}\n")
    m.write(f"Started at {datetime.now()}\n")
    m.write(f"Scenes={SCENES}\n")
    m.write(f"Epochs={EPOCHS}\n")
    m.write(f"Max fires per scene={MAX_FIRES}\n")
    m.write(f"Seed={SEED}\n")
    m.write(f"Groups={len(GROUPS)}\n")
    m.write("\n".join(GROUPS) + "\n")

rows = []

for i, group in enumerate(GROUPS, 1):
    print("=" * 70, flush=True)
    print(f"[{i}/{len(GROUPS)}] Running group: {group} at {datetime.now()}", flush=True)
    print("=" * 70, flush=True)

    with master.open("a") as m:
        m.write(f"\n[{i}/{len(GROUPS)}] Running {group} at {datetime.now()}\n")

    log_file = logs_dir / f"{group}.log"
    out_dir = run_root / f"experiments_{group}"

    cmd = [
        sys.executable,
        "experiments/wildfire/per_scene.py",
        "--mode", "per_scene",
        "--data_root", DATA_ROOT,
        "--scenes", SCENES,
        "--gpus", GPUS,
        "--epochs", EPOCHS,
        "--feature_group", group,
        "--max_fires_per_scene", MAX_FIRES,
        "--output", str(out_dir),
        "--seed", SEED,
        "--sequential",
    ]

    status = run_cmd(cmd, log_file)
    parsed = parse_log(log_file)

    row = {
        "group": group,
        "status": status,
        **parsed,
    }
    rows.append(row)

    with master.open("a") as m:
        m.write(f"Finished {group} status={status} at {datetime.now()}\n")
        m.write(f"avg_corr={row['avg_corr']} avg_rel_rmse={row['avg_rel_rmse']}\n")

    time.sleep(10)

summary = run_root / "SUMMARY_TABLE.csv"
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["group", "status", "avg_corr", "avg_rel_rmse", "scene_results"])
    writer.writeheader()
    writer.writerows(rows)

ranked = run_root / "SUMMARY_RANKED_BY_RMSE.csv"
ranked_rows = [r for r in rows if r["avg_rel_rmse"]]
ranked_rows.sort(key=lambda r: float(r["avg_rel_rmse"]))
with ranked.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["group", "status", "avg_corr", "avg_rel_rmse", "scene_results"])
    writer.writeheader()
    writer.writerows(ranked_rows)

with master.open("a") as m:
    m.write(f"\nFinished all at {datetime.now()}\n")
    m.write(f"Summary: {summary}\n")
    m.write(f"Ranked: {ranked}\n")

print(f"SUMMARY_TABLE={summary}")
print(f"SUMMARY_RANKED={ranked}")
