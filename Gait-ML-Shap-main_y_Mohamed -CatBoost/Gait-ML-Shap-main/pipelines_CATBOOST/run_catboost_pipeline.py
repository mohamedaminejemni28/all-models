from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pipeline_common import PROJECT_ROOT, resolve_config_path

PHASES = [
    ("1", "phase1_reviewfeatures.py"),
    ("2", "phase2_removecorrelated_features.py"),
    ("3", "phase3_split_datasets.py"),
    ("4.1", "phase4_1_SFS_CATBOOST.py"),
    ("4.2", "phase4_2_model_combinations_catboost.py"),
    ("4.3", "phase4_3_scores_catboost.py"),
    ("4.4", "phase4_4_SHAP_catboost.py"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CatBoost gait ML pipeline.")
    parser.add_argument("--config", default=None, help="Config name/path, for example config_Flatfoot_2024.yaml")
    parser.add_argument("--start", default="1", choices=[phase for phase, _ in PHASES])
    parser.add_argument("--stop", default="4.4", choices=[phase for phase, _ in PHASES])
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config or "config.yaml")
    phase_names = [phase for phase, _ in PHASES]
    start_index = phase_names.index(args.start)
    stop_index = phase_names.index(args.stop)

    if start_index > stop_index:
        raise ValueError("--start must be before --stop")

    pipeline_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["CATBOOST_CONFIG"] = str(config_path)

    print(f"Using config: {config_path}", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)

    for phase, script_name in PHASES[start_index : stop_index + 1]:
        script_path = pipeline_dir / script_name
        print(f"\n===== Running CatBoost phase {phase}: {script_name} =====", flush=True)
        command = [sys.executable, str(script_path), "--config", str(config_path)]
        result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print("\nCatBoost pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
