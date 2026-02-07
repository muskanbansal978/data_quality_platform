"""
Data Quality Platform CLI Pipeline.

Usage:
    python pipeline.py --all            Run full pipeline (generate data + profile + detect + explain)
    python pipeline.py --no-generate    Run pipeline on existing data (profile + detect + explain)
"""

import argparse
import sys

from src.data_generator import main as generate_data
from src.data_profiler import main as profile_data
from src.anomaly_detector import main as detect_anomalies
from src.llm_explainer import main as explain_anomalies


def run_pipeline(generate: bool):
    """Run the data quality pipeline."""
    if generate:
        print("\n>>> STEP 1/4: Generating sample data...")
        generate_data()
        step_offset = 0
    else:
        step_offset = -1
        print("\n>>> Skipping data generation, using existing data in data/")

    print(f"\n>>> STEP {2 + step_offset}/{3 + (1 if generate else 0)}: Profiling data...")
    profile_data()

    print(f"\n>>> STEP {3 + step_offset}/{3 + (1 if generate else 0)}: Detecting anomalies...")
    detect_anomalies()

    print(f"\n>>> STEP {4 + step_offset}/{3 + (1 if generate else 0)}: Explaining anomalies...")
    explain_anomalies()

    print("\n" + "=" * 50)
    print("   Pipeline complete!")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Data Quality Platform Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run full pipeline including data generation")
    group.add_argument("--no-generate", action="store_true", help="Run pipeline on existing data (skip generation)")

    args = parser.parse_args()
    run_pipeline(generate=args.all)


if __name__ == "__main__":
    main()
