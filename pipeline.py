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
from src.alerting import send_alerts


def run_pipeline(generate: bool):
    """Run the data quality pipeline."""
    total_steps = 4 if generate else 3
    step = 1

    if generate:
        print(f"\n>>> STEP {step}/{total_steps}: Generating sample data...")
        generate_data()
        step += 1
    else:
        print("\n>>> Skipping data generation, using existing data in data/")

    print(f"\n>>> STEP {step}/{total_steps}: Profiling data...")
    profile_data()
    step += 1

    print(f"\n>>> STEP {step}/{total_steps}: Detecting anomalies...")
    all_anomalies = detect_anomalies() or {}
    step += 1

    print(f"\n>>> STEP {step}/{total_steps}: Explaining anomalies...")
    explanations = explain_anomalies() or []

    # Send alerts for detected anomalies
    flat_anomalies = [a for lst in all_anomalies.values() for a in lst]
    if flat_anomalies:
        print("\n>>> Sending alerts...")
        send_alerts(flat_anomalies, explanations)

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
