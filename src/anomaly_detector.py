"""
Anomaly detection for data quality monitoring.

Detects anomalies in data profiles using statistical methods (z-score, IQR).
Identifies volume drops, null spikes, value range violations, and distribution shifts.

Run with: python -m src.anomaly_detector
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Anomaly types
VOLUME_DROP = "volume_drop"
VOLUME_SPIKE = "volume_spike"
NULL_RATE_INCREASE = "null_rate_increase"
VALUE_OUT_OF_RANGE = "value_out_of_range"
NEGATIVE_VALUES = "negative_values"
DISTRIBUTION_SHIFT = "distribution_shift"
MEAN_SHIFT = "mean_shift"

# Severity levels
LOW = "low"
MEDIUM = "medium"
HIGH = "high"
CRITICAL = "critical"

# Default thresholds
DEFAULT_VOLUME_Z_THRESHOLD = 2.5
DEFAULT_NULL_Z_THRESHOLD = 2.5
DEFAULT_VALUE_Z_THRESHOLD = 3.0
DEFAULT_MIN_HISTORY_DAYS = 5


def create_anomaly(
    anomaly_type: str,
    severity: str,
    date: datetime,
    column: Optional[str],
    message: str,
    expected_value: float,
    actual_value: float,
    z_score: float,
) -> dict[str, Any]:
    """Create an anomaly dictionary."""
    return {
        "type": anomaly_type,
        "severity": severity,
        "date": date,
        "column": column,
        "message": message,
        "expected": expected_value,
        "actual": actual_value,
        "z_score": z_score,
    }


def create_daily_profile(
    date: datetime,
    row_count: int,
    null_counts: Optional[dict[str, int]] = None,
    null_rates: Optional[dict[str, float]] = None,
    means: Optional[dict[str, float]] = None,
    mins: Optional[dict[str, float]] = None,
    maxs: Optional[dict[str, float]] = None,
    stds: Optional[dict[str, float]] = None,
    category_distributions: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, Any]:
    """Create a daily profile dictionary."""
    return {
        "date": date,
        "row_count": row_count,
        "null_counts": null_counts or {},
        "null_rates": null_rates or {},
        "means": means or {},
        "mins": mins or {},
        "maxs": maxs or {},
        "stds": stds or {},
        "category_distributions": category_distributions or {},
    }


def calculate_z_score(
    value: float,
    historical_values: list[float],
) -> tuple[float, float, float]:
    """
    Calculate z-score for a value given historical data.

    Returns:
        Tuple of (z_score, mean, std)
    """
    if len(historical_values) < 2:
        return 0.0, value, 0.0

    mean = np.mean(historical_values)
    std = np.std(historical_values)

    if std == 0:
        return 0.0, mean, std

    z_score = (value - mean) / std
    return z_score, mean, std


def get_severity(z_score: float) -> str:
    """Determine severity based on z-score magnitude."""
    abs_z = abs(z_score)

    if abs_z >= 4.0:
        return CRITICAL
    elif abs_z >= 3.0:
        return HIGH
    elif abs_z >= 2.0:
        return MEDIUM
    else:
        return LOW


def compute_daily_profiles(
    df: pd.DataFrame,
    date_column: str,
    numeric_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Compute daily profile statistics from a DataFrame.

    Args:
        df: DataFrame with data
        date_column: Name of the date column
        numeric_columns: Columns to compute numeric stats for
        categorical_columns: Columns to compute distribution stats for

    Returns:
        List of daily profile dictionaries, one per day
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df[date_column]).dt.date

    # Auto-detect column types if not specified
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_columns = [
            c for c in categorical_columns
            if c != date_column and not c.endswith("_id") and c != "_date"
        ]

    profiles = []

    for date, group in df.groupby("_date"):
        null_counts = {}
        null_rates = {}
        means = {}
        mins = {}
        maxs = {}
        stds = {}
        category_distributions = {}

        # Compute null rates for all columns
        for col in df.columns:
            if col in ["_date", date_column]:
                continue
            null_count = group[col].isna().sum()
            null_counts[col] = null_count
            null_rates[col] = null_count / len(group) * 100 if len(group) > 0 else 0

        # Compute numeric stats
        for col in numeric_columns:
            if col in group.columns:
                non_null = group[col].dropna()
                if len(non_null) > 0:
                    means[col] = float(non_null.mean())
                    mins[col] = float(non_null.min())
                    maxs[col] = float(non_null.max())
                    stds[col] = float(non_null.std())

        # Compute category distributions
        for col in categorical_columns:
            if col in group.columns:
                value_counts = group[col].value_counts(normalize=True)
                category_distributions[col] = value_counts.to_dict()

        profile = create_daily_profile(
            date=datetime.combine(date, datetime.min.time()),
            row_count=len(group),
            null_counts=null_counts,
            null_rates=null_rates,
            means=means,
            mins=mins,
            maxs=maxs,
            stds=stds,
            category_distributions=category_distributions,
        )
        profiles.append(profile)

    # Sort by date
    profiles.sort(key=lambda p: p["date"])

    return profiles


def detect_volume_anomalies(
    profiles: list[dict[str, Any]],
    volume_z_threshold: float = DEFAULT_VOLUME_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect volume (row count) anomalies."""
    anomalies = []

    for i, profile in enumerate(profiles):
        if i < min_history_days:
            continue

        historical = [p["row_count"] for p in profiles[:i]]
        z_score, mean, std = calculate_z_score(profile["row_count"], historical)

        if z_score < -volume_z_threshold:
            anomalies.append(create_anomaly(
                anomaly_type=VOLUME_DROP,
                severity=get_severity(z_score),
                date=profile["date"],
                column=None,
                message=f"Row count dropped to {profile['row_count']:,} (expected ~{mean:,.0f} +/- {std:,.0f})",
                expected_value=mean,
                actual_value=profile["row_count"],
                z_score=round(z_score, 2),
            ))
        elif z_score > volume_z_threshold:
            anomalies.append(create_anomaly(
                anomaly_type=VOLUME_SPIKE,
                severity=get_severity(z_score),
                date=profile["date"],
                column=None,
                message=f"Row count spiked to {profile['row_count']:,} (expected ~{mean:,.0f} +/- {std:,.0f})",
                expected_value=mean,
                actual_value=profile["row_count"],
                z_score=round(z_score, 2),
            ))

    return anomalies


def detect_null_anomalies(
    profiles: list[dict[str, Any]],
    null_z_threshold: float = DEFAULT_NULL_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect null rate spike anomalies."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["null_rates"].keys())

    for column in all_columns:
        for i, profile in enumerate(profiles):
            if i < min_history_days:
                continue

            if column not in profile["null_rates"]:
                continue

            historical = [
                p["null_rates"].get(column, 0)
                for p in profiles[:i]
                if column in p["null_rates"]
            ]

            if len(historical) < min_history_days:
                continue

            current_rate = profile["null_rates"][column]
            z_score, mean, std = calculate_z_score(current_rate, historical)

            if z_score > null_z_threshold and current_rate > 1.0:
                anomalies.append(create_anomaly(
                    anomaly_type=NULL_RATE_INCREASE,
                    severity=get_severity(z_score),
                    date=profile["date"],
                    column=column,
                    message=f"Null rate jumped to {current_rate:.1f}% (expected ~{mean:.1f}%)",
                    expected_value=mean,
                    actual_value=current_rate,
                    z_score=round(z_score, 2),
                ))

    return anomalies


def detect_value_range_anomalies(
    profiles: list[dict[str, Any]],
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect value range anomalies (min/max out of expected bounds)."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["mins"].keys())

    for column in all_columns:
        for i, profile in enumerate(profiles):
            if i < min_history_days:
                continue

            if column not in profile["mins"]:
                continue

            historical_mins = [
                p["mins"].get(column)
                for p in profiles[:i]
                if column in p["mins"] and p["mins"].get(column) is not None
            ]

            if not historical_mins:
                continue

            historical_min = min(historical_mins)
            current_min = profile["mins"][column]

            if historical_min >= 0 and current_min < 0:
                anomalies.append(create_anomaly(
                    anomaly_type=NEGATIVE_VALUES,
                    severity=HIGH,
                    date=profile["date"],
                    column=column,
                    message=f"Negative values appeared: min={current_min:.2f} (historical min={historical_min:.2f})",
                    expected_value=historical_min,
                    actual_value=current_min,
                    z_score=0,
                ))

    return anomalies


def detect_mean_shift_anomalies(
    profiles: list[dict[str, Any]],
    value_z_threshold: float = DEFAULT_VALUE_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect significant shifts in mean values."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["means"].keys())

    for column in all_columns:
        for i, profile in enumerate(profiles):
            if i < min_history_days:
                continue

            if column not in profile["means"]:
                continue

            historical = [
                p["means"].get(column)
                for p in profiles[:i]
                if column in p["means"] and p["means"].get(column) is not None
            ]

            if len(historical) < min_history_days:
                continue

            current_mean = profile["means"][column]
            z_score, mean, std = calculate_z_score(current_mean, historical)

            if abs(z_score) > value_z_threshold:
                direction = "increased" if z_score > 0 else "decreased"
                anomalies.append(create_anomaly(
                    anomaly_type=MEAN_SHIFT,
                    severity=get_severity(z_score),
                    date=profile["date"],
                    column=column,
                    message=f"Mean {direction} to {current_mean:.2f} (expected ~{mean:.2f})",
                    expected_value=mean,
                    actual_value=current_mean,
                    z_score=round(z_score, 2),
                ))

    return anomalies


def detect_distribution_shift(
    profiles: list[dict[str, Any]],
    threshold: float = 0.15,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect significant shifts in categorical distributions."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["category_distributions"].keys())

    for column in all_columns:
        for i, profile in enumerate(profiles):
            if i < min_history_days:
                continue

            if column not in profile["category_distributions"]:
                continue

            current_dist = profile["category_distributions"][column]

            historical_dists = [
                p["category_distributions"].get(column, {})
                for p in profiles[:i]
                if column in p["category_distributions"]
            ]

            if len(historical_dists) < min_history_days:
                continue

            all_categories = set()
            for d in historical_dists:
                all_categories.update(d.keys())
            all_categories.update(current_dist.keys())

            avg_dist = {}
            for cat in all_categories:
                values = [d.get(cat, 0) for d in historical_dists]
                avg_dist[cat] = np.mean(values)

            for cat in all_categories:
                current_share = current_dist.get(cat, 0)
                expected_share = avg_dist.get(cat, 0)
                diff = abs(current_share - expected_share)

                if diff > threshold:
                    direction = "increased" if current_share > expected_share else "decreased"
                    anomalies.append(create_anomaly(
                        anomaly_type=DISTRIBUTION_SHIFT,
                        severity=MEDIUM if diff < 0.25 else HIGH,
                        date=profile["date"],
                        column=f"{column}:{cat}",
                        message=f"'{cat}' share {direction} from {expected_share:.1%} to {current_share:.1%}",
                        expected_value=expected_share * 100,
                        actual_value=current_share * 100,
                        z_score=0,
                    ))

    return anomalies


def detect_all_anomalies(
    profiles: list[dict[str, Any]],
    volume_z_threshold: float = DEFAULT_VOLUME_Z_THRESHOLD,
    null_z_threshold: float = DEFAULT_NULL_Z_THRESHOLD,
    value_z_threshold: float = DEFAULT_VALUE_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """
    Run all anomaly detection methods.

    Args:
        profiles: List of daily profiles sorted by date
        volume_z_threshold: Z-score threshold for volume anomalies
        null_z_threshold: Z-score threshold for null rate anomalies
        value_z_threshold: Z-score threshold for value anomalies
        min_history_days: Minimum days of history needed

    Returns:
        List of all detected anomalies
    """
    all_anomalies = []

    all_anomalies.extend(detect_volume_anomalies(profiles, volume_z_threshold, min_history_days))
    all_anomalies.extend(detect_null_anomalies(profiles, null_z_threshold, min_history_days))
    all_anomalies.extend(detect_value_range_anomalies(profiles, min_history_days))
    all_anomalies.extend(detect_mean_shift_anomalies(profiles, value_z_threshold, min_history_days))
    all_anomalies.extend(detect_distribution_shift(profiles, min_history_days=min_history_days))

    all_anomalies.sort(key=lambda a: a["date"])

    return all_anomalies


def print_anomalies(anomalies: list[dict[str, Any]]) -> None:
    """Print detected anomalies."""
    if not anomalies:
        print("No anomalies detected")
        return

    print(f"\n{len(anomalies)} anomalies detected\n")

    by_date: dict[str, list[dict[str, Any]]] = {}
    for a in anomalies:
        date_str = a["date"].strftime("%Y-%m-%d")
        if date_str not in by_date:
            by_date[date_str] = []
        by_date[date_str].append(a)

    severity_labels = {
        CRITICAL: "[CRITICAL]",
        HIGH: "[HIGH]",
        MEDIUM: "[MEDIUM]",
        LOW: "[LOW]",
    }

    for date_str, date_anomalies in sorted(by_date.items()):
        print(f"Date: {date_str}")
        print("-" * 40)

        for a in date_anomalies:
            label = severity_labels.get(a["severity"], "[UNKNOWN]")
            column_info = f" [{a['column']}]" if a["column"] else ""
            z_info = f" (z={a['z_score']})" if a["z_score"] != 0 else ""

            print(f"  {label} {a['type']}{column_info}")
            print(f"     {a['message']}{z_info}")

        print()


def detect_date_column(df: pd.DataFrame) -> str | None:
    """Auto-detect a date column by checking datetime types or parsing object columns."""
    # First check for columns already parsed as datetime
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns
    if len(datetime_cols) > 0:
        return datetime_cols[0]

    # Then try parsing object columns
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            pd.to_datetime(df[col])
            return col
        except (ValueError, TypeError):
            continue
    return None


def main():
    """Run anomaly detection on all CSV files in the data directory."""
    data_dir = Path("data")

    print("\n" + "=" * 50)
    print("        Anomaly Detector")
    print("=" * 50)

    # Find all CSV files
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        print("\nNo CSV files found in data/ directory.")
        print("Run data_generator first:")
        print("  python -m src.data_generator")
        return

    print(f"\nFound {len(csv_files)} CSV file(s)")

    all_anomalies: dict[str, list[dict[str, Any]]] = {}

    for file_path in csv_files:
        file_name = file_path.name
        print(f"\n{'=' * 50}")
        print(f"Analyzing: {file_name}")
        print("=" * 50)

        df = pd.read_csv(file_path)

        # Auto-detect date column
        date_column = detect_date_column(df)

        if date_column is None:
            print("  Warning: No date column found, skipping")
            continue

        print(f"  Using date column: {date_column}")

        # Compute daily profiles
        profiles = compute_daily_profiles(df, date_column)
        print(f"  Computed profiles for {len(profiles)} days")

        if len(profiles) < DEFAULT_MIN_HISTORY_DAYS:
            print(f"  Warning: Only {len(profiles)} days of data, need at least {DEFAULT_MIN_HISTORY_DAYS}")
            continue

        # Detect anomalies
        anomalies = detect_all_anomalies(profiles)

        if anomalies:
            all_anomalies[file_name] = anomalies
            print_anomalies(anomalies)
        else:
            print("  No anomalies detected")

    # Summary
    total_anomalies = sum(len(a) for a in all_anomalies.values())

    print("\n" + "=" * 50)
    print(f"SUMMARY: {total_anomalies} total anomalies detected")
    print("=" * 50)

    if total_anomalies > 0:
        # Count by file
        print("\nBy file:")
        for file_name, anomalies in all_anomalies.items():
            print(f"  - {file_name}: {len(anomalies)}")

        # Count by type
        type_counts: dict[str, int] = {}
        for anomalies in all_anomalies.values():
            for a in anomalies:
                t = a["type"]
                type_counts[t] = type_counts.get(t, 0) + 1

        print("\nBy type:")
        for anomaly_type, count in sorted(type_counts.items()):
            print(f"  - {anomaly_type}: {count}")


if __name__ == "__main__":
    main()
