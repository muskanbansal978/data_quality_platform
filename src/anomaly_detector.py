"""
Anomaly detection for data quality monitoring.

Detects anomalies in data profiles using statistical methods (z-score, IQR).
Identifies volume drops, null spikes, value range violations, and distribution shifts.

Run with: python -m src.anomaly_detector
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

# Anomaly types - Volume
VOLUME_DROP = "volume_drop"
VOLUME_SPIKE = "volume_spike"

# Anomaly types - Null/Completeness
NULL_RATE_INCREASE = "null_rate_increase"
COMPLETENESS_VIOLATION = "completeness_violation"

# Anomaly types - Value Range
VALUE_OUT_OF_RANGE = "value_out_of_range"
NEGATIVE_VALUES = "negative_values"
EXTREME_VALUE = "extreme_value"

# Anomaly types - Statistical
DISTRIBUTION_SHIFT = "distribution_shift"
MEAN_SHIFT = "mean_shift"
VARIANCE_SHIFT = "variance_shift"
MEDIAN_SHIFT = "median_shift"
OUTLIER_SPIKE = "outlier_spike"

# Anomaly types - Cardinality
CARDINALITY_INCREASE = "cardinality_increase"
CARDINALITY_DECREASE = "cardinality_decrease"
NEW_CATEGORY = "new_category"
MISSING_CATEGORY = "missing_category"

# Anomaly types - Duplicates
DUPLICATE_SPIKE = "duplicate_spike"
DUPLICATE_KEY_VIOLATION = "duplicate_key_violation"

# Anomaly types - Freshness
DATA_STALENESS = "data_staleness"
FUTURE_DATE = "future_date"
LATE_ARRIVING_DATA = "late_arriving_data"

# Anomaly types - String
STRING_LENGTH_ANOMALY = "string_length_anomaly"
WHITESPACE_ANOMALY = "whitespace_anomaly"
PATTERN_VIOLATION = "pattern_violation"

# Anomaly types - Schema
SCHEMA_DRIFT = "schema_drift"
TYPE_MISMATCH = "type_mismatch"

# Anomaly types - Correlation
CORRELATION_BREAK = "correlation_break"

# Anomaly types - Sequence
SEQUENCE_GAP = "sequence_gap"
OUT_OF_ORDER = "out_of_order"

# Severity levels
LOW = "low"
MEDIUM = "medium"
HIGH = "high"
CRITICAL = "critical"

# Default thresholds
DEFAULT_VOLUME_Z_THRESHOLD = 2.5
DEFAULT_NULL_Z_THRESHOLD = 2.5
DEFAULT_VALUE_Z_THRESHOLD = 3.0
DEFAULT_VARIANCE_Z_THRESHOLD = 2.5
DEFAULT_CARDINALITY_Z_THRESHOLD = 2.5
DEFAULT_DUPLICATE_Z_THRESHOLD = 2.5
DEFAULT_OUTLIER_Z_THRESHOLD = 2.5
DEFAULT_STRING_LENGTH_Z_THRESHOLD = 2.5
DEFAULT_CORRELATION_THRESHOLD = 0.3  # Minimum correlation change to flag
DEFAULT_MIN_HISTORY_DAYS = 5
DEFAULT_OUTLIER_IQR_MULTIPLIER = 1.5


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
    medians: Optional[dict[str, float]] = None,
    variances: Optional[dict[str, float]] = None,
    category_distributions: Optional[dict[str, dict[str, float]]] = None,
    cardinalities: Optional[dict[str, int]] = None,
    unique_values: Optional[dict[str, set]] = None,
    duplicate_count: int = 0,
    duplicate_rate: float = 0.0,
    outlier_counts: Optional[dict[str, int]] = None,
    outlier_rates: Optional[dict[str, float]] = None,
    string_lengths: Optional[dict[str, dict[str, float]]] = None,
    whitespace_counts: Optional[dict[str, int]] = None,
    max_date: Optional[datetime] = None,
    min_date: Optional[datetime] = None,
    columns: Optional[list[str]] = None,
    column_types: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Create a daily profile dictionary with extended metrics."""
    return {
        "date": date,
        "row_count": row_count,
        "null_counts": null_counts or {},
        "null_rates": null_rates or {},
        "means": means or {},
        "mins": mins or {},
        "maxs": maxs or {},
        "stds": stds or {},
        "medians": medians or {},
        "variances": variances or {},
        "category_distributions": category_distributions or {},
        "cardinalities": cardinalities or {},
        "unique_values": unique_values or {},
        "duplicate_count": duplicate_count,
        "duplicate_rate": duplicate_rate,
        "outlier_counts": outlier_counts or {},
        "outlier_rates": outlier_rates or {},
        "string_lengths": string_lengths or {},
        "whitespace_counts": whitespace_counts or {},
        "max_date": max_date,
        "min_date": min_date,
        "columns": columns or [],
        "column_types": column_types or {},
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


def compute_outlier_count(series: pd.Series, iqr_multiplier: float = DEFAULT_OUTLIER_IQR_MULTIPLIER) -> int:
    """Count outliers using IQR method."""
    if len(series) < 4:
        return 0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    return int(((series < lower_bound) | (series > upper_bound)).sum())


def compute_daily_profiles(
    df: pd.DataFrame,
    date_column: str,
    numeric_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    string_columns: Optional[list[str]] = None,
    key_columns: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Compute daily profile statistics from a DataFrame.

    Args:
        df: DataFrame with data
        date_column: Name of the date column
        numeric_columns: Columns to compute numeric stats for
        categorical_columns: Columns to compute distribution stats for
        string_columns: Columns to compute string stats for
        key_columns: Columns to check for duplicate keys

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

    if string_columns is None:
        string_columns = df.select_dtypes(include=["object"]).columns.tolist()
        string_columns = [c for c in string_columns if c != date_column and c != "_date"]

    # Store all columns and their types
    all_columns = [c for c in df.columns if c != "_date"]
    column_types = {col: str(df[col].dtype) for col in all_columns}

    profiles = []

    for date, group in df.groupby("_date"):
        null_counts = {}
        null_rates = {}
        means = {}
        mins = {}
        maxs = {}
        stds = {}
        medians = {}
        variances = {}
        category_distributions = {}
        cardinalities = {}
        unique_values = {}
        outlier_counts = {}
        outlier_rates = {}
        string_lengths = {}
        whitespace_counts = {}

        # Compute null rates for all columns
        for col in df.columns:
            if col in ["_date", date_column]:
                continue
            null_count = group[col].isna().sum()
            null_counts[col] = int(null_count)
            null_rates[col] = null_count / len(group) * 100 if len(group) > 0 else 0

        # Compute numeric stats
        for col in numeric_columns:
            if col in group.columns:
                non_null = group[col].dropna()
                if len(non_null) > 0:
                    means[col] = float(non_null.mean())
                    mins[col] = float(non_null.min())
                    maxs[col] = float(non_null.max())
                    stds[col] = float(non_null.std()) if len(non_null) > 1 else 0.0
                    medians[col] = float(non_null.median())
                    variances[col] = float(non_null.var()) if len(non_null) > 1 else 0.0

                    # Compute outliers
                    outlier_count = compute_outlier_count(non_null)
                    outlier_counts[col] = outlier_count
                    outlier_rates[col] = outlier_count / len(non_null) * 100 if len(non_null) > 0 else 0

        # Compute category distributions and cardinality
        for col in categorical_columns:
            if col in group.columns:
                value_counts = group[col].value_counts(normalize=True)
                category_distributions[col] = value_counts.to_dict()
                cardinalities[col] = int(group[col].nunique())
                unique_values[col] = set(group[col].dropna().unique())

        # Compute string statistics
        for col in string_columns:
            if col in group.columns:
                non_null = group[col].dropna().astype(str)
                if len(non_null) > 0:
                    lengths = non_null.str.len()
                    string_lengths[col] = {
                        "mean": float(lengths.mean()),
                        "min": float(lengths.min()),
                        "max": float(lengths.max()),
                        "std": float(lengths.std()) if len(lengths) > 1 else 0.0,
                    }
                    # Count values with leading/trailing whitespace
                    whitespace_count = int((non_null != non_null.str.strip()).sum())
                    whitespace_counts[col] = whitespace_count

        # Compute duplicate stats
        duplicate_count = int(group.duplicated().sum())
        duplicate_rate = duplicate_count / len(group) * 100 if len(group) > 0 else 0

        # Get date range from datetime columns
        max_date_val = None
        min_date_val = None
        datetime_cols = group.select_dtypes(include=["datetime64"]).columns
        if len(datetime_cols) > 0:
            for dt_col in datetime_cols:
                col_max = group[dt_col].max()
                col_min = group[dt_col].min()
                if pd.notna(col_max):
                    if max_date_val is None or col_max > max_date_val:
                        max_date_val = col_max
                if pd.notna(col_min):
                    if min_date_val is None or col_min < min_date_val:
                        min_date_val = col_min

        profile = create_daily_profile(
            date=datetime.combine(date, datetime.min.time()),
            row_count=len(group),
            null_counts=null_counts,
            null_rates=null_rates,
            means=means,
            mins=mins,
            maxs=maxs,
            stds=stds,
            medians=medians,
            variances=variances,
            category_distributions=category_distributions,
            cardinalities=cardinalities,
            unique_values=unique_values,
            duplicate_count=duplicate_count,
            duplicate_rate=duplicate_rate,
            outlier_counts=outlier_counts,
            outlier_rates=outlier_rates,
            string_lengths=string_lengths,
            whitespace_counts=whitespace_counts,
            max_date=max_date_val,
            min_date=min_date_val,
            columns=all_columns,
            column_types=column_types,
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


def detect_duplicate_anomalies(
    profiles: list[dict[str, Any]],
    duplicate_z_threshold: float = DEFAULT_DUPLICATE_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect anomalies in duplicate rates."""
    anomalies = []

    for i, profile in enumerate(profiles):
        if i < min_history_days:
            continue

        historical = [p["duplicate_rate"] for p in profiles[:i]]
        current_rate = profile["duplicate_rate"]
        z_score, mean, std = calculate_z_score(current_rate, historical)

        if z_score > duplicate_z_threshold and current_rate > 1.0:
            anomalies.append(create_anomaly(
                anomaly_type=DUPLICATE_SPIKE,
                severity=get_severity(z_score),
                date=profile["date"],
                column=None,
                message=f"Duplicate rate spiked to {current_rate:.1f}% ({profile['duplicate_count']} duplicates, expected ~{mean:.1f}%)",
                expected_value=mean,
                actual_value=current_rate,
                z_score=round(z_score, 2),
            ))

    return anomalies


def detect_cardinality_anomalies(
    profiles: list[dict[str, Any]],
    cardinality_z_threshold: float = DEFAULT_CARDINALITY_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect cardinality changes and new/missing categories."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["cardinalities"].keys())

    for column in all_columns:
        # Track all seen values across history
        all_seen_values: set = set()

        for i, profile in enumerate(profiles):
            if column not in profile["cardinalities"]:
                continue

            current_values = profile.get("unique_values", {}).get(column, set())

            if i < min_history_days:
                all_seen_values.update(current_values)
                continue

            # Check for cardinality z-score anomalies
            historical = [
                p["cardinalities"].get(column, 0)
                for p in profiles[:i]
                if column in p["cardinalities"]
            ]

            if len(historical) < min_history_days:
                all_seen_values.update(current_values)
                continue

            current_cardinality = profile["cardinalities"][column]
            z_score, mean, std = calculate_z_score(current_cardinality, historical)

            if z_score > cardinality_z_threshold:
                anomalies.append(create_anomaly(
                    anomaly_type=CARDINALITY_INCREASE,
                    severity=get_severity(z_score),
                    date=profile["date"],
                    column=column,
                    message=f"Cardinality increased to {current_cardinality} unique values (expected ~{mean:.0f})",
                    expected_value=mean,
                    actual_value=current_cardinality,
                    z_score=round(z_score, 2),
                ))
            elif z_score < -cardinality_z_threshold:
                anomalies.append(create_anomaly(
                    anomaly_type=CARDINALITY_DECREASE,
                    severity=get_severity(z_score),
                    date=profile["date"],
                    column=column,
                    message=f"Cardinality decreased to {current_cardinality} unique values (expected ~{mean:.0f})",
                    expected_value=mean,
                    actual_value=current_cardinality,
                    z_score=round(z_score, 2),
                ))

            # Check for new categories
            new_values = current_values - all_seen_values
            if new_values and len(new_values) <= 5:  # Only report if reasonable number
                for new_val in list(new_values)[:3]:  # Limit to 3 per day
                    anomalies.append(create_anomaly(
                        anomaly_type=NEW_CATEGORY,
                        severity=LOW,
                        date=profile["date"],
                        column=column,
                        message=f"New category value appeared: '{new_val}'",
                        expected_value=0,
                        actual_value=1,
                        z_score=0,
                    ))

            # Check for missing categories (values that were common but disappeared)
            if all_seen_values:
                # Get commonly occurring values from history
                common_values = set()
                for p in profiles[:i]:
                    p_values = p.get("unique_values", {}).get(column, set())
                    common_values.update(p_values)

                missing_values = common_values - current_values
                if missing_values and len(missing_values) <= 3:
                    for missing_val in list(missing_values)[:2]:
                        anomalies.append(create_anomaly(
                            anomaly_type=MISSING_CATEGORY,
                            severity=LOW,
                            date=profile["date"],
                            column=column,
                            message=f"Category value missing: '{missing_val}'",
                            expected_value=1,
                            actual_value=0,
                            z_score=0,
                        ))

            all_seen_values.update(current_values)

    return anomalies


def detect_variance_anomalies(
    profiles: list[dict[str, Any]],
    variance_z_threshold: float = DEFAULT_VARIANCE_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect significant changes in variance/standard deviation."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["variances"].keys())

    for column in all_columns:
        for i, profile in enumerate(profiles):
            if i < min_history_days:
                continue

            if column not in profile["variances"]:
                continue

            historical = [
                p["variances"].get(column)
                for p in profiles[:i]
                if column in p["variances"] and p["variances"].get(column) is not None
            ]

            if len(historical) < min_history_days:
                continue

            current_variance = profile["variances"][column]
            z_score, mean, std = calculate_z_score(current_variance, historical)

            if abs(z_score) > variance_z_threshold:
                direction = "increased" if z_score > 0 else "decreased"
                current_std = profile["stds"].get(column, 0)
                historical_std = np.mean([p["stds"].get(column, 0) for p in profiles[:i] if column in p["stds"]])
                anomalies.append(create_anomaly(
                    anomaly_type=VARIANCE_SHIFT,
                    severity=get_severity(z_score),
                    date=profile["date"],
                    column=column,
                    message=f"Variance {direction}: std {direction} from {historical_std:.2f} to {current_std:.2f}",
                    expected_value=mean,
                    actual_value=current_variance,
                    z_score=round(z_score, 2),
                ))

    return anomalies


def detect_median_shift_anomalies(
    profiles: list[dict[str, Any]],
    value_z_threshold: float = DEFAULT_VALUE_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect significant shifts in median values (more robust than mean)."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["medians"].keys())

    for column in all_columns:
        for i, profile in enumerate(profiles):
            if i < min_history_days:
                continue

            if column not in profile["medians"]:
                continue

            historical = [
                p["medians"].get(column)
                for p in profiles[:i]
                if column in p["medians"] and p["medians"].get(column) is not None
            ]

            if len(historical) < min_history_days:
                continue

            current_median = profile["medians"][column]
            z_score, mean, std = calculate_z_score(current_median, historical)

            if abs(z_score) > value_z_threshold:
                direction = "increased" if z_score > 0 else "decreased"
                anomalies.append(create_anomaly(
                    anomaly_type=MEDIAN_SHIFT,
                    severity=get_severity(z_score),
                    date=profile["date"],
                    column=column,
                    message=f"Median {direction} to {current_median:.2f} (expected ~{mean:.2f})",
                    expected_value=mean,
                    actual_value=current_median,
                    z_score=round(z_score, 2),
                ))

    return anomalies


def detect_outlier_anomalies(
    profiles: list[dict[str, Any]],
    outlier_z_threshold: float = DEFAULT_OUTLIER_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect spikes in outlier counts."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["outlier_rates"].keys())

    for column in all_columns:
        for i, profile in enumerate(profiles):
            if i < min_history_days:
                continue

            if column not in profile["outlier_rates"]:
                continue

            historical = [
                p["outlier_rates"].get(column, 0)
                for p in profiles[:i]
                if column in p["outlier_rates"]
            ]

            if len(historical) < min_history_days:
                continue

            current_rate = profile["outlier_rates"][column]
            z_score, mean, std = calculate_z_score(current_rate, historical)

            if z_score > outlier_z_threshold and current_rate > 1.0:
                outlier_count = profile["outlier_counts"].get(column, 0)
                anomalies.append(create_anomaly(
                    anomaly_type=OUTLIER_SPIKE,
                    severity=get_severity(z_score),
                    date=profile["date"],
                    column=column,
                    message=f"Outlier rate spiked to {current_rate:.1f}% ({outlier_count} outliers, expected ~{mean:.1f}%)",
                    expected_value=mean,
                    actual_value=current_rate,
                    z_score=round(z_score, 2),
                ))

    return anomalies


def detect_extreme_values(
    profiles: list[dict[str, Any]],
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect extreme values that exceed historical bounds significantly."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["maxs"].keys())

    for column in all_columns:
        for i, profile in enumerate(profiles):
            if i < min_history_days:
                continue

            if column not in profile["maxs"] or column not in profile["mins"]:
                continue

            historical_maxs = [
                p["maxs"].get(column)
                for p in profiles[:i]
                if column in p["maxs"] and p["maxs"].get(column) is not None
            ]
            historical_mins = [
                p["mins"].get(column)
                for p in profiles[:i]
                if column in p["mins"] and p["mins"].get(column) is not None
            ]

            if not historical_maxs or not historical_mins:
                continue

            hist_max = max(historical_maxs)
            hist_min = min(historical_mins)
            hist_range = hist_max - hist_min

            if hist_range == 0:
                continue

            current_max = profile["maxs"][column]
            current_min = profile["mins"][column]

            # Check if current max exceeds historical by more than 50% of range
            if current_max > hist_max + 0.5 * hist_range:
                excess = (current_max - hist_max) / hist_range * 100
                anomalies.append(create_anomaly(
                    anomaly_type=EXTREME_VALUE,
                    severity=HIGH if excess > 100 else MEDIUM,
                    date=profile["date"],
                    column=column,
                    message=f"Extreme max value: {current_max:.2f} exceeds historical max {hist_max:.2f} by {excess:.0f}% of range",
                    expected_value=hist_max,
                    actual_value=current_max,
                    z_score=0,
                ))

            # Check if current min is below historical by more than 50% of range
            if current_min < hist_min - 0.5 * hist_range:
                excess = (hist_min - current_min) / hist_range * 100
                anomalies.append(create_anomaly(
                    anomaly_type=EXTREME_VALUE,
                    severity=HIGH if excess > 100 else MEDIUM,
                    date=profile["date"],
                    column=column,
                    message=f"Extreme min value: {current_min:.2f} below historical min {hist_min:.2f} by {excess:.0f}% of range",
                    expected_value=hist_min,
                    actual_value=current_min,
                    z_score=0,
                ))

    return anomalies


def detect_freshness_anomalies(
    profiles: list[dict[str, Any]],
    reference_date: Optional[datetime] = None,
    staleness_days: int = 2,
) -> list[dict[str, Any]]:
    """Detect data freshness issues like stale data or future dates."""
    anomalies = []

    if reference_date is None:
        reference_date = datetime.now()

    for profile in profiles:
        profile_date = profile["date"]

        # Check for future dates in the data
        if profile.get("max_date") and profile["max_date"] > reference_date:
            days_in_future = (profile["max_date"] - reference_date).days
            anomalies.append(create_anomaly(
                anomaly_type=FUTURE_DATE,
                severity=HIGH,
                date=profile_date,
                column=None,
                message=f"Data contains future dates: {profile['max_date']} is {days_in_future} days in the future",
                expected_value=0,
                actual_value=days_in_future,
                z_score=0,
            ))

    # Check for data staleness (gap between last profile date and reference)
    if profiles:
        last_profile = profiles[-1]
        last_date = last_profile["date"]
        days_stale = (reference_date - last_date).days

        if days_stale > staleness_days:
            anomalies.append(create_anomaly(
                anomaly_type=DATA_STALENESS,
                severity=MEDIUM if days_stale < 7 else HIGH,
                date=last_date,
                column=None,
                message=f"Data is {days_stale} days stale (last update: {last_date.strftime('%Y-%m-%d')})",
                expected_value=staleness_days,
                actual_value=days_stale,
                z_score=0,
            ))

    return anomalies


def detect_string_anomalies(
    profiles: list[dict[str, Any]],
    string_length_z_threshold: float = DEFAULT_STRING_LENGTH_Z_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect string-related anomalies like length changes and whitespace issues."""
    anomalies = []

    all_columns = set()
    for p in profiles:
        all_columns.update(p["string_lengths"].keys())

    for column in all_columns:
        for i, profile in enumerate(profiles):
            if i < min_history_days:
                continue

            if column not in profile["string_lengths"]:
                continue

            current_stats = profile["string_lengths"][column]
            current_mean_length = current_stats["mean"]

            historical_means = [
                p["string_lengths"].get(column, {}).get("mean")
                for p in profiles[:i]
                if column in p["string_lengths"] and p["string_lengths"][column].get("mean") is not None
            ]

            if len(historical_means) < min_history_days:
                continue

            z_score, mean, std = calculate_z_score(current_mean_length, historical_means)

            if abs(z_score) > string_length_z_threshold:
                direction = "increased" if z_score > 0 else "decreased"
                anomalies.append(create_anomaly(
                    anomaly_type=STRING_LENGTH_ANOMALY,
                    severity=get_severity(z_score),
                    date=profile["date"],
                    column=column,
                    message=f"Average string length {direction} from {mean:.1f} to {current_mean_length:.1f} chars",
                    expected_value=mean,
                    actual_value=current_mean_length,
                    z_score=round(z_score, 2),
                ))

            # Check for whitespace anomalies
            current_whitespace = profile["whitespace_counts"].get(column, 0)
            historical_whitespace = [
                p["whitespace_counts"].get(column, 0)
                for p in profiles[:i]
                if column in p["whitespace_counts"]
            ]

            if historical_whitespace and current_whitespace > 0:
                ws_z_score, ws_mean, ws_std = calculate_z_score(current_whitespace, historical_whitespace)
                if ws_z_score > string_length_z_threshold:
                    anomalies.append(create_anomaly(
                        anomaly_type=WHITESPACE_ANOMALY,
                        severity=MEDIUM,
                        date=profile["date"],
                        column=column,
                        message=f"Whitespace issues increased: {current_whitespace} values have leading/trailing whitespace (expected ~{ws_mean:.0f})",
                        expected_value=ws_mean,
                        actual_value=current_whitespace,
                        z_score=round(ws_z_score, 2),
                    ))

    return anomalies


def detect_schema_drift(
    profiles: list[dict[str, Any]],
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect schema changes like new columns, missing columns, or type changes."""
    anomalies = []

    if len(profiles) < 2:
        return anomalies

    # Get baseline columns from early profiles
    baseline_columns = set()
    baseline_types = {}

    for i, profile in enumerate(profiles[:min_history_days]):
        baseline_columns.update(profile.get("columns", []))
        for col, dtype in profile.get("column_types", {}).items():
            if col not in baseline_types:
                baseline_types[col] = dtype

    # Check subsequent profiles for drift
    for i, profile in enumerate(profiles):
        if i < min_history_days:
            continue

        current_columns = set(profile.get("columns", []))
        current_types = profile.get("column_types", {})

        # Check for new columns
        new_columns = current_columns - baseline_columns
        for col in new_columns:
            anomalies.append(create_anomaly(
                anomaly_type=SCHEMA_DRIFT,
                severity=MEDIUM,
                date=profile["date"],
                column=col,
                message=f"New column appeared: '{col}' (type: {current_types.get(col, 'unknown')})",
                expected_value=0,
                actual_value=1,
                z_score=0,
            ))

        # Check for missing columns
        missing_columns = baseline_columns - current_columns
        for col in missing_columns:
            anomalies.append(create_anomaly(
                anomaly_type=SCHEMA_DRIFT,
                severity=HIGH,
                date=profile["date"],
                column=col,
                message=f"Column disappeared: '{col}'",
                expected_value=1,
                actual_value=0,
                z_score=0,
            ))

        # Check for type changes
        for col in current_columns & baseline_columns:
            if col in baseline_types and col in current_types:
                if baseline_types[col] != current_types[col]:
                    anomalies.append(create_anomaly(
                        anomaly_type=TYPE_MISMATCH,
                        severity=HIGH,
                        date=profile["date"],
                        column=col,
                        message=f"Column type changed: '{col}' from {baseline_types[col]} to {current_types[col]}",
                        expected_value=0,
                        actual_value=1,
                        z_score=0,
                    ))

        # Update baseline for next iteration
        baseline_columns.update(current_columns)
        baseline_types.update(current_types)

    return anomalies


def detect_sequence_anomalies(
    df: pd.DataFrame,
    id_column: str,
    date_column: str,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect gaps or issues in sequential ID columns."""
    anomalies = []

    if id_column not in df.columns or date_column not in df.columns:
        return anomalies

    df = df.copy()
    df["_date"] = pd.to_datetime(df[date_column]).dt.date

    # Check if ID column is numeric (sequential IDs)
    if not pd.api.types.is_numeric_dtype(df[id_column]):
        return anomalies

    for date, group in df.groupby("_date"):
        ids = group[id_column].dropna().sort_values()

        if len(ids) < 2:
            continue

        # Check for gaps in sequential IDs
        id_diffs = ids.diff().dropna()
        gaps = id_diffs[id_diffs > 1]

        if len(gaps) > 0:
            total_missing = int(gaps.sum() - len(gaps))
            max_gap = int(gaps.max())

            if total_missing > 10:  # Only report significant gaps
                anomalies.append(create_anomaly(
                    anomaly_type=SEQUENCE_GAP,
                    severity=MEDIUM if total_missing < 100 else HIGH,
                    date=datetime.combine(date, datetime.min.time()),
                    column=id_column,
                    message=f"Sequence gaps detected: ~{total_missing} missing IDs (largest gap: {max_gap})",
                    expected_value=0,
                    actual_value=total_missing,
                    z_score=0,
                ))

    return anomalies


def detect_correlation_anomalies(
    df: pd.DataFrame,
    date_column: str,
    column_pairs: Optional[list[tuple[str, str]]] = None,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """Detect significant changes in correlations between numeric columns."""
    anomalies = []

    df = df.copy()
    df["_date"] = pd.to_datetime(df[date_column]).dt.date

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if column_pairs is None:
        # Auto-detect pairs with historically strong correlations
        if len(numeric_cols) < 2:
            return anomalies
        column_pairs = []
        for i, col1 in enumerate(numeric_cols[:10]):  # Limit to prevent explosion
            for col2 in numeric_cols[i + 1:10]:
                column_pairs.append((col1, col2))

    daily_correlations: dict[str, list[tuple[datetime, float]]] = {
        f"{c1}|{c2}": [] for c1, c2 in column_pairs
    }

    dates = sorted(df["_date"].unique())

    for date in dates:
        group = df[df["_date"] == date]

        for col1, col2 in column_pairs:
            if col1 not in group.columns or col2 not in group.columns:
                continue

            valid_data = group[[col1, col2]].dropna()
            if len(valid_data) < 10:
                continue

            corr = valid_data[col1].corr(valid_data[col2])
            if pd.notna(corr):
                key = f"{col1}|{col2}"
                daily_correlations[key].append((
                    datetime.combine(date, datetime.min.time()),
                    corr
                ))

    # Check for correlation breaks
    for pair_key, corr_history in daily_correlations.items():
        if len(corr_history) < min_history_days + 1:
            continue

        col1, col2 = pair_key.split("|")

        for i in range(min_history_days, len(corr_history)):
            date, current_corr = corr_history[i]
            historical_corrs = [c[1] for c in corr_history[:i]]

            mean_corr = np.mean(historical_corrs)
            corr_change = abs(current_corr - mean_corr)

            # Only flag if there was a strong historical correlation that broke
            if abs(mean_corr) > 0.5 and corr_change > correlation_threshold:
                anomalies.append(create_anomaly(
                    anomaly_type=CORRELATION_BREAK,
                    severity=MEDIUM if corr_change < 0.5 else HIGH,
                    date=date,
                    column=f"{col1} vs {col2}",
                    message=f"Correlation changed from {mean_corr:.2f} to {current_corr:.2f} (Δ={corr_change:.2f})",
                    expected_value=mean_corr,
                    actual_value=current_corr,
                    z_score=0,
                ))

    return anomalies


def detect_all_anomalies(
    profiles: list[dict[str, Any]],
    df: Optional[pd.DataFrame] = None,
    date_column: Optional[str] = None,
    volume_z_threshold: float = DEFAULT_VOLUME_Z_THRESHOLD,
    null_z_threshold: float = DEFAULT_NULL_Z_THRESHOLD,
    value_z_threshold: float = DEFAULT_VALUE_Z_THRESHOLD,
    variance_z_threshold: float = DEFAULT_VARIANCE_Z_THRESHOLD,
    cardinality_z_threshold: float = DEFAULT_CARDINALITY_Z_THRESHOLD,
    duplicate_z_threshold: float = DEFAULT_DUPLICATE_Z_THRESHOLD,
    outlier_z_threshold: float = DEFAULT_OUTLIER_Z_THRESHOLD,
    string_length_z_threshold: float = DEFAULT_STRING_LENGTH_Z_THRESHOLD,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    check_freshness: bool = True,
    check_correlations: bool = True,
    check_sequences: bool = False,
    id_column: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Run all anomaly detection methods.

    Args:
        profiles: List of daily profiles sorted by date
        df: Original DataFrame (needed for correlation and sequence checks)
        date_column: Name of date column (needed for correlation and sequence checks)
        volume_z_threshold: Z-score threshold for volume anomalies
        null_z_threshold: Z-score threshold for null rate anomalies
        value_z_threshold: Z-score threshold for value anomalies
        variance_z_threshold: Z-score threshold for variance anomalies
        cardinality_z_threshold: Z-score threshold for cardinality anomalies
        duplicate_z_threshold: Z-score threshold for duplicate anomalies
        outlier_z_threshold: Z-score threshold for outlier anomalies
        string_length_z_threshold: Z-score threshold for string length anomalies
        correlation_threshold: Threshold for correlation change detection
        min_history_days: Minimum days of history needed
        check_freshness: Whether to check for data freshness issues
        check_correlations: Whether to check for correlation breaks
        check_sequences: Whether to check for sequence gaps
        id_column: Column to check for sequence gaps

    Returns:
        List of all detected anomalies
    """
    all_anomalies = []

    # Original detectors
    all_anomalies.extend(detect_volume_anomalies(profiles, volume_z_threshold, min_history_days))
    all_anomalies.extend(detect_null_anomalies(profiles, null_z_threshold, min_history_days))
    all_anomalies.extend(detect_value_range_anomalies(profiles, min_history_days))
    all_anomalies.extend(detect_mean_shift_anomalies(profiles, value_z_threshold, min_history_days))
    all_anomalies.extend(detect_distribution_shift(profiles, min_history_days=min_history_days))

    # New statistical detectors
    all_anomalies.extend(detect_duplicate_anomalies(profiles, duplicate_z_threshold, min_history_days))
    all_anomalies.extend(detect_cardinality_anomalies(profiles, cardinality_z_threshold, min_history_days))
    all_anomalies.extend(detect_variance_anomalies(profiles, variance_z_threshold, min_history_days))
    all_anomalies.extend(detect_median_shift_anomalies(profiles, value_z_threshold, min_history_days))
    all_anomalies.extend(detect_outlier_anomalies(profiles, outlier_z_threshold, min_history_days))
    all_anomalies.extend(detect_extreme_values(profiles, min_history_days))

    # String and schema detectors
    all_anomalies.extend(detect_string_anomalies(profiles, string_length_z_threshold, min_history_days))
    all_anomalies.extend(detect_schema_drift(profiles, min_history_days))

    # Freshness check
    if check_freshness:
        all_anomalies.extend(detect_freshness_anomalies(profiles))

    # Correlation check (requires original DataFrame)
    if check_correlations and df is not None and date_column is not None:
        all_anomalies.extend(detect_correlation_anomalies(
            df, date_column, correlation_threshold=correlation_threshold, min_history_days=min_history_days
        ))

    # Sequence check (requires original DataFrame and ID column)
    if check_sequences and df is not None and date_column is not None and id_column is not None:
        all_anomalies.extend(detect_sequence_anomalies(df, id_column, date_column, min_history_days))

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


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect a date column by checking datetime types, column names, or parsing object columns."""
    # First check for columns already parsed as datetime
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns
    if len(datetime_cols) > 0:
        return datetime_cols[0]

    # Check for columns with date-related names
    date_keywords = ['date', 'time', 'timestamp', 'dt', 'day', 'created', 'updated', 'modified']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            # Verify it can be parsed as datetime
            try:
                pd.to_datetime(df[col])
                return col
            except (ValueError, TypeError):
                continue

    # Finally, try parsing any object columns that might be dates
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

        # Auto-detect ID column for sequence checks
        id_column = None
        for col in df.columns:
            if col.endswith("_id") and pd.api.types.is_numeric_dtype(df[col]):
                id_column = col
                break

        # Detect anomalies with enhanced detection
        anomalies = detect_all_anomalies(
            profiles,
            df=df,
            date_column=date_column,
            check_freshness=True,
            check_correlations=True,
            check_sequences=id_column is not None,
            id_column=id_column,
        )

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
