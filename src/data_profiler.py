"""
Data profiler for computing statistical profiles of datasets.

This module provides tools to analyze datasets and compute comprehensive
statistical profiles that serve as baselines for anomaly detection.

Run with: python -m src.data_profiler
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Default thresholds
DEFAULT_NULL_THRESHOLD = 0.05  # Alert if null% changes by more than 5%
DEFAULT_NUMERIC_THRESHOLD = 0.10  # Alert if mean changes by more than 10%
DEFAULT_OUTLIER_IQR_MULTIPLIER = 1.5

# Common regex patterns for validation
PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "phone": r"^[\d\s\-\+\(\)]{7,20}$",
    "url": r"^https?://[^\s]+$",
    "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "date_iso": r"^\d{4}-\d{2}-\d{2}$",
}


def create_column_profile(
    name: str,
    dtype: str,
    count: int,
    null_count: int,
    null_percent: float,
    **kwargs
) -> dict[str, Any]:
    """
    Create a column profile dictionary.

    Args:
        name: Column name
        dtype: Data type
        count: Total count
        null_count: Number of null values
        null_percent: Percentage of null values
        **kwargs: Additional stats (mean, std, unique_count, etc.)

    Returns:
        Dictionary containing column profile
    """
    profile = {
        "name": name,
        "dtype": dtype,
        "count": count,
        "null_count": null_count,
        "null_percent": null_percent,
    }

    # Add any additional stats, filtering out None values
    profile.update({k: v for k, v in kwargs.items() if v is not None})

    return profile


def create_table_profile(
    name: str,
    profiled_at: datetime,
    row_count: int,
    column_count: int,
    memory_bytes: int,
    columns: dict[str, dict[str, Any]],
    total_null_cells: int = 0,
    null_cell_percent: float = 0.0,
    duplicate_rows: int = 0,
    duplicate_percent: float = 0.0,
    correlations: Optional[dict[str, dict[str, float]]] = None,
    high_correlations: Optional[list[dict[str, Any]]] = None,
    constant_columns: Optional[list[str]] = None,
    high_null_columns: Optional[list[str]] = None,
    numeric_column_count: int = 0,
    categorical_column_count: int = 0,
    datetime_column_count: int = 0,
) -> dict[str, Any]:
    """
    Create a table profile dictionary.

    Args:
        name: Dataset name
        profiled_at: Timestamp when profiled
        row_count: Number of rows
        column_count: Number of columns
        memory_bytes: Memory usage in bytes
        columns: Dictionary of column profiles
        total_null_cells: Total number of null cells
        null_cell_percent: Percentage of null cells
        duplicate_rows: Number of duplicate rows
        duplicate_percent: Percentage of duplicate rows
        correlations: Correlation matrix for numeric columns
        high_correlations: List of highly correlated column pairs
        constant_columns: List of columns with single unique value
        high_null_columns: List of columns with high null rates
        numeric_column_count: Count of numeric columns
        categorical_column_count: Count of categorical columns
        datetime_column_count: Count of datetime columns

    Returns:
        Dictionary containing table profile
    """
    return {
        "name": name,
        "profiled_at": profiled_at,
        "row_count": row_count,
        "column_count": column_count,
        "memory_bytes": memory_bytes,
        "columns": columns,
        "total_null_cells": total_null_cells,
        "null_cell_percent": null_cell_percent,
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": duplicate_percent,
        "correlations": correlations or {},
        "high_correlations": high_correlations or [],
        "constant_columns": constant_columns or [],
        "high_null_columns": high_null_columns or [],
        "column_types": {
            "numeric": numeric_column_count,
            "categorical": categorical_column_count,
            "datetime": datetime_column_count,
        },
    }


def create_profile_comparison(
    baseline_name: str,
    current_name: str,
    compared_at: datetime,
    changes: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Create a profile comparison dictionary.

    Args:
        baseline_name: Name of baseline profile
        current_name: Name of current profile
        compared_at: Timestamp when compared
        changes: List of detected changes

    Returns:
        Dictionary containing comparison results
    """
    return {
        "baseline_name": baseline_name,
        "current_name": current_name,
        "compared_at": compared_at,
        "changes": changes,
    }


def has_significant_changes(comparison: dict[str, Any]) -> bool:
    """Check if comparison has significant changes."""
    return len(comparison["changes"]) > 0


def compute_outliers(series: pd.Series, iqr_multiplier: float = DEFAULT_OUTLIER_IQR_MULTIPLIER) -> dict[str, Any]:
    """Compute outlier statistics using IQR method."""
    if len(series) < 4:
        return {"count": 0, "percent": 0.0, "lower_bound": None, "upper_bound": None}

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return {
        "count": int(len(outliers)),
        "percent": round(len(outliers) / len(series) * 100, 2) if len(series) > 0 else 0.0,
        "lower_bound": round(float(lower_bound), 4),
        "upper_bound": round(float(upper_bound), 4),
    }


def add_numeric_stats(profile: dict[str, Any], series: pd.Series) -> dict[str, Any]:
    """Add comprehensive numeric statistics to a column profile."""
    non_null = series.dropna()

    if len(non_null) > 0:
        percentiles = non_null.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

        # Basic stats
        mean_val = float(non_null.mean())
        std_val = float(non_null.std()) if len(non_null) > 1 else 0.0
        median_val = float(non_null.median())
        variance_val = float(non_null.var()) if len(non_null) > 1 else 0.0

        profile.update({
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "variance": round(variance_val, 4),
            "median": round(median_val, 4),
            "min_value": round(float(non_null.min()), 4),
            "max_value": round(float(non_null.max()), 4),
            "range": round(float(non_null.max() - non_null.min()), 4),
            "percentile_01": round(float(percentiles[0.01]), 4),
            "percentile_05": round(float(percentiles[0.05]), 4),
            "percentile_25": round(float(percentiles[0.25]), 4),
            "percentile_50": round(float(percentiles[0.5]), 4),
            "percentile_75": round(float(percentiles[0.75]), 4),
            "percentile_95": round(float(percentiles[0.95]), 4),
            "percentile_99": round(float(percentiles[0.99]), 4),
            "iqr": round(float(percentiles[0.75] - percentiles[0.25]), 4),
        })

        # Skewness and Kurtosis (need at least 3 values)
        if len(non_null) >= 3:
            try:
                skew_val = float(scipy_stats.skew(non_null, nan_policy='omit'))
                profile["skewness"] = round(skew_val, 4)
            except Exception:
                profile["skewness"] = None

        if len(non_null) >= 4:
            try:
                kurt_val = float(scipy_stats.kurtosis(non_null, nan_policy='omit'))
                profile["kurtosis"] = round(kurt_val, 4)
            except Exception:
                profile["kurtosis"] = None

        # Zero and negative value counts
        zero_count = int((non_null == 0).sum())
        negative_count = int((non_null < 0).sum())
        positive_count = int((non_null > 0).sum())

        profile.update({
            "zero_count": zero_count,
            "zero_percent": round(zero_count / len(non_null) * 100, 2),
            "negative_count": negative_count,
            "negative_percent": round(negative_count / len(non_null) * 100, 2),
            "positive_count": positive_count,
        })

        # Outlier detection
        outlier_stats = compute_outliers(non_null)
        profile["outliers"] = outlier_stats

        # Check if column might be categorical (low cardinality numeric)
        unique_count = non_null.nunique()
        if unique_count <= 20 and unique_count < len(non_null) * 0.05:
            profile["possibly_categorical"] = True
            profile["unique_values"] = sorted(non_null.unique().tolist())[:20]

        # Check if column is constant
        if unique_count == 1:
            profile["is_constant"] = True
            profile["constant_value"] = float(non_null.iloc[0])

    return profile


def detect_pattern(series: pd.Series) -> Optional[str]:
    """Detect if a string column matches a common pattern."""
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return None

    # Sample for performance
    sample = non_null.head(100)

    for pattern_name, pattern in PATTERNS.items():
        matches = sample.str.match(pattern, na=False).sum()
        if matches / len(sample) > 0.8:  # 80% match threshold
            return pattern_name

    return None


def add_string_stats(profile: dict[str, Any], series: pd.Series) -> dict[str, Any]:
    """Add comprehensive string/categorical statistics to a column profile."""
    non_null = series.dropna()

    if len(non_null) > 0:
        unique_count = non_null.nunique()

        # Top 10 most common values
        value_counts = non_null.value_counts()
        top_values = list(zip(
            value_counts.head(10).index.tolist(),
            value_counts.head(10).values.tolist()
        ))

        # Bottom 5 least common values (if enough unique values)
        bottom_values = []
        if unique_count > 10:
            bottom_values = list(zip(
                value_counts.tail(5).index.tolist(),
                value_counts.tail(5).values.tolist()
            ))

        profile.update({
            "unique_count": unique_count,
            "unique_percent": round(unique_count / len(non_null) * 100, 2),
            "top_values": top_values,
            "bottom_values": bottom_values,
        })

        # Check if column is constant
        if unique_count == 1:
            profile["is_constant"] = True
            profile["constant_value"] = str(non_null.iloc[0])

        # String length statistics
        if series.dtype == "object":
            str_series = non_null.astype(str)
            lengths = str_series.str.len()

            profile["length_stats"] = {
                "mean": round(float(lengths.mean()), 2),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "std": round(float(lengths.std()), 2) if len(lengths) > 1 else 0.0,
                "median": int(lengths.median()),
            }

            # Empty string count (different from null)
            empty_count = int((str_series == "").sum())
            profile["empty_count"] = empty_count
            profile["empty_percent"] = round(empty_count / len(non_null) * 100, 2)

            # Whitespace issues
            whitespace_leading = int(str_series.str.match(r"^\s").sum())
            whitespace_trailing = int(str_series.str.match(r"\s$").sum())
            whitespace_only = int(str_series.str.match(r"^\s*$").sum())

            profile["whitespace_issues"] = {
                "leading": whitespace_leading,
                "trailing": whitespace_trailing,
                "only_whitespace": whitespace_only,
                "total_with_issues": int((str_series != str_series.str.strip()).sum()),
            }

            # Case analysis
            lowercase_count = int(str_series.str.islower().sum())
            uppercase_count = int(str_series.str.isupper().sum())
            mixedcase_count = len(str_series) - lowercase_count - uppercase_count

            profile["case_stats"] = {
                "lowercase": lowercase_count,
                "uppercase": uppercase_count,
                "mixed": mixedcase_count,
            }

            # Pattern detection
            detected_pattern = detect_pattern(non_null)
            if detected_pattern:
                profile["detected_pattern"] = detected_pattern

                # Validate pattern match rate
                pattern_regex = PATTERNS[detected_pattern]
                match_count = int(str_series.str.match(pattern_regex, na=False).sum())
                profile["pattern_match_rate"] = round(match_count / len(str_series) * 100, 2)
                profile["pattern_violations"] = len(str_series) - match_count

            # Check for potential numeric strings
            numeric_like = str_series.str.match(r"^-?\d+\.?\d*$", na=False).sum()
            if numeric_like / len(str_series) > 0.8:
                profile["possibly_numeric"] = True
                profile["numeric_like_percent"] = round(numeric_like / len(str_series) * 100, 2)

    return profile


def add_datetime_stats(profile: dict[str, Any], series: pd.Series) -> dict[str, Any]:
    """Add datetime statistics to a column profile."""
    non_null = series.dropna()

    if len(non_null) > 0:
        min_date = non_null.min()
        max_date = non_null.max()

        profile.update({
            "min_date": str(min_date),
            "max_date": str(max_date),
            "date_range_days": (max_date - min_date).days,
        })

    return profile


def profile_column(series: pd.Series, name: str) -> dict[str, Any]:
    """Profile a single column based on its type."""
    dtype_str = str(series.dtype)
    count = len(series)
    null_count = series.isna().sum()
    null_percent = (null_count / count * 100) if count > 0 else 0

    base_profile = create_column_profile(
        name=name,
        dtype=dtype_str,
        count=count,
        null_count=null_count,
        null_percent=round(null_percent, 2),
    )

    # Determine column type and compute appropriate stats
    if pd.api.types.is_numeric_dtype(series):
        return add_numeric_stats(base_profile, series)
    elif pd.api.types.is_datetime64_any_dtype(series):
        return add_datetime_stats(base_profile, series)
    else:
        # Treat as string/categorical
        return add_string_stats(base_profile, series)


def compute_correlation_matrix(
    df: pd.DataFrame,
    threshold: float = 0.7,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """
    Compute correlation matrix for numeric columns.

    Args:
        df: DataFrame with data
        threshold: Threshold for flagging high correlations

    Returns:
        Tuple of (correlation dict, list of high correlation pairs)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {}, []

    # Limit to first 50 columns for performance
    numeric_cols = numeric_cols[:50]

    corr_matrix = df[numeric_cols].corr()
    correlations = {}
    high_correlations = []

    for col in numeric_cols:
        correlations[col] = {}
        for other_col in numeric_cols:
            corr_val = corr_matrix.loc[col, other_col]
            if pd.notna(corr_val):
                correlations[col][other_col] = round(float(corr_val), 4)

                # Track high correlations (excluding self-correlation)
                if col < other_col and abs(corr_val) >= threshold:
                    high_correlations.append({
                        "column1": col,
                        "column2": other_col,
                        "correlation": round(float(corr_val), 4),
                        "strength": "strong" if abs(corr_val) >= 0.9 else "moderate",
                    })

    # Sort by absolute correlation
    high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return correlations, high_correlations


def profile_dataframe(
    df: pd.DataFrame,
    name: str = "dataset",
    compute_correlations: bool = True,
    correlation_threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Generate a complete profile for a DataFrame.

    Args:
        df: The DataFrame to profile
        name: Name for the dataset
        compute_correlations: Whether to compute correlation matrix
        correlation_threshold: Threshold for flagging high correlations

    Returns:
        Dictionary with statistics for all columns
    """
    row_count = len(df)
    column_count = len(df.columns)

    # Profile each column
    columns = {}
    constant_columns = []
    high_null_columns = []

    # Count column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col_name in df.columns:
        series = df[col_name]
        col_profile = profile_column(series, col_name)
        columns[col_name] = col_profile

        # Track constant columns
        if col_profile.get("is_constant"):
            constant_columns.append(col_name)

        # Track high null columns (>20%)
        if col_profile.get("null_percent", 0) > 20:
            high_null_columns.append(col_name)

    # Calculate overall quality metrics
    total_null_cells = df.isna().sum().sum()
    total_cells = row_count * column_count
    null_cell_percent = (
        (total_null_cells / total_cells * 100) if total_cells > 0 else 0
    )
    duplicate_rows = int(df.duplicated().sum())
    duplicate_percent = (
        (duplicate_rows / row_count * 100) if row_count > 0 else 0
    )

    # Compute correlations
    correlations = {}
    high_correlations = []
    if compute_correlations:
        correlations, high_correlations = compute_correlation_matrix(
            df, threshold=correlation_threshold
        )

    return create_table_profile(
        name=name,
        profiled_at=datetime.now(),
        row_count=row_count,
        column_count=column_count,
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        columns=columns,
        total_null_cells=int(total_null_cells),
        null_cell_percent=round(null_cell_percent, 2),
        duplicate_rows=duplicate_rows,
        duplicate_percent=round(duplicate_percent, 2),
        correlations=correlations,
        high_correlations=high_correlations,
        constant_columns=constant_columns,
        high_null_columns=high_null_columns,
        numeric_column_count=len(numeric_cols),
        categorical_column_count=len(categorical_cols),
        datetime_column_count=len(datetime_cols),
    )


class ProfileEncoder(json.JSONEncoder):
    """Custom JSON encoder for profile data."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


def save_profile(profile: dict[str, Any], filepath: Union[Path, str]) -> None:
    """
    Save a profile to a JSON file.

    Args:
        profile: Profile dictionary to save
        filepath: Path to save the profile
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(profile, f, indent=2, cls=ProfileEncoder)


def load_profile(filepath: Union[Path, str]) -> dict[str, Any]:
    """
    Load a profile from a JSON file.

    Args:
        filepath: Path to the profile file

    Returns:
        Profile dictionary
    """
    filepath = Path(filepath)

    with open(filepath) as f:
        profile = json.load(f)

    # Convert profiled_at back to datetime
    if "profiled_at" in profile and isinstance(profile["profiled_at"], str):
        profile["profiled_at"] = datetime.fromisoformat(profile["profiled_at"])

    return profile


def list_saved_profiles(profile_dir: Union[Path, str] = "profiles") -> list[dict[str, Any]]:
    """
    List all saved profiles in a directory.

    Args:
        profile_dir: Directory containing profile files

    Returns:
        List of profile metadata (name, path, profiled_at)
    """
    profile_dir = Path(profile_dir)
    profiles = []

    if not profile_dir.exists():
        return profiles

    for filepath in profile_dir.glob("*.json"):
        try:
            profile = load_profile(filepath)
            profiles.append({
                "name": profile.get("name", filepath.stem),
                "path": str(filepath),
                "profiled_at": profile.get("profiled_at"),
                "row_count": profile.get("row_count", 0),
                "column_count": profile.get("column_count", 0),
            })
        except Exception:
            continue

    # Sort by profiled_at descending
    profiles.sort(key=lambda x: x.get("profiled_at") or "", reverse=True)
    return profiles


def compare_profiles(
    current: dict[str, Any],
    baseline: dict[str, Any],
    null_threshold: float = DEFAULT_NULL_THRESHOLD,
    numeric_threshold: float = DEFAULT_NUMERIC_THRESHOLD,
) -> dict[str, Any]:
    """
    Compare two profiles and identify significant changes.

    Args:
        current: Current profile to compare
        baseline: Baseline profile to compare against
        null_threshold: Threshold for null rate changes (default 0.05)
        numeric_threshold: Threshold for numeric mean changes (default 0.10)

    Returns:
        Dictionary with a list of detected changes
        that might indicate data quality issues.
    """
    changes = []

    # Compare row counts
    if baseline["row_count"] > 0:
        row_change = (current["row_count"] - baseline["row_count"]) / baseline["row_count"]
        if abs(row_change) > 0.20:  # 20% change in row count
            changes.append({
                "type": "row_count_change",
                "column": None,
                "baseline": baseline["row_count"],
                "current": current["row_count"],
                "change_percent": round(row_change * 100, 2),
                "severity": "high" if abs(row_change) > 0.50 else "medium",
            })

    # Compare each column
    for col_name in current["columns"]:
        if col_name not in baseline["columns"]:
            changes.append({
                "type": "new_column",
                "column": col_name,
                "severity": "medium",
            })
            continue

        curr_col = current["columns"][col_name]
        base_col = baseline["columns"][col_name]

        # Null rate change
        null_change = curr_col["null_percent"] - base_col["null_percent"]
        if abs(null_change) > null_threshold * 100:
            changes.append({
                "type": "null_rate_change",
                "column": col_name,
                "baseline": base_col["null_percent"],
                "current": curr_col["null_percent"],
                "change": round(null_change, 2),
                "severity": "high" if null_change > 10 else "medium",
            })

        # Numeric changes
        if "mean" in curr_col and "mean" in base_col:
            if base_col["mean"] != 0:
                mean_change = (curr_col["mean"] - base_col["mean"]) / abs(base_col["mean"])
                if abs(mean_change) > numeric_threshold:
                    changes.append({
                        "type": "mean_change",
                        "column": col_name,
                        "baseline": base_col["mean"],
                        "current": curr_col["mean"],
                        "change_percent": round(mean_change * 100, 2),
                        "severity": "high" if abs(mean_change) > 0.30 else "medium",
                    })

            # Check for new negative values
            if base_col.get("min_value", 0) >= 0 and curr_col.get("min_value", 0) < 0:
                changes.append({
                    "type": "negative_values_appeared",
                    "column": col_name,
                    "current_min": curr_col["min_value"],
                    "severity": "high",
                })

        # Unique count changes (potential duplicates or missing data)
        if "unique_count" in curr_col and "unique_count" in base_col:
            if base_col["unique_count"] > 0:
                unique_change = (
                    (curr_col["unique_count"] - base_col["unique_count"])
                    / base_col["unique_count"]
                )
                if abs(unique_change) > 0.20:  # 20% change
                    changes.append({
                        "type": "unique_count_change",
                        "column": col_name,
                        "baseline": base_col["unique_count"],
                        "current": curr_col["unique_count"],
                        "change_percent": round(unique_change * 100, 2),
                        "severity": "medium",
                    })

        # Variance changes
        if "variance" in curr_col and "variance" in base_col:
            if base_col["variance"] > 0:
                variance_change = (curr_col["variance"] - base_col["variance"]) / base_col["variance"]
                if abs(variance_change) > 0.50:  # 50% change in variance
                    changes.append({
                        "type": "variance_change",
                        "column": col_name,
                        "baseline": base_col["variance"],
                        "current": curr_col["variance"],
                        "change_percent": round(variance_change * 100, 2),
                        "severity": "medium",
                    })

        # Median changes
        if "median" in curr_col and "median" in base_col:
            if base_col["median"] != 0:
                median_change = (curr_col["median"] - base_col["median"]) / abs(base_col["median"])
                if abs(median_change) > numeric_threshold:
                    changes.append({
                        "type": "median_change",
                        "column": col_name,
                        "baseline": base_col["median"],
                        "current": curr_col["median"],
                        "change_percent": round(median_change * 100, 2),
                        "severity": "medium",
                    })

        # Outlier rate changes
        if "outliers" in curr_col and "outliers" in base_col:
            curr_outlier_pct = curr_col["outliers"].get("percent", 0)
            base_outlier_pct = base_col["outliers"].get("percent", 0)
            outlier_diff = curr_outlier_pct - base_outlier_pct
            if outlier_diff > 5:  # Outlier rate increased by more than 5%
                changes.append({
                    "type": "outlier_increase",
                    "column": col_name,
                    "baseline": base_outlier_pct,
                    "current": curr_outlier_pct,
                    "change": round(outlier_diff, 2),
                    "severity": "high" if outlier_diff > 10 else "medium",
                })

        # String length changes
        if "length_stats" in curr_col and "length_stats" in base_col:
            curr_len = curr_col["length_stats"].get("mean", 0)
            base_len = base_col["length_stats"].get("mean", 0)
            if base_len > 0:
                len_change = (curr_len - base_len) / base_len
                if abs(len_change) > 0.20:  # 20% change in string length
                    changes.append({
                        "type": "string_length_change",
                        "column": col_name,
                        "baseline": base_len,
                        "current": curr_len,
                        "change_percent": round(len_change * 100, 2),
                        "severity": "low",
                    })

        # Whitespace issues
        if "whitespace_issues" in curr_col and "whitespace_issues" in base_col:
            curr_ws = curr_col["whitespace_issues"].get("total_with_issues", 0)
            base_ws = base_col["whitespace_issues"].get("total_with_issues", 0)
            if curr_ws > base_ws + 10:  # More than 10 new whitespace issues
                changes.append({
                    "type": "whitespace_issues_increased",
                    "column": col_name,
                    "baseline": base_ws,
                    "current": curr_ws,
                    "severity": "low",
                })

        # Column became constant
        if curr_col.get("is_constant") and not base_col.get("is_constant"):
            changes.append({
                "type": "column_became_constant",
                "column": col_name,
                "constant_value": curr_col.get("constant_value"),
                "severity": "high",
            })

    # Check for missing columns
    for col_name in baseline["columns"]:
        if col_name not in current["columns"]:
            changes.append({
                "type": "missing_column",
                "column": col_name,
                "severity": "high",
            })

    # Check duplicate rate change
    if baseline.get("duplicate_percent", 0) > 0 or current.get("duplicate_percent", 0) > 0:
        dup_diff = current.get("duplicate_percent", 0) - baseline.get("duplicate_percent", 0)
        if dup_diff > 1:  # Duplicate rate increased by more than 1%
            changes.append({
                "type": "duplicate_rate_change",
                "column": None,
                "baseline": baseline.get("duplicate_percent", 0),
                "current": current.get("duplicate_percent", 0),
                "change": round(dup_diff, 2),
                "severity": "high" if dup_diff > 5 else "medium",
            })

    return create_profile_comparison(
        baseline_name=baseline["name"],
        current_name=current["name"],
        compared_at=datetime.now(),
        changes=changes,
    )


def print_profile(profile: dict[str, Any], verbose: bool = False) -> None:
    """Pretty print a profile to console."""
    # Header
    print(f"\n{'='*80}")
    print(f"Profile: {profile['name']}")
    print(f"{'='*80}")

    # Basic info
    print(f"Rows:           {profile['row_count']:,}")
    print(f"Columns:        {profile['column_count']}")

    # Column type breakdown
    col_types = profile.get('column_types', {})
    if col_types:
        print(f"  - Numeric:    {col_types.get('numeric', 0)}")
        print(f"  - Categorical:{col_types.get('categorical', 0)}")
        print(f"  - Datetime:   {col_types.get('datetime', 0)}")

    print(f"Memory:         {profile['memory_bytes'] / 1024 / 1024:.2f} MB")
    print(f"Null Cells:     {profile['total_null_cells']:,} ({profile['null_cell_percent']:.2f}%)")
    print(f"Duplicate Rows: {profile['duplicate_rows']:,} ({profile.get('duplicate_percent', 0):.2f}%)")

    # Profiled timestamp
    profiled_at = profile['profiled_at']
    if isinstance(profiled_at, str):
        print(f"Profiled At:    {profiled_at}")
    else:
        print(f"Profiled At:    {profiled_at.strftime('%Y-%m-%d %H:%M:%S')}")

    # Warnings section
    constant_cols = profile.get('constant_columns', [])
    high_null_cols = profile.get('high_null_columns', [])
    high_corrs = profile.get('high_correlations', [])

    if constant_cols or high_null_cols:
        print(f"\n{'='*80}")
        print("Data Quality Warnings")
        print(f"{'='*80}")

        if constant_cols:
            print(f"\nConstant Columns ({len(constant_cols)}):")
            for col in constant_cols[:5]:
                print(f"  - {col}")
            if len(constant_cols) > 5:
                print(f"  ... and {len(constant_cols) - 5} more")

        if high_null_cols:
            print(f"\nHigh Null Columns ({len(high_null_cols)}):")
            for col in high_null_cols[:5]:
                null_pct = profile['columns'][col]['null_percent']
                print(f"  - {col}: {null_pct:.1f}% nulls")
            if len(high_null_cols) > 5:
                print(f"  ... and {len(high_null_cols) - 5} more")

    # High correlations
    if high_corrs:
        print(f"\n{'='*80}")
        print(f"High Correlations ({len(high_corrs)} pairs)")
        print(f"{'='*80}")
        for corr in high_corrs[:10]:
            strength = corr.get('strength', 'moderate')
            print(f"  {corr['column1']} <-> {corr['column2']}: {corr['correlation']:.3f} ({strength})")
        if len(high_corrs) > 10:
            print(f"  ... and {len(high_corrs) - 10} more pairs")

    print(f"\n{'='*80}")

    # Column details table
    print("\nColumn Statistics")
    print(f"{'-'*80}")
    print(f"{'Column':<20} {'Type':<12} {'Nulls':>15} {'Stats':<30}")
    print(f"{'-'*80}")

    for col_name, col in profile['columns'].items():
        null_str = f"{col['null_count']:,} ({col['null_percent']:.1f}%)"

        # Format stats based on column type
        if "mean" in col:
            stats = f"μ={col['mean']:.2f}, σ={col['std']:.2f}, [{col['min_value']:.2f}, {col['max_value']:.2f}]"
            # Add outlier info
            if col.get('outliers', {}).get('count', 0) > 0:
                outlier_pct = col['outliers']['percent']
                stats += f" [outliers: {outlier_pct:.1f}%]"
            # Add warning for negative min values
            if col.get('min_value', 0) < 0:
                stats = f"{stats} [NEG]"
            # Mark if possibly categorical
            if col.get('possibly_categorical'):
                stats += " [?CAT]"
        elif "unique_count" in col:
            stats = f"unique={col['unique_count']:,} ({col['unique_percent']:.1f}%)"
            # Add length stats if available
            if col.get('length_stats'):
                len_stats = col['length_stats']
                stats += f" len=[{len_stats['min']}-{len_stats['max']}]"
            # Add whitespace warning
            ws_issues = col.get('whitespace_issues', {}).get('total_with_issues', 0)
            if ws_issues > 0:
                stats += f" [WS:{ws_issues}]"
            # Pattern detected
            if col.get('detected_pattern'):
                stats += f" [{col['detected_pattern']}]"
        elif "min_date" in col:
            stats = f"{col['min_date'][:10]} to {col['max_date'][:10]}"
        else:
            stats = "-"

        # Mark constant columns
        if col.get('is_constant'):
            stats = f"[CONSTANT: {col.get('constant_value', '?')}]"

        # Add warning markers for high null rates
        if col['null_percent'] > 20:
            null_str = f"{null_str} [!]"
        elif col['null_percent'] > 5:
            null_str = f"{null_str} [?]"

        # Truncate column name if too long
        col_display = col_name[:19] if len(col_name) > 19 else col_name

        print(f"{col_display:<20} {col['dtype']:<12} {null_str:>15} {stats:<30}")

    print(f"{'-'*80}")

    # Verbose mode: show detailed stats for each column
    if verbose:
        print("\nDetailed Column Statistics")
        print(f"{'='*80}")
        for col_name, col in profile['columns'].items():
            print(f"\n{col_name}")
            print(f"  Type: {col['dtype']}")
            print(f"  Count: {col['count']:,}")
            print(f"  Nulls: {col['null_count']:,} ({col['null_percent']:.2f}%)")

            if "mean" in col:
                print(f"  Mean: {col['mean']:.4f}")
                print(f"  Std: {col['std']:.4f}")
                print(f"  Variance: {col.get('variance', 'N/A')}")
                print(f"  Median: {col.get('median', 'N/A')}")
                print(f"  Min: {col['min_value']:.4f}")
                print(f"  Max: {col['max_value']:.4f}")
                print(f"  Range: {col.get('range', 'N/A')}")
                print(f"  IQR: {col.get('iqr', 'N/A')}")
                if col.get('skewness') is not None:
                    print(f"  Skewness: {col['skewness']:.4f}")
                if col.get('kurtosis') is not None:
                    print(f"  Kurtosis: {col['kurtosis']:.4f}")
                print(f"  Zero Count: {col.get('zero_count', 0)}")
                print(f"  Negative Count: {col.get('negative_count', 0)}")
                if col.get('outliers'):
                    print(f"  Outliers: {col['outliers']['count']} ({col['outliers']['percent']:.2f}%)")

            elif "unique_count" in col:
                print(f"  Unique: {col['unique_count']} ({col['unique_percent']:.2f}%)")
                if col.get('length_stats'):
                    ls = col['length_stats']
                    print(f"  Length: mean={ls['mean']:.1f}, min={ls['min']}, max={ls['max']}")
                if col.get('whitespace_issues'):
                    ws = col['whitespace_issues']
                    print(f"  Whitespace Issues: {ws['total_with_issues']}")
                if col.get('top_values'):
                    print(f"  Top Values: {col['top_values'][:3]}")


def print_comparison(comparison: dict[str, Any]) -> None:
    if not has_significant_changes(comparison):
        print("\n No significant changes detected")
        return

    print(f"\n {len(comparison['changes'])} changes detected\n")

    # Print comparison table
    print("Profile Changes")
    print(f"{'-'*90}")
    print(f"{'Type':<25} {'Column':<15} {'Baseline':>12} {'Current':>12} {'Change':>10} {'Severity':<10}")
    print(f"{'-'*90}")

    for change in comparison['changes']:
        severity = change.get("severity", "low").upper()

        change_val = ""
        if "change" in change or "change_percent" in change:
            change_val = str(change.get("change_percent", change.get("change", "-"))) + "%"
        else:
            change_val = "-"

        print(
            f"{change['type']:<25} "
            f"{change.get('column', '-'):<15} "
            f"{str(change.get('baseline', '-')):>12} "
            f"{str(change.get('current', '-')):>12} "
            f"{change_val:>10} "
            f"{severity:<10}"
        )

    print(f"{'-'*90}")


def detect_and_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically detect and parse date columns."""
    for col in df.columns:
        # Check if column name suggests it's a date
        if any(date_keyword in col.lower() for date_keyword in ['date', 'time', 'timestamp', 'created', 'updated']):
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass  # If conversion fails, leave as is
    return df


def find_csv_files(data_dir: Path) -> list[Path]:
    """Find all CSV files in the data directory."""
    csv_files = list(data_dir.glob("*.csv"))
    return sorted(csv_files)


def main():
    """Profile all CSV files in the data directory."""
    data_dir = Path("data")
    profile_dir = Path("profiles")

    print("\n" + "=" * 60)
    print("        Enhanced Data Profiler")
    print("=" * 60)

    if not data_dir.exists():
        print(f"\nError: Directory '{data_dir}' does not exist")
        return

    # Find all CSV files
    csv_files = find_csv_files(data_dir)

    if not csv_files:
        print(f"\nNo CSV files found in '{data_dir}'")
        print("Please add CSV files to profile or run data generator first:")
        print("  python -m src.data_generator")
        return

    print(f"\nFound {len(csv_files)} CSV file(s) to profile:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")

    # Profile each dataset
    all_profiles = []

    for filepath in csv_files:
        print(f"\nLoading {filepath.name}...")

        try:
            df = pd.read_csv(filepath)

            # Automatically detect and parse date columns
            df = detect_and_parse_dates(df)

            profile = profile_dataframe(df, filepath.stem)
            print_profile(profile)
            print()

            # Save profile to JSON
            profile_path = profile_dir / f"{filepath.stem}_profile.json"
            save_profile(profile, profile_path)
            print(f"  Profile saved to: {profile_path}")

            all_profiles.append((filepath.stem, profile))

        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Data Quality Summary
    if all_profiles:
        print("\n" + "=" * 60)
        print("Data Quality Summary")
        print("=" * 60)

        total_issues = 0

        for dataset_name, profile in all_profiles:
            issues = []

            # Check for high null percentages
            if profile['null_cell_percent'] > 5:
                issues.append(f"High null rate: {profile['null_cell_percent']:.1f}%")

            # Check for duplicates
            if profile['duplicate_rows'] > 0:
                dup_pct = profile.get('duplicate_percent', 0)
                issues.append(f"{profile['duplicate_rows']:,} duplicate rows ({dup_pct:.1f}%)")

            # Check for constant columns
            const_cols = profile.get('constant_columns', [])
            if const_cols:
                issues.append(f"{len(const_cols)} constant column(s): {', '.join(const_cols[:3])}")

            # Check for high correlations
            high_corrs = profile.get('high_correlations', [])
            if high_corrs:
                issues.append(f"{len(high_corrs)} highly correlated column pair(s)")

            # Check columns for issues
            for col_name, col in profile['columns'].items():
                if col['null_percent'] > 20:
                    issues.append(f"{col_name}: {col['null_percent']:.1f}% nulls")

                if 'min_value' in col and col['min_value'] < 0:
                    neg_pct = col.get('negative_percent', 0)
                    if neg_pct > 0:
                        issues.append(f"{col_name}: {neg_pct:.1f}% negative values")

                # Outlier warnings
                if col.get('outliers', {}).get('percent', 0) > 5:
                    issues.append(f"{col_name}: {col['outliers']['percent']:.1f}% outliers")

                # Whitespace issues
                ws_issues = col.get('whitespace_issues', {}).get('total_with_issues', 0)
                if ws_issues > 10:
                    issues.append(f"{col_name}: {ws_issues} values with whitespace issues")

                # Possibly miscategorized columns
                if col.get('possibly_numeric'):
                    issues.append(f"{col_name}: possibly numeric (stored as string)")

            if issues:
                total_issues += len(issues)
                print(f"\n{dataset_name}:")
                for issue in issues:
                    print(f"  - {issue}")

        if total_issues == 0:
            print("\nNo significant data quality issues detected!")
        else:
            print(f"\n{total_issues} total issue(s) found across {len(all_profiles)} dataset(s)")

        # Print saved profiles info
        print(f"\nProfiles saved to: {profile_dir}/")
        print("Use load_profile() to load and compare profiles over time.")


if __name__ == "__main__":
    main()
