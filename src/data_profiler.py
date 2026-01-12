"""
Data profiler for computing statistical profiles of datasets.

This module provides tools to analyze datasets and compute comprehensive
statistical profiles that serve as baselines for anomaly detection.

Run with: python -m src.profiler
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Default thresholds
DEFAULT_NULL_THRESHOLD = 0.05  # Alert if null% changes by more than 5%
DEFAULT_NUMERIC_THRESHOLD = 0.10  # Alert if mean changes by more than 10%


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


def add_numeric_stats(profile: dict[str, Any], series: pd.Series) -> dict[str, Any]:
    """Add numeric statistics to a column profile."""
    non_null = series.dropna()

    if len(non_null) > 0:
        percentiles = non_null.quantile([0.25, 0.5, 0.75])

        profile.update({
            "mean": round(float(non_null.mean()), 4),
            "std": round(float(non_null.std()), 4),
            "min_value": round(float(non_null.min()), 4),
            "max_value": round(float(non_null.max()), 4),
            "percentile_25": round(float(percentiles[0.25]), 4),
            "percentile_50": round(float(percentiles[0.5]), 4),
            "percentile_75": round(float(percentiles[0.75]), 4),
        })

    return profile


def add_string_stats(profile: dict[str, Any], series: pd.Series) -> dict[str, Any]:
    """Add string/categorical statistics to a column profile."""
    non_null = series.dropna()

    if len(non_null) > 0:
        unique_count = non_null.nunique()

        # Top 5 most common values
        value_counts = non_null.value_counts().head(5)
        top_values = list(zip(
            value_counts.index.tolist(),
            value_counts.values.tolist()
        ))

        profile.update({
            "unique_count": unique_count,
            "unique_percent": round(unique_count / len(non_null) * 100, 2),
            "top_values": top_values,
        })

        # Average length for strings
        if series.dtype == "object":
            lengths = non_null.astype(str).str.len()
            profile["avg_length"] = round(float(lengths.mean()), 2)

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


def profile_dataframe(df: pd.DataFrame, name: str = "dataset") -> dict[str, Any]:
    """
    Generate a complete profile for a DataFrame.

    Args:
        df: The DataFrame to profile
        name: Name for the dataset

    Returns:
        Dictionary with statistics for all columns
    """
    row_count = len(df)
    column_count = len(df.columns)

    # Profile each column
    columns = {}
    for col_name in df.columns:
        series = df[col_name]
        col_profile = profile_column(series, col_name)
        columns[col_name] = col_profile

    # Calculate overall quality metrics
    total_null_cells = df.isna().sum().sum()
    total_cells = row_count * column_count
    null_cell_percent = (
        (total_null_cells / total_cells * 100) if total_cells > 0 else 0
    )
    duplicate_rows = df.duplicated().sum()

    return create_table_profile(
        name=name,
        profiled_at=datetime.now(),
        row_count=row_count,
        column_count=column_count,
        memory_bytes=df.memory_usage(deep=True).sum(),
        columns=columns,
        total_null_cells=total_null_cells,
        null_cell_percent=null_cell_percent,
        duplicate_rows=duplicate_rows,
    )


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

    # Check for missing columns
    for col_name in baseline["columns"]:
        if col_name not in current["columns"]:
            changes.append({
                "type": "missing_column",
                "column": col_name,
                "severity": "high",
            })

    return create_profile_comparison(
        baseline_name=baseline["name"],
        current_name=current["name"],
        compared_at=datetime.now(),
        changes=changes,
    )


def print_profile(profile: dict[str, Any]) -> None:
    """Pretty print a profile to console using rich."""
    # Header panel
    header = Table.grid(padding=1)
    header.add_column(style="cyan", justify="right")
    header.add_column(style="green")

    header.add_row("Rows:", f"{profile['row_count']:,}")
    header.add_row("Columns:", str(profile['column_count']))
    header.add_row("Memory:", f"{profile['memory_bytes'] / 1024 / 1024:.2f} MB")
    header.add_row("Null Cells:", f"{profile['total_null_cells']:,} ({profile['null_cell_percent']:.2f}%)")
    header.add_row("Duplicate Rows:", f"{profile['duplicate_rows']:,}")
    header.add_row("Profiled At:", profile['profiled_at'].strftime("%Y-%m-%d %H:%M:%S"))

    console.print(Panel(header, title=f"[bold]Profile: {profile['name']}[/bold]", border_style="blue"))

    # Column details table
    table = Table(title="Column Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Nulls", justify="right")
    table.add_column("Stats", style="green")

    for col_name, col in profile['columns'].items():
        null_str = f"{col['null_count']:,} ({col['null_percent']:.1f}%)"

        # Format stats based on column type
        if "mean" in col:
            stats = f"μ={col['mean']:.2f}, σ={col['std']:.2f}, [{col['min_value']:.2f}, {col['max_value']:.2f}]"
            # Highlight negative min values
            if col['min_value'] < 0:
                stats = f"[red]{stats}[/red]"
        elif "unique_count" in col:
            stats = f"unique={col['unique_count']:,} ({col['unique_percent']:.1f}%)"
        elif "min_date" in col:
            stats = f"{col['min_date'][:10]} to {col['max_date'][:10]}"
        else:
            stats = "-"

        # Highlight high null rates
        if col['null_percent'] > 5:
            null_str = f"[yellow]{null_str}[/yellow]"
        if col['null_percent'] > 20:
            null_str = f"[red]{null_str}[/red]"

        table.add_row(col_name, col['dtype'], null_str, stats)

    console.print(table)


def print_comparison(comparison: dict[str, Any]) -> None:
    """Pretty print a profile comparison."""
    if not has_significant_changes(comparison):
        console.print("[green]✓ No significant changes detected[/green]")
        return

    console.print(f"\n[bold yellow]⚠ {len(comparison['changes'])} changes detected[/bold yellow]\n")

    table = Table(title="Profile Changes", show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Column")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Severity")

    for change in comparison['changes']:
        severity_style = {
            "high": "[red]HIGH[/red]",
            "medium": "[yellow]MEDIUM[/yellow]",
            "low": "[green]LOW[/green]",
        }.get(change.get("severity", "low"), change.get("severity", ""))

        table.add_row(
            change["type"],
            change.get("column", "-"),
            str(change.get("baseline", "-")),
            str(change.get("current", "-")),
            str(change.get("change_percent", change.get("change", "-"))) + "%"
                if "change" in change or "change_percent" in change else "-",
            severity_style,
        )

    console.print(table)


def main():
    """Profile the generated sample data."""
    data_dir = Path("data")

    console.print("\n[bold blue]╔══════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║        Data Profiler                 ║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════╝[/bold blue]\n")

    if not (data_dir / "orders.csv").exists():
        console.print("[red]Error: No data found. Run data_generator first:[/red]")
        console.print("  python -m src.data_generator")
        return

    # Profile each dataset
    datasets = ["orders.csv", "order_items.csv", "products.csv"]

    for filename in datasets:
        filepath = data_dir / filename
        console.print(f"\n[dim]Loading {filename}...[/dim]")

        df = pd.read_csv(filepath)

        # Parse dates if present
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"])

        profile = profile_dataframe(df, filename.replace(".csv", ""))
        print_profile(profile)
        console.print()

    # Highlight detected issues
    console.print("[bold]═══ Data Quality Summary ═══[/bold]\n")

    # Check orders
    orders = pd.read_csv(data_dir / "orders.csv")
    null_customers = orders["customer_id"].isna().sum()
    if null_customers > 0:
        console.print(f"[yellow]⚠ Found {null_customers} orders with null customer_id[/yellow]")

    # Check order_items
    items = pd.read_csv(data_dir / "order_items.csv")
    negative_prices = (items["unit_price"] < 0).sum()
    if negative_prices > 0:
        console.print(f"[red]✗ Found {negative_prices} items with negative prices[/red]")

    console.print("\n[dim]Next step: Store profiles in database with 'python -m src.profile_store'[/dim]")


if __name__ == "__main__":
    main()
