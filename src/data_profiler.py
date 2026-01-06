"""
Data profiler for computing statistical profiles of datasets.

This module provides tools to analyze datasets and compute comprehensive
statistical profiles that serve as baselines for anomaly detection.

Run with: python -m src.profiler
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class ColumnProfile:
    """Profile statistics for a single column."""
    
    name: str
    dtype: str
    count: int
    null_count: int
    null_percent: float
    
    # Numeric stats (optional)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    percentile_25: Optional[float] = None
    percentile_50: Optional[float] = None
    percentile_75: Optional[float] = None
    
    # String stats (optional)
    unique_count: Optional[int] = None
    unique_percent: Optional[float] = None
    top_values: Optional[list[tuple[Any, int]]] = None
    avg_length: Optional[float] = None
    
    # Date stats (optional)
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TableProfile:
    """Profile for an entire table/dataset."""
    
    name: str
    profiled_at: datetime
    row_count: int
    column_count: int
    memory_bytes: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    
    # Quality indicators
    total_null_cells: int = 0
    null_cell_percent: float = 0.0
    duplicate_rows: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "profiled_at": self.profiled_at.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_bytes": self.memory_bytes,
            "total_null_cells": self.total_null_cells,
            "null_cell_percent": self.null_cell_percent,
            "duplicate_rows": self.duplicate_rows,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }


@dataclass
class ProfileComparison:
    """Result of comparing two profiles."""
    
    baseline_name: str
    current_name: str
    compared_at: datetime
    changes: list[dict[str, Any]] = field(default_factory=list)
    
    @property
    def has_significant_changes(self) -> bool:
        return len(self.changes) > 0


class DataProfiler:
    """
    Profiles datasets to compute statistical summaries.
    
    This profiler analyzes DataFrames and computes comprehensive statistics
    for each column, which can then be used as baselines for detecting
    data quality anomalies.
    
    Usage:
        profiler = DataProfiler()
        profile = profiler.profile(df, "orders")
        profiler.print_profile(profile)
        
        # Compare to baseline
        changes = profiler.compare_profiles(profile, baseline_profile)
    """
    
    def __init__(
        self,
        null_threshold: float = 0.05,  # Alert if null% changes by more than 5%
        numeric_threshold: float = 0.10,  # Alert if mean changes by more than 10%
    ):
        self.null_threshold = null_threshold
        self.numeric_threshold = numeric_threshold
    
    def profile(self, df: pd.DataFrame, name: str = "dataset") -> TableProfile:
        """
        Generate a complete profile for a DataFrame.
        
        Args:
            df: The DataFrame to profile
            name: Name for the dataset
        
        Returns:
            TableProfile with statistics for all columns
        """
        profile = TableProfile(
            name=name,
            profiled_at=datetime.now(),
            row_count=len(df),
            column_count=len(df.columns),
            memory_bytes=df.memory_usage(deep=True).sum(),
        )
        
        # Profile each column
        for col_name in df.columns:
            series = df[col_name]
            col_profile = self._profile_column(series, col_name)
            profile.columns[col_name] = col_profile
        
        # Calculate overall quality metrics
        profile.total_null_cells = df.isna().sum().sum()
        total_cells = profile.row_count * profile.column_count
        profile.null_cell_percent = (
            (profile.total_null_cells / total_cells * 100) if total_cells > 0 else 0
        )
        profile.duplicate_rows = df.duplicated().sum()
        
        return profile
    
    def _profile_column(self, series: pd.Series, name: str) -> ColumnProfile:
        """Profile a single column based on its type."""
        dtype_str = str(series.dtype)
        count = len(series)
        null_count = series.isna().sum()
        null_percent = (null_count / count * 100) if count > 0 else 0
        
        base_profile = ColumnProfile(
            name=name,
            dtype=dtype_str,
            count=count,
            null_count=null_count,
            null_percent=round(null_percent, 2),
        )
        
        # Determine column type and compute appropriate stats
        if pd.api.types.is_numeric_dtype(series):
            return self._add_numeric_stats(base_profile, series)
        elif pd.api.types.is_datetime64_any_dtype(series):
            return self._add_datetime_stats(base_profile, series)
        else:
            # Treat as string/categorical
            return self._add_string_stats(base_profile, series)
    
    def _add_numeric_stats(
        self, profile: ColumnProfile, series: pd.Series
    ) -> ColumnProfile:
        """Add numeric statistics to a column profile."""
        non_null = series.dropna()
        
        if len(non_null) > 0:
            profile.mean = round(float(non_null.mean()), 4)
            profile.std = round(float(non_null.std()), 4)
            profile.min_value = round(float(non_null.min()), 4)
            profile.max_value = round(float(non_null.max()), 4)
            
            percentiles = non_null.quantile([0.25, 0.5, 0.75])
            profile.percentile_25 = round(float(percentiles[0.25]), 4)
            profile.percentile_50 = round(float(percentiles[0.5]), 4)
            profile.percentile_75 = round(float(percentiles[0.75]), 4)
        
        return profile
    
    def _add_string_stats(
        self, profile: ColumnProfile, series: pd.Series
    ) -> ColumnProfile:
        """Add string/categorical statistics to a column profile."""
        non_null = series.dropna()
        
        if len(non_null) > 0:
            profile.unique_count = non_null.nunique()
            profile.unique_percent = round(
                profile.unique_count / len(non_null) * 100, 2
            )
            
            # Top 5 most common values
            value_counts = non_null.value_counts().head(5)
            profile.top_values = list(zip(
                value_counts.index.tolist(),
                value_counts.values.tolist()
            ))
            
            # Average length for strings
            if series.dtype == "object":
                lengths = non_null.astype(str).str.len()
                profile.avg_length = round(float(lengths.mean()), 2)
        
        return profile
    
    def _add_datetime_stats(
        self, profile: ColumnProfile, series: pd.Series
    ) -> ColumnProfile:
        """Add datetime statistics to a column profile."""
        non_null = series.dropna()
        
        if len(non_null) > 0:
            min_date = non_null.min()
            max_date = non_null.max()
            
            profile.min_date = str(min_date)
            profile.max_date = str(max_date)
            profile.date_range_days = (max_date - min_date).days
        
        return profile
    
    def compare_profiles(
        self,
        current: TableProfile,
        baseline: TableProfile,
    ) -> ProfileComparison:
        """
        Compare two profiles and identify significant changes.
        
        Returns a comparison object with a list of detected changes
        that might indicate data quality issues.
        """
        comparison = ProfileComparison(
            baseline_name=baseline.name,
            current_name=current.name,
            compared_at=datetime.now(),
        )
        
        # Compare row counts
        if baseline.row_count > 0:
            row_change = (current.row_count - baseline.row_count) / baseline.row_count
            if abs(row_change) > 0.20:  # 20% change in row count
                comparison.changes.append({
                    "type": "row_count_change",
                    "column": None,
                    "baseline": baseline.row_count,
                    "current": current.row_count,
                    "change_percent": round(row_change * 100, 2),
                    "severity": "high" if abs(row_change) > 0.50 else "medium",
                })
        
        # Compare each column
        for col_name in current.columns:
            if col_name not in baseline.columns:
                comparison.changes.append({
                    "type": "new_column",
                    "column": col_name,
                    "severity": "medium",
                })
                continue
            
            curr_col = current.columns[col_name]
            base_col = baseline.columns[col_name]
            
            # Null rate change
            null_change = curr_col.null_percent - base_col.null_percent
            if abs(null_change) > self.null_threshold * 100:
                comparison.changes.append({
                    "type": "null_rate_change",
                    "column": col_name,
                    "baseline": base_col.null_percent,
                    "current": curr_col.null_percent,
                    "change": round(null_change, 2),
                    "severity": "high" if null_change > 10 else "medium",
                })
            
            # Numeric changes
            if curr_col.mean is not None and base_col.mean is not None:
                if base_col.mean != 0:
                    mean_change = (curr_col.mean - base_col.mean) / abs(base_col.mean)
                    if abs(mean_change) > self.numeric_threshold:
                        comparison.changes.append({
                            "type": "mean_change",
                            "column": col_name,
                            "baseline": base_col.mean,
                            "current": curr_col.mean,
                            "change_percent": round(mean_change * 100, 2),
                            "severity": "high" if abs(mean_change) > 0.30 else "medium",
                        })
                
                # Check for new negative values
                if base_col.min_value >= 0 and curr_col.min_value < 0:
                    comparison.changes.append({
                        "type": "negative_values_appeared",
                        "column": col_name,
                        "current_min": curr_col.min_value,
                        "severity": "high",
                    })
            
            # Unique count changes (potential duplicates or missing data)
            if curr_col.unique_count is not None and base_col.unique_count is not None:
                if base_col.unique_count > 0:
                    unique_change = (
                        (curr_col.unique_count - base_col.unique_count) 
                        / base_col.unique_count
                    )
                    if abs(unique_change) > 0.20:  # 20% change
                        comparison.changes.append({
                            "type": "unique_count_change",
                            "column": col_name,
                            "baseline": base_col.unique_count,
                            "current": curr_col.unique_count,
                            "change_percent": round(unique_change * 100, 2),
                            "severity": "medium",
                        })
        
        # Check for missing columns
        for col_name in baseline.columns:
            if col_name not in current.columns:
                comparison.changes.append({
                    "type": "missing_column",
                    "column": col_name,
                    "severity": "high",
                })
        
        return comparison
    
    def print_profile(self, profile: TableProfile) -> None:
        """Pretty print a profile to console using rich."""
        # Header panel
        header = Table.grid(padding=1)
        header.add_column(style="cyan", justify="right")
        header.add_column(style="green")
        
        header.add_row("Rows:", f"{profile.row_count:,}")
        header.add_row("Columns:", str(profile.column_count))
        header.add_row("Memory:", f"{profile.memory_bytes / 1024 / 1024:.2f} MB")
        header.add_row("Null Cells:", f"{profile.total_null_cells:,} ({profile.null_cell_percent:.2f}%)")
        header.add_row("Duplicate Rows:", f"{profile.duplicate_rows:,}")
        header.add_row("Profiled At:", profile.profiled_at.strftime("%Y-%m-%d %H:%M:%S"))
        
        console.print(Panel(header, title=f"[bold]Profile: {profile.name}[/bold]", border_style="blue"))
        
        # Column details table
        table = Table(title="Column Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="dim")
        table.add_column("Nulls", justify="right")
        table.add_column("Stats", style="green")
        
        for col_name, col in profile.columns.items():
            null_str = f"{col.null_count:,} ({col.null_percent:.1f}%)"
            
            # Format stats based on column type
            if col.mean is not None:
                stats = f"μ={col.mean:.2f}, σ={col.std:.2f}, [{col.min_value:.2f}, {col.max_value:.2f}]"
                # Highlight negative min values
                if col.min_value < 0:
                    stats = f"[red]{stats}[/red]"
            elif col.unique_count is not None:
                stats = f"unique={col.unique_count:,} ({col.unique_percent:.1f}%)"
            elif col.min_date is not None:
                stats = f"{col.min_date[:10]} to {col.max_date[:10]}"
            else:
                stats = "-"
            
            # Highlight high null rates
            if col.null_percent > 5:
                null_str = f"[yellow]{null_str}[/yellow]"
            if col.null_percent > 20:
                null_str = f"[red]{null_str}[/red]"
            
            table.add_row(col_name, col.dtype, null_str, stats)
        
        console.print(table)
    
    def print_comparison(self, comparison: ProfileComparison) -> None:
        """Pretty print a profile comparison."""
        if not comparison.has_significant_changes:
            console.print("[green]✓ No significant changes detected[/green]")
            return
        
        console.print(f"\n[bold yellow]⚠ {len(comparison.changes)} changes detected[/bold yellow]\n")
        
        table = Table(title="Profile Changes", show_header=True)
        table.add_column("Type", style="cyan")
        table.add_column("Column")
        table.add_column("Baseline", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Change", justify="right")
        table.add_column("Severity")
        
        for change in comparison.changes:
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
    
    profiler = DataProfiler()
    
    # Profile each dataset
    datasets = ["orders.csv", "order_items.csv", "products.csv"]
    
    for filename in datasets:
        filepath = data_dir / filename
        console.print(f"\n[dim]Loading {filename}...[/dim]")
        
        df = pd.read_csv(filepath)
        
        # Parse dates if present
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"])
        
        profile = profiler.profile(df, filename.replace(".csv", ""))
        profiler.print_profile(profile)
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
