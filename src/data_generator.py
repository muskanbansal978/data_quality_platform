"""
Sample data generator for the Data Quality Platform.

Generates realistic e-commerce data with intentional anomalies
that we'll detect in later steps.

Run with: python -m src.data_generator
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from faker import Faker
from rich.console import Console
from rich.table import Table

fake = Faker()
console = Console()


class EcommerceDataGenerator:
    """
    Generates fake e-commerce data with configurable anomalies.
    
    This creates realistic order data that mimics what you'd see
    in a real e-commerce platform, with intentional data quality
    issues injected for testing.
    """
    
    CATEGORIES = ["Electronics", "Clothing", "Home", "Books", "Sports"]
    ORDER_STATUSES = ["pending", "completed", "cancelled", "refunded"]
    STATUS_WEIGHTS = [0.1, 0.75, 0.1, 0.05]  # Most orders complete
    
    def __init__(self, seed: int = 42):
        """Initialize with a seed for reproducibility."""
        random.seed(seed)
        Faker.seed(seed)
        self.products = self._generate_products(100)
        self._anomalies_injected: dict[int, str] = {}
    
    def _generate_products(self, n: int) -> pd.DataFrame:
        """Generate a product catalog."""
        products = []
        for i in range(n):
            category = random.choice(self.CATEGORIES)
            # Price ranges vary by category
            price_ranges = {
                "Electronics": (50, 1000),
                "Clothing": (20, 200),
                "Home": (30, 500),
                "Books": (10, 50),
                "Sports": (25, 300),
            }
            min_price, max_price = price_ranges[category]
            
            products.append({
                "product_id": f"PROD_{i:04d}",
                "name": fake.catch_phrase(),
                "category": category,
                "base_price": round(random.uniform(min_price, max_price), 2),
            })
        
        return pd.DataFrame(products)
    
    def _get_daily_order_count(
        self, 
        day: int, 
        base_count: int,
        anomaly_config: Optional[dict] = None,
    ) -> int:
        """
        Calculate number of orders for a given day.
        
        Applies natural variation (weekday/weekend patterns)
        plus any configured anomalies.
        """
        # Natural weekly pattern: weekends have 20% more orders
        day_of_week = day % 7
        is_weekend = day_of_week in [5, 6]
        multiplier = 1.2 if is_weekend else 1.0
        
        # Add some random noise (±10%)
        noise = random.uniform(0.9, 1.1)
        count = int(base_count * multiplier * noise)
        
        # Check for volume anomaly
        if anomaly_config and day in anomaly_config:
            if anomaly_config[day] == "volume_drop":
                count = int(count * 0.5)  # 50% drop
                self._anomalies_injected[day] = "volume_drop"
        
        return count
    
    def _get_category_weights(
        self,
        day: int,
        anomaly_config: Optional[dict] = None,
    ) -> list[float]:
        """Get category selection weights, possibly shifted."""
        # Normal distribution across categories
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # Electronics, Clothing, Home, Books, Sports
        
        if anomaly_config and day in anomaly_config:
            if anomaly_config[day] == "category_shift":
                # Suddenly Electronics dominates
                weights = [0.60, 0.15, 0.10, 0.10, 0.05]
                self._anomalies_injected[day] = "category_shift"
        
        return weights
    
    def generate_orders(
        self,
        start_date: datetime,
        days: int = 30,
        orders_per_day: int = 1000,
        anomaly_config: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate orders and order items for the specified period.
        
        Args:
            start_date: First day of data generation
            days: Number of days to generate
            orders_per_day: Base number of orders per day
            anomaly_config: Dict mapping day number to anomaly type
                Supported anomalies:
                - "volume_drop": 50% fewer orders
                - "negative_prices": 5% of prices become negative
                - "null_customers": 20% missing customer IDs
                - "category_shift": Electronics becomes dominant
        
        Returns:
            Tuple of (orders_df, order_items_df)
        """
        orders = []
        order_items = []
        order_counter = 0
        item_counter = 0
        
        console.print(f"\n[bold]Generating {days} days of data...[/bold]\n")
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            daily_order_count = self._get_daily_order_count(
                day, orders_per_day, anomaly_config
            )
            category_weights = self._get_category_weights(day, anomaly_config)
            
            # Check for other anomalies on this day
            inject_negative_prices = (
                anomaly_config 
                and day in anomaly_config 
                and anomaly_config[day] == "negative_prices"
            )
            inject_null_customers = (
                anomaly_config 
                and day in anomaly_config 
                and anomaly_config[day] == "null_customers"
            )
            
            if inject_negative_prices:
                self._anomalies_injected[day] = "negative_prices"
            if inject_null_customers:
                self._anomalies_injected[day] = "null_customers"
            
            for _ in range(daily_order_count):
                order_counter += 1
                order_id = f"ORD_{order_counter:06d}"
                
                # Customer ID (possibly null)
                if inject_null_customers and random.random() < 0.20:
                    customer_id = None
                else:
                    customer_id = f"CUST_{random.randint(1, 10000):05d}"
                
                # Random time during the day
                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
                order_time = current_date.replace(hour=hour, minute=minute)
                
                # Generate 1-5 items per order
                num_items = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
                order_total = 0.0
                
                # Select products based on category weights
                for _ in range(num_items):
                    item_counter += 1
                    
                    # Pick category first, then product from that category
                    category = random.choices(self.CATEGORIES, weights=category_weights)[0]
                    category_products = self.products[self.products["category"] == category]
                    product = category_products.sample(1).iloc[0]
                    
                    quantity = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
                    
                    # Price with some variation from base
                    price = product["base_price"] * random.uniform(0.9, 1.1)
                    
                    # Inject negative price anomaly
                    if inject_negative_prices and random.random() < 0.05:
                        price = -abs(price)  # Make it negative
                    
                    price = round(price, 2)
                    order_total += price * quantity
                    
                    order_items.append({
                        "item_id": f"ITEM_{item_counter:07d}",
                        "order_id": order_id,
                        "product_id": product["product_id"],
                        "quantity": quantity,
                        "unit_price": price,
                        "line_total": round(price * quantity, 2),
                    })
                
                orders.append({
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "order_date": order_time.isoformat(),
                    "total_amount": round(order_total, 2),
                    "status": random.choices(self.ORDER_STATUSES, weights=self.STATUS_WEIGHTS)[0],
                    "item_count": num_items,
                })
        
        return pd.DataFrame(orders), pd.DataFrame(order_items)
    
    def get_anomaly_summary(self) -> dict[int, str]:
        """Return summary of injected anomalies."""
        return self._anomalies_injected.copy()


def main():
    """Generate sample data and save to CSV files."""
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    console.print("[bold blue]╔══════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║   Data Quality Platform Generator    ║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════╝[/bold blue]")
    
    generator = EcommerceDataGenerator(seed=42)
    
    # Configure anomalies - these are the issues we'll detect later
    anomalies = {
        10: "volume_drop",      # Day 10: 50% fewer orders
        15: "negative_prices",  # Day 15: Some negative prices
        20: "null_customers",   # Day 20: Missing customer IDs
        25: "category_shift",   # Day 25: Category distribution changes
    }
    
    start_date = datetime(2024, 1, 1)
    orders, items = generator.generate_orders(
        start_date=start_date,
        days=30,
        orders_per_day=1000,
        anomaly_config=anomalies,
    )
    
    # Save to CSV
    orders.to_csv(data_dir / "orders.csv", index=False)
    items.to_csv(data_dir / "order_items.csv", index=False)
    generator.products.to_csv(data_dir / "products.csv", index=False)
    
    # Print summary
    console.print("\n[bold green] Data generated successfully![/bold green]\n")
    
    # Summary table
    table = Table(title="Generated Data Summary")
    table.add_column("Dataset", style="cyan")
    table.add_column("Records", style="green", justify="right")
    table.add_column("File", style="yellow")
    
    table.add_row("Orders", f"{len(orders):,}", "data/orders.csv")
    table.add_row("Order Items", f"{len(items):,}", "data/order_items.csv")
    table.add_row("Products", f"{len(generator.products):,}", "data/products.csv")
    
    console.print(table)
    
    # Anomaly summary
    console.print("\n[bold yellow]Injected Anomalies:[/bold yellow]")
    for day, anomaly_type in sorted(generator.get_anomaly_summary().items()):
        date = start_date + timedelta(days=day)
        console.print(f"  • Day {day} ({date.strftime('%Y-%m-%d')}): [red]{anomaly_type}[/red]")
    
    # Quick stats
    console.print("\n[bold]Quick Stats:[/bold]")
    console.print(f"  • Date range: {start_date.date()} to {(start_date + timedelta(days=29)).date()}")
    console.print(f"  • Avg orders/day: {len(orders) / 30:.0f}")
    console.print(f"  • Null customer_ids: {orders['customer_id'].isna().sum()}")
    console.print(f"  • Negative prices: {(items['unit_price'] < 0).sum()}")
    
    console.print("\n[dim]Next step: Run the profiler with 'python -m src.profiler'[/dim]")


if __name__ == "__main__":
    main()
