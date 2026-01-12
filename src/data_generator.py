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

fake = Faker()

# Constants
CATEGORIES = ["Electronics", "Clothing", "Home", "Books", "Sports"]
ORDER_STATUSES = ["pending", "completed", "cancelled", "refunded"]
STATUS_WEIGHTS = [0.1, 0.75, 0.1, 0.05]  # Most orders complete

PRICE_RANGES = {
    "Electronics": (50, 1000),
    "Clothing": (20, 200),
    "Home": (30, 500),
    "Books": (10, 50),
    "Sports": (25, 300),
}


def initialize_random_seed(seed: int = 42) -> None:
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    Faker.seed(seed)


def generate_products(n: int = 100) -> pd.DataFrame:
    """Generate a product catalog."""
    products = []
    for i in range(n):
        category = random.choice(CATEGORIES)
        min_price, max_price = PRICE_RANGES[category]

        products.append({
            "product_id": f"PROD_{i:04d}",
            "name": fake.catch_phrase(),
            "category": category,
            "base_price": round(random.uniform(min_price, max_price), 2),
        })

    return pd.DataFrame(products)


def get_daily_order_count(
    day: int,
    base_count: int,
    anomaly_config: Optional[dict] = None,
    anomalies_injected: Optional[dict] = None,
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
            if anomalies_injected is not None:
                anomalies_injected[day] = "volume_drop"

    return count


def get_category_weights(
    day: int,
    anomaly_config: Optional[dict] = None,
    anomalies_injected: Optional[dict] = None,
) -> list[float]:
    """Get category selection weights, possibly shifted."""
    # Normal distribution across categories
    weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # Electronics, Clothing, Home, Books, Sports

    if anomaly_config and day in anomaly_config:
        if anomaly_config[day] == "category_shift":
            # Suddenly Electronics dominates
            weights = [0.60, 0.15, 0.10, 0.10, 0.05]
            if anomalies_injected is not None:
                anomalies_injected[day] = "category_shift"

    return weights


def generate_orders(
    products: pd.DataFrame,
    start_date: datetime,
    days: int = 30,
    orders_per_day: int = 1000,
    anomaly_config: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """
    Generate orders and order items for the specified period.

    Args:
        products: DataFrame of products to use for order items
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
        Tuple of (orders_df, order_items_df, anomalies_injected)
    """
    orders = []
    order_items = []
    order_counter = 0
    item_counter = 0
    anomalies_injected: dict[int, str] = {}

    print(f"\nGenerating {days} days of data...\n")

    for day in range(days):
        current_date = start_date + timedelta(days=day)
        daily_order_count = get_daily_order_count(
            day, orders_per_day, anomaly_config, anomalies_injected
        )
        category_weights = get_category_weights(day, anomaly_config, anomalies_injected)

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
            anomalies_injected[day] = "negative_prices"
        if inject_null_customers:
            anomalies_injected[day] = "null_customers"

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
                category = random.choices(CATEGORIES, weights=category_weights)[0]
                category_products = products[products["category"] == category]
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
                "status": random.choices(ORDER_STATUSES, weights=STATUS_WEIGHTS)[0],
                "item_count": num_items,
            })

    return pd.DataFrame(orders), pd.DataFrame(order_items), anomalies_injected


def print_summary(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    anomalies_injected: dict[int, str],
    start_date: datetime,
) -> None:
    """Print a summary of generated data."""
    print("\n✓ Data generated successfully!\n")

    # Summary table
    print("Generated Data Summary")
    print("=" * 60)
    print(f"{'Dataset':<20} {'Records':>15} {'File':<25}")
    print("-" * 60)
    print(f"{'Orders':<20} {len(orders):>15,} {'data/orders.csv':<25}")
    print(f"{'Order Items':<20} {len(items):>15,} {'data/order_items.csv':<25}")
    print(f"{'Products':<20} {len(products):>15,} {'data/products.csv':<25}")
    print("=" * 60)

    # Anomaly summary
    print("\nInjected Anomalies:")
    for day, anomaly_type in sorted(anomalies_injected.items()):
        date = start_date + timedelta(days=day)
        print(f"  • Day {day} ({date.strftime('%Y-%m-%d')}): {anomaly_type}")

    # Quick stats
    print("\nQuick Stats:")
    print(f"  • Date range: {start_date.date()} to {(start_date + timedelta(days=29)).date()}")
    print(f"  • Avg orders/day: {len(orders) / 30:.0f}")
    print(f"  • Null customer_ids: {orders['customer_id'].isna().sum()}")
    print(f"  • Negative prices: {(items['unit_price'] < 0).sum()}")

    print("\nNext step: Run the profiler with 'python -m src.data_profiler'")


def main():
    """Generate sample data and save to CSV files."""
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print("   Data Quality Platform Generator")
    print("=" * 50)

    # Initialize random seed
    initialize_random_seed(seed=42)

    # Generate products
    products = generate_products(n=100)

    # Configure anomalies - these are the issues we'll detect later
    anomalies = {
        10: "volume_drop",      # Day 10: 50% fewer orders
        15: "negative_prices",  # Day 15: Some negative prices
        20: "null_customers",   # Day 20: Missing customer IDs
        25: "category_shift",   # Day 25: Category distribution changes
    }

    start_date = datetime(2024, 1, 1)
    orders, items, anomalies_injected = generate_orders(
        products=products,
        start_date=start_date,
        days=30,
        orders_per_day=1000,
        anomaly_config=anomalies,
    )

    # Save to CSV
    orders.to_csv(data_dir / "orders.csv", index=False)
    items.to_csv(data_dir / "order_items.csv", index=False)
    products.to_csv(data_dir / "products.csv", index=False)

    # Print summary
    print_summary(orders, items, products, anomalies_injected, start_date)


if __name__ == "__main__":
    main()
