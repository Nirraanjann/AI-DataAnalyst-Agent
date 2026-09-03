from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

DATA_DIR = Path("data/raw")

# Update the password below to match what you set
DB_URL = "postgresql+psycopg2://postgres:achu@localhost:5432/ecommerce_analyst"

def backfill_missing_categories(engine, products_csv: str, translation_csv: str) -> None:
    products_df = pd.read_csv(DATA_DIR / products_csv)
    translation_df = pd.read_csv(DATA_DIR / translation_csv)

    product_categories = set(products_df["product_category_name"].dropna())
    known_categories = set(translation_df["product_category_name"])

    missing = product_categories - known_categories

    if missing:
        print(f"Found {len(missing)} category name(s) missing from translation table: {missing}")
        missing_df = pd.DataFrame({
            "product_category_name": list(missing),
            "product_category_name_english": list(missing),  # placeholder, same as original
        })
        missing_df.to_sql("category_translation", engine, if_exists="append", index=False)
        print(f"Backfilled {len(missing)} missing categories into category_translation")


def load_table(engine, csv_name: str, table_name: str, date_cols=None, keep_cols=None) -> None:
    file_path = DATA_DIR / csv_name
    df = pd.read_csv(file_path, parse_dates=date_cols)

    if keep_cols is not None:
        df = df[keep_cols]

    df.to_sql(table_name, engine, if_exists="append", index=False)

    print(f"Loaded {len(df):,} rows into '{table_name}' from {csv_name}")


def main() -> None:
    engine = create_engine(DB_URL)

    # Order matters: load parent tables before child tables (foreign keys)
    load_table(engine, "product_category_name_translation.csv", "category_translation")
    load_table(engine, "olist_customers_dataset.csv", "customers")
    load_table(engine, "olist_sellers_dataset.csv", "sellers")
    backfill_missing_categories(engine, "olist_products_dataset.csv", "product_category_name_translation.csv")
    
    load_table(
    engine,
    "olist_products_dataset.csv",
    "products",
    keep_cols=[
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ],
)
    load_table(
        engine,
        "olist_orders_dataset.csv",
        "orders",
        date_cols=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    load_table(
        engine,
        "olist_order_items_dataset.csv",
        "order_items",
        date_cols=["shipping_limit_date"],
    )
    load_table(engine, "olist_order_payments_dataset.csv", "order_payments")

    print("\nAll tables loaded successfully.")


if __name__ == "__main__":
    main()