from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw")


def inspect_csv(file_path: Path) -> None:
    df = pd.read_csv(file_path)

    print("=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFirst 3 rows:")
    print(df.head(3))


def main() -> None:
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    for file_path in csv_files:
        inspect_csv(file_path)


if __name__ == "__main__":
    main()