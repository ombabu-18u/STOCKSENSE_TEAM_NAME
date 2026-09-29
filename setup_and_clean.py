import os
import pandas as pd
import numpy as np

def main():
    print("1. Creating folders...")
    dirs = ['data/raw', 'data/processed', 'reports']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("2. Generating sample CSV files with traps...")
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-01-10')

    # Stores
    stores = pd.DataFrame({
        'store_id': [1, 2, 3],
        'region': ['North', 'south', 'North '] # Trap: inconsistent casing and whitespace
    })
    stores.to_csv('data/raw/stores.csv', index=False)

    # Products
    products = pd.DataFrame({
        'product_id': [101, 102, 103, 103], # Trap: duplicate product 103
        'category': ['Electronics', 'Clothing', None, 'Groceries'] # Trap: missing category
    })
    products.to_csv('data/raw/products.csv', index=False)

    # Transactions
    n_trans = 500
    transactions = pd.DataFrame({
        'transaction_id': range(n_trans),
        'date': np.random.choice(dates, n_trans),
        'store_id': np.random.choice([1, 2, 3], n_trans),
        'product_id': np.random.choice([101, 102, 103], n_trans),
        'quantity': np.random.randint(-5, 10, n_trans), # Trap: negative quantities
        'price': np.random.uniform(5, 50, n_trans)
    })
    # Introduce missing dates
    transactions.loc[0:20, 'date'] = np.nan
    transactions.to_csv('data/raw/transactions.csv', index=False)

    # Inventory
    inventory = pd.DataFrame({
        'date': np.tile(dates, 9),
        'store_id': np.repeat([1,2,3], 30),
        'product_id': np.tile(np.repeat([101,102,103], 10), 3),
        'stock_level': np.random.randint(0, 100, 90)
    })
    inventory.loc[10:15, 'stock_level'] = np.nan # Trap: missing stock
    inventory.to_csv('data/raw/inventory.csv', index=False)

    # External factors
    external = pd.DataFrame({
        'date': dates,
        'holiday': np.random.choice([0, 1], len(dates)),
        'temperature': np.random.uniform(-10, 35, len(dates))
    })
    external.to_csv('data/raw/external_factors.csv', index=False)

    print("3. Loading, cleaning, and aggregating data...")
    stores_cln = stores.copy()
    stores_cln['region'] = stores_cln['region'].str.strip().str.title()

    products_cln = products.drop_duplicates(subset=['product_id'], keep='first').copy()
    products_cln['category'] = products_cln['category'].fillna('Unknown')

    trans_cln = transactions.dropna(subset=['date']).copy()
    trans_cln = trans_cln[trans_cln['quantity'] > 0]

    inv_cln = inventory.copy()
    inv_cln['stock_level'] = inv_cln['stock_level'].bfill().ffill()

    # Aggregate to daily Store x Product
    daily_sales = trans_cln.groupby(['date', 'store_id', 'product_id']).agg({
        'quantity': 'sum',
        'price': 'mean'
    }).reset_index()

    master = pd.merge(daily_sales, stores_cln, on='store_id', how='left')
    master = pd.merge(master, products_cln, on='product_id', how='left')
    master = pd.merge(master, inv_cln, on=['date', 'store_id', 'product_id'], how='left')
    master = pd.merge(master, external, on='date', how='left')

    master.to_csv('data/processed/master_analytics_dataset.csv', index=False)
    print("Master dataset created at data/processed/master_analytics_dataset.csv")

    print("4. Creating Data Quality Report...")
    report_content = f"""# Data Quality Report

## 1. Stores
- Addressed inconsistent whitespace and casing in the 'region' column.

## 2. Products
- Identified and removed {len(products) - len(products_cln)} duplicate product entries.
- Handled missing product categories (imputed {products['category'].isna().sum()} missing values with 'Unknown').

## 3. Transactions
- Removed {transactions['date'].isna().sum()} records with missing transaction dates.
- Filtered out {len(transactions) - len(trans_cln) - transactions['date'].isna().sum()} invalid transactions with zero or negative quantities.

## 4. Inventory
- Addressed missing stock level values using forward/backward fill.
"""
    with open('reports/data_quality_report.md', 'w') as f:
        f.write(report_content)

    print("Data Quality Report generated at reports/data_quality_report.md")
    print("Setup and cleaning complete!")

if __name__ == '__main__':
    main()
