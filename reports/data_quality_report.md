# Data Quality Report

## 1. Stores
- Addressed inconsistent whitespace and casing in the 'region' column.

## 2. Products
- Identified and removed 1 duplicate product entries.
- Handled missing product categories (imputed 1 missing values with 'Unknown').

## 3. Transactions
- Removed 21 records with missing transaction dates.
- Filtered out 179 invalid transactions with zero or negative quantities.

## 4. Inventory
- Addressed missing stock level values using forward/backward fill.
