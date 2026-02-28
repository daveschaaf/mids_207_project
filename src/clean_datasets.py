import pandas as pd
import numpy as np
from datetime import datetime

# Define paths
base_path = "/Users/karimmattar11/Desktop/Berkeley/ds207/Project/archive (4)/"

# Initialize summary report
summary = {
    'articles': {},
    'customers': {},
    'transactions': {}
}

print("=" * 60)
print("DATA CLEANING REPORT - H&M Datasets")
print("=" * 60)
print(f"Cleaning started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ============================================================
# 1. CLEAN ARTICLES DATASET
# ============================================================
print("\n[1/3] CLEANING ARTICLES DATASET")
print("-" * 40)

articles = pd.read_csv(base_path + "articles_hm.csv")
summary['articles']['original_rows'] = len(articles)
summary['articles']['original_cols'] = len(articles.columns)

print(f"Original shape: {articles.shape}")

# Check for duplicates
duplicates = articles.duplicated(subset=['article_id']).sum()
summary['articles']['duplicates_removed'] = duplicates
if duplicates > 0:
    articles = articles.drop_duplicates(subset=['article_id'], keep='first')
    print(f"Removed {duplicates} duplicate article_id rows")

# Check for missing values
missing_before = articles.isnull().sum().sum()
summary['articles']['missing_values_before'] = missing_before

# Handle missing values
# For categorical columns, fill with 'Unknown'
categorical_cols = articles.select_dtypes(include=['object']).columns
for col in categorical_cols:
    missing_count = articles[col].isnull().sum()
    if missing_count > 0:
        articles[col] = articles[col].fillna('Unknown')
        print(f"  Filled {missing_count} missing values in '{col}' with 'Unknown'")

# For numeric columns, fill with median
numeric_cols = articles.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    missing_count = articles[col].isnull().sum()
    if missing_count > 0:
        median_val = articles[col].median()
        articles[col] = articles[col].fillna(median_val)
        print(f"  Filled {missing_count} missing values in '{col}' with median ({median_val})")

summary['articles']['missing_values_after'] = articles.isnull().sum().sum()

# Ensure article_id is string and properly formatted
articles['article_id'] = articles['article_id'].astype(str).str.zfill(10)
articles['product_code'] = articles['product_code'].astype(str).str.zfill(7)

# Strip whitespace from string columns
for col in categorical_cols:
    if col in articles.columns:
        articles[col] = articles[col].str.strip()

summary['articles']['final_rows'] = len(articles)
print(f"Final shape: {articles.shape}")

# Save cleaned articles
articles.to_csv(base_path + "articles_hm_cleaned.csv", index=False)
print("Saved: articles_hm_cleaned.csv")

# ============================================================
# 2. CLEAN CUSTOMERS DATASET
# ============================================================
print("\n[2/3] CLEANING CUSTOMERS DATASET")
print("-" * 40)

customers = pd.read_csv(base_path + "customer_hm.csv")
summary['customers']['original_rows'] = len(customers)
summary['customers']['original_cols'] = len(customers.columns)

print(f"Original shape: {customers.shape}")

# Check for duplicates
duplicates = customers.duplicated(subset=['customer_id']).sum()
summary['customers']['duplicates_removed'] = duplicates
if duplicates > 0:
    customers = customers.drop_duplicates(subset=['customer_id'], keep='first')
    print(f"Removed {duplicates} duplicate customer_id rows")

# Check for missing values
missing_before = customers.isnull().sum().sum()
summary['customers']['missing_values_before'] = missing_before

# Handle missing values for each column appropriately
# FN and Active - binary columns, fill with mode (0)
for col in ['FN', 'Active']:
    missing_count = customers[col].isnull().sum()
    if missing_count > 0:
        customers[col] = customers[col].fillna(0).astype(int)
        print(f"  Filled {missing_count} missing values in '{col}' with 0")

# club_member_status - fill with 'Unknown'
if customers['club_member_status'].isnull().sum() > 0:
    missing_count = customers['club_member_status'].isnull().sum()
    customers['club_member_status'] = customers['club_member_status'].fillna('Unknown')
    print(f"  Filled {missing_count} missing values in 'club_member_status' with 'Unknown'")

# fashion_news_frequency - fill with 'NONE'
if customers['fashion_news_frequency'].isnull().sum() > 0:
    missing_count = customers['fashion_news_frequency'].isnull().sum()
    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NONE')
    print(f"  Filled {missing_count} missing values in 'fashion_news_frequency' with 'NONE'")

# age - fill with median, handle outliers
if customers['age'].isnull().sum() > 0:
    missing_count = customers['age'].isnull().sum()
    median_age = customers['age'].median()
    customers['age'] = customers['age'].fillna(median_age)
    print(f"  Filled {missing_count} missing values in 'age' with median ({median_age})")

# Handle age outliers (cap at reasonable bounds: 10-100)
age_outliers = ((customers['age'] < 10) | (customers['age'] > 100)).sum()
if age_outliers > 0:
    customers['age'] = customers['age'].clip(lower=10, upper=100)
    print(f"  Capped {age_outliers} age outliers to range [10, 100]")
    summary['customers']['age_outliers_fixed'] = age_outliers

summary['customers']['missing_values_after'] = customers.isnull().sum().sum()

# Standardize categorical values
customers['club_member_status'] = customers['club_member_status'].str.upper().str.strip()
customers['fashion_news_frequency'] = customers['fashion_news_frequency'].str.upper().str.strip()

# Ensure proper data types
customers['FN'] = customers['FN'].astype(int)
customers['Active'] = customers['Active'].astype(int)
customers['age'] = customers['age'].astype(int)

summary['customers']['final_rows'] = len(customers)
print(f"Final shape: {customers.shape}")

# Save cleaned customers
customers.to_csv(base_path + "customer_hm_cleaned.csv", index=False)
print("Saved: customer_hm_cleaned.csv")

# ============================================================
# 3. CLEAN TRANSACTIONS DATASET
# ============================================================
print("\n[3/3] CLEANING TRANSACTIONS DATASET")
print("-" * 40)

transactions = pd.read_csv(base_path + "transactions_hm.csv")
summary['transactions']['original_rows'] = len(transactions)
summary['transactions']['original_cols'] = len(transactions.columns)

print(f"Original shape: {transactions.shape}")

# Check for duplicates (exact duplicates across all columns)
duplicates = transactions.duplicated().sum()
summary['transactions']['duplicates_removed'] = duplicates
if duplicates > 0:
    transactions = transactions.drop_duplicates(keep='first')
    print(f"Removed {duplicates} exact duplicate rows")

# Check for missing values
missing_before = transactions.isnull().sum().sum()
summary['transactions']['missing_values_before'] = missing_before

# Handle missing values
for col in transactions.columns:
    missing_count = transactions[col].isnull().sum()
    if missing_count > 0:
        print(f"  Found {missing_count} missing values in '{col}'")
        if col == 't_dat':
            transactions = transactions.dropna(subset=['t_dat'])
            print(f"    Dropped rows with missing dates")
        elif col == 'customer_id':
            transactions = transactions.dropna(subset=['customer_id'])
            print(f"    Dropped rows with missing customer_id")
        elif col == 'article_id':
            transactions = transactions.dropna(subset=['article_id'])
            print(f"    Dropped rows with missing article_id")
        elif col == 'price':
            median_price = transactions['price'].median()
            transactions[col] = transactions[col].fillna(median_price)
            print(f"    Filled with median price ({median_price})")
        elif col == 'sales_channel_id':
            mode_channel = transactions['sales_channel_id'].mode()[0]
            transactions[col] = transactions[col].fillna(mode_channel)
            print(f"    Filled with mode ({mode_channel})")

summary['transactions']['missing_values_after'] = transactions.isnull().sum().sum()

# Convert date column to datetime
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], errors='coerce')

# Remove any rows where date conversion failed
invalid_dates = transactions['t_dat'].isnull().sum()
if invalid_dates > 0:
    transactions = transactions.dropna(subset=['t_dat'])
    print(f"Removed {invalid_dates} rows with invalid dates")
    summary['transactions']['invalid_dates_removed'] = invalid_dates

# Ensure article_id is properly formatted
transactions['article_id'] = transactions['article_id'].astype(str).str.zfill(10)

# Handle price outliers (remove negative or extremely high prices)
price_outliers = ((transactions['price'] < 0) | (transactions['price'] > 1)).sum()
if price_outliers > 0:
    transactions = transactions[(transactions['price'] >= 0) & (transactions['price'] <= 1)]
    print(f"Removed {price_outliers} rows with invalid prices (outside 0-1 range)")
    summary['transactions']['price_outliers_removed'] = price_outliers

# Ensure proper data types
transactions['sales_channel_id'] = transactions['sales_channel_id'].astype(int)
transactions['price'] = transactions['price'].astype(float)

summary['transactions']['final_rows'] = len(transactions)
print(f"Final shape: {transactions.shape}")

# Save cleaned transactions
transactions.to_csv(base_path + "transactions_hm_cleaned.csv", index=False)
print("Saved: transactions_hm_cleaned.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

for dataset, stats in summary.items():
    print(f"\n{dataset.upper()}:")
    print(f"  Original rows: {stats.get('original_rows', 'N/A')}")
    print(f"  Final rows: {stats.get('final_rows', 'N/A')}")
    print(f"  Duplicates removed: {stats.get('duplicates_removed', 0)}")
    print(f"  Missing values (before): {stats.get('missing_values_before', 0)}")
    print(f"  Missing values (after): {stats.get('missing_values_after', 0)}")

print("\n" + "=" * 60)
print("CLEANING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nCleaned files saved:")
print("  - articles_hm_cleaned.csv")
print("  - customer_hm_cleaned.csv")
print("  - transactions_hm_cleaned.csv")
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
