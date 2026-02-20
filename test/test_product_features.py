import pytest
import pandas as pd
from src.product_features import ProductFeatureEngineer


def test_popularity_features_single_product():
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01', '2019-10-25', '2019-10-28'],
        'customer_id': ['c1', 'c2', 'c3'],
        'article_id': [123, 123, 123],
        'price': [0.01, 0.01, 0.01]
    })
    
    test_articles = pd.DataFrame({
        'article_id': [123],
        'department_name': ['Jersey']
    })
    
    pfe = ProductFeatureEngineer(test_articles, test_transactions)
    result = pfe.calculate_popularity(as_of_date='2019-10-31')
    
    product_row = result[result['article_id'] == 123].iloc[0]
    
    assert product_row['sales_last_7_days'] == 2
    assert product_row['sales_last_30_days'] == 3
    assert product_row['days_since_first_sale'] == 30
    assert product_row['days_since_last_sale'] == 3

def test_popularity_no_recent_sales():
    """Product sold long ago, but not in last 7 or 30 days"""
    test_transactions = pd.DataFrame({
        't_dat': ['2019-08-01', '2019-08-15'],
        'customer_id': ['c1', 'c2'],
        'article_id': [123, 123],
        'price': [0.01, 0.01]
    })
    
    test_articles = pd.DataFrame({
        'article_id': [123],
        'department_name': ['Jersey']
    })
    
    pfe = ProductFeatureEngineer(test_articles, test_transactions)
    result = pfe.calculate_popularity(as_of_date='2019-10-31')
    
    product_row = result[result['article_id'] == 123].iloc[0]
    
    # No sales in last 7 or 30 days
    assert product_row['sales_last_7_days'] == 0
    assert product_row['sales_last_30_days'] == 0
    
    # Days since first sale: Oct 31 - Aug 1 = 91 days
    assert product_row['days_since_first_sale'] == 91
    
    # Days since last sale: Oct 31 - Aug 15 = 77 days
    assert product_row['days_since_last_sale'] == 77


def test_popularity_multiple_products():
    """Multiple products with different popularity"""
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-25', '2019-10-26', '2019-10-28', '2019-10-01'],
        'customer_id': ['c1', 'c2', 'c3', 'c4'],
        'article_id': [123, 123, 456, 456],
        'price': [0.01, 0.01, 0.02, 0.02]
    })
    
    test_articles = pd.DataFrame({
        'article_id': [123, 456]
    })
    
    pfe = ProductFeatureEngineer(test_articles, test_transactions)
    result = pfe.calculate_popularity(as_of_date='2019-10-31')
    
    product_123 = result[result['article_id'] == 123].iloc[0]
    assert product_123['sales_last_7_days'] == 2
    assert product_123['sales_last_30_days'] == 2
    
    product_456 = result[result['article_id'] == 456].iloc[0]
    assert product_456['sales_last_7_days'] == 1
    assert product_456['sales_last_30_days'] == 2


def test_popularity_product_never_sold():
    """Product exists in articles but has no sales"""
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-25'],
        'customer_id': ['c1'],
        'article_id': [123],
        'price': [0.01]
    })
    
    test_articles = pd.DataFrame({
        'article_id': [123, 456]
    })
    
    pfe = ProductFeatureEngineer(test_articles, test_transactions)
    result = pfe.calculate_popularity(as_of_date='2019-10-31')
    
    assert 123 in result['article_id'].values
    assert 456 not in result['article_id'].values
