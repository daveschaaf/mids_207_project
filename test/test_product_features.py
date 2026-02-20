import pytest
import pandas as pd
import numpy as np
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
    
    assert product_row['sales_last_7_days'] == 0
    assert product_row['sales_last_30_days'] == 0
    assert product_row['days_since_first_sale'] == 91
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


def test_price_features_single_product():
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01', '2019-10-15', '2019-10-25'],
        'customer_id': ['c1', 'c2', 'c3'],
        'article_id': [123, 123, 123],
        'price': [0.01, 0.02, 0.03]
    })
    
    test_articles = pd.DataFrame({'article_id': [123]})
    
    pfe = ProductFeatureEngineer(test_articles, test_transactions)
    result = pfe.calculate_price_features(as_of_date='2019-10-31')
    
    product_row = result[result['article_id'] == 123].iloc[0]
    
    assert product_row['avg_price'] == 0.02  # mean([0.01, 0.02, 0.03])
    assert product_row['min_price'] == 0.01
    assert product_row['max_price'] == 0.03
    assert np.isclose(product_row['product_price_std'], np.std([0.01, 0.02, 0.03], ddof=1))

def test_product_all_features_integration(real_data):
    """Integration test using actual H&M data"""
    pfe = ProductFeatureEngineer(real_data['articles'], real_data['transactions'])
    result = pfe.calculate_all_features(as_of_date='2019-10-31')
    expected_columns = [
        'article_id', 
        'sales_last_7_days', 'sales_last_30_days', 
        'days_since_first_sale', 'days_since_last_sale',
        'avg_price', 'product_price_std', 'min_price', 'max_price'
    ]
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"
    
    assert len(result) > 0

    assert (result['sales_last_7_days'] >= 0).all()
    assert (result['sales_last_30_days'] >= result['sales_last_7_days']).all()
    assert (result['days_since_first_sale'] >= 0).all()
    assert (result['days_since_last_sale'] >= 0).all()
    assert (result['avg_price'] > 0).all()

    tolerance = 1e-9
    assert ((result['min_price'] <= result['avg_price'] + tolerance)).all()
    assert ((result['max_price'] >= result['avg_price'] - tolerance)).all()

    products_with_nan_std = result[result['product_price_std'].isna()]
    print(f"\n{len(products_with_nan_std)} products have NaN price_std (likely single sale)")
