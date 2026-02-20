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
