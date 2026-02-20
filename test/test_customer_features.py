import pytest
import pandas as pd
import numpy as np
from src.customer_features import CustomerFeatureEngineer
from pathlib import Path


@pytest.fixture
def real_data():
    data_path = Path("./data")
    
    if not data_path.exists():
        pytest.skip("Data directory not found - skipping integration test")
    
    customers_path = data_path / 'customer_hm_cleaned.csv'
    transactions_path = data_path / 'transactions_hm_cleaned.csv'
    articles_path = data_path / 'articles_hm_cleaned.csv'
    
    if not all([customers_path.exists(), transactions_path.exists(), articles_path.exists()]):
        pytest.skip("Data files not found - skipping integration test")
    
    return {
        'customers': pd.read_csv(customers_path),
        'transactions': pd.read_csv(transactions_path),
        'articles': pd.read_csv(articles_path)
    }

def test_customer_feature_engineer_init():
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01'],
        'customer_id': ['cust_1'],
        'article_id': [123],
        'price': [0.01]
    })
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    assert isinstance(cfe.customers, pd.DataFrame)
    assert isinstance(cfe.transactions, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(cfe.transactions['t_dat'])


def test_rfm_returns_correct_columns():
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01'],
        'customer_id': ['cust_1'],
        'article_id': [123],
        'price': [0.01]
    })
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_rfm_and_behaviors(as_of_date='2019-10-31')
    
    assert 'customer_id' in result.columns
    assert 'days_since_last_purchase' in result.columns

def test_rfm_correct_types():
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01'],
        'customer_id': ['cust_1'],
        'article_id': [123],
        'price': [0.01]
    })
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)

    result = cfe.calculate_rfm_and_behaviors(as_of_date='2019-10-31')
    assert result['days_since_last_purchase'].dtype == 'int64'
    assert result['num_purchases'].dtype == 'int64'
    assert result['total_spent'].dtype == 'float64'

def test_rfm_with_single_customer():
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01', '2019-10-15'],
        'customer_id': ['cust_1', 'cust_1'],
        'article_id': [123, 456],
        'price': [0.01, 0.02]
    })
    test_customers = pd.DataFrame({
        "customer_id": ['cust_1']
    })
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_rfm_and_behaviors(as_of_date='2019-10-31')
    
    assert result.loc[result['customer_id'] == 'cust_1', 'days_since_last_purchase'].values[0] == 16 
    assert result.loc[result['customer_id'] == 'cust_1', 'num_purchases'].values[0] == 2
    assert result.loc[result['customer_id'] == 'cust_1', 'total_spent'].values[0] == 0.03

def test_behavioral_features_returns_correct_columns():
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01', '2019-10-10', '2019-10-20'],
        'customer_id': ['cust_1', 'cust_1', 'cust_1'],
        'article_id': [123, 456, 789],
        'price': [0.01, 0.02, 0.03]
    })
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_rfm_and_behaviors(as_of_date='2019-10-31')
    
    assert 'customer_id' in result.columns
    assert 'avg_transaction_value' in result.columns
    assert 'avg_days_between_purchases' in result.columns
    assert 'price_std' in result.columns


def test_behavioral_features_correct_types():
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01', '2019-10-10', '2019-10-20'],
        'customer_id': ['cust_1', 'cust_1', 'cust_1'],
        'article_id': [123, 456, 789],
        'price': [0.01, 0.02, 0.03]
    })
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_rfm_and_behaviors(as_of_date='2019-10-31')
    
    assert result['avg_transaction_value'].dtype == 'float64'
    assert result['avg_days_between_purchases'].dtype == 'float64'
    assert result['price_std'].dtype == 'float64'


def test_behavioral_features_single_customer():
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01', '2019-10-10', '2019-10-20'],
        'customer_id': ['cust_1', 'cust_1', 'cust_1'],
        'article_id': [123, 456, 789],
        'price': [0.01, 0.02, 0.03]
    })
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_rfm_and_behaviors(as_of_date='2019-10-31')
    customer_row = result[result['customer_id'] == 'cust_1'].iloc[0]
    
    assert customer_row['avg_transaction_value'] == 0.02
    assert customer_row['avg_days_between_purchases'] == 9.5
    expected_std = np.std([0.01, 0.02, 0.03], ddof=1)
    assert np.isclose(customer_row['price_std'], expected_std)

def test_behavioral_features_single_transaction():
    """Customer with only 1 purchase - can't calculate days between or std"""
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-15'],
        'customer_id': ['cust_1'],
        'article_id': [123],
        'price': [0.02]
    })
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_rfm_and_behaviors(as_of_date='2019-10-31')
    
    customer_row = result[result['customer_id'] == 'cust_1'].iloc[0]
    
    assert customer_row['avg_transaction_value'] == 0.02
    assert pd.isna(customer_row['avg_days_between_purchases']) or customer_row['avg_days_between_purchases'] == 0
    assert pd.isna(customer_row['price_std']) or customer_row['price_std'] == 0


def test_behavioral_features_no_transactions():
    """Customer with no purchases in time window"""
    test_transactions = pd.DataFrame({
        't_dat': [],
        'customer_id': [],
        'article_id': [],
        'price': []
    })
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_rfm_and_behaviors(as_of_date='2019-10-31')
    
    assert len(result) == 0

def test_category_preferences_returns_correct_columns():
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01'],
        'customer_id': ['cust_1'],
        'article_id': [123]
    })
    test_articles = pd.DataFrame({
        'article_id': [123],
        'department_name': ['Jersey Basic'],
        'garment_group_name': ['Jersey']
    })
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_category_preferences(
        as_of_date='2019-10-31',
        articles_df=test_articles
    )
    
    assert 'customer_id' in result.columns
    assert 'primary_department' in result.columns
    assert 'primary_garment_group' in result.columns
    assert 'category_diversity' in result.columns


def test_category_preferences_single_category():
    """Most common case - customer buys from one category"""
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01', '2019-10-10'],
        'customer_id': ['cust_1', 'cust_1'],
        'article_id': [123, 456]
    })
    
    test_articles = pd.DataFrame({
        'article_id': [123, 456],
        'department_name': ['Jersey Basic', 'Jersey Basic'],
        'garment_group_name': ['Jersey', 'Jersey']
    })
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_category_preferences(
        as_of_date='2019-10-31',
        articles_df=test_articles
    )
    
    customer_row = result[result['customer_id'] == 'cust_1'].iloc[0]
    
    assert customer_row['primary_department'] == 'Jersey Basic'
    assert customer_row['primary_garment_group'] == 'Jersey'
    assert customer_row['category_diversity'] == 1


def test_category_preferences_multiple_categories():
    """Customer buys from multiple categories - pick most common"""
    test_transactions = pd.DataFrame({
        't_dat': ['2019-10-01', '2019-10-10', '2019-10-15'],
        'customer_id': ['cust_1', 'cust_1', 'cust_1'],
        'article_id': [123, 456, 789]
    })
    
    test_articles = pd.DataFrame({
        'article_id': [123, 456, 789],
        'department_name': ['Jersey Basic', 'Jersey Basic', 'Shoes'],
        'garment_group_name': ['Jersey', 'Jersey', 'Footwear']
    })
    test_customers = pd.DataFrame({'customer_id': ['cust_1']})
    
    cfe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = cfe.calculate_category_preferences(
        as_of_date='2019-10-31',
        articles_df=test_articles
    )
    
    customer_row = result[result['customer_id'] == 'cust_1'].iloc[0]
    
    assert customer_row['primary_department'] == 'Jersey Basic'  # 2 out of 3
    assert customer_row['primary_garment_group'] == 'Jersey'
    assert customer_row['category_diversity'] == 2  # Jersey Basic + Shoes


def test_calculate_all_features_integration(real_data):
    """Integration test using actual H&M data"""
    cfe = CustomerFeatureEngineer(real_data['customers'], real_data['transactions'])
    result = cfe.calculate_all_features(
        articles_df=real_data['articles'], 
        as_of_date='2019-10-31'
    )
    
    # Check structure
    expected_columns = [
        'customer_id', 'days_since_last_purchase', 'num_purchases', 
        'total_spent', 'avg_transaction_value', 'price_std',
        'avg_days_between_purchases', 'primary_department',
        'primary_garment_group', 'category_diversity'
    ]
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"
    
    # Check we got results
    assert len(result) > 0
    assert len(result) <= len(real_data['customers'])
    
    # Check data quality
    assert (result['num_purchases'] >= 1).all()
    assert (result['days_since_last_purchase'] >= 0).all()
    assert (result['category_diversity'] >= 1).all()
