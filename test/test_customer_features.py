import pytest
import pandas as pd
from src.customer_features import CustomerFeatureEngineer
from pathlib import Path

data_path = Path("./data")
customers_path = data_path / 'customer_hm_cleaned.csv'
transactions_path = data_path / 'transactions_hm_cleaned.csv'
customers_df = pd.read_csv(customers_path)
transactions_df = pd.read_csv(transactions_path)

@pytest.fixture
def cfe():
    return CustomerFeatureEngineer(customers_df, transactions_df)

def test_customer_feature_engineer_init(cfe):
    assert isinstance(cfe.customers, pd.DataFrame)
    assert isinstance(cfe.transactions, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(cfe.transactions['t_dat'])

def test_rfm_returns_correct_columns(cfe):
    result = cfe.calculate_rfm(as_of_date='2019-10-31')
    assert 'customer_id' in result.columns
    assert 'days_since_last_purchase' in result.columns
    assert 'num_purchases' in result.columns
    assert 'total_spent' in result.columns

def test_rfm_correct_types(cfe):
    result = cfe.calculate_rfm(as_of_date='2019-10-31')
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
    
    fe = CustomerFeatureEngineer(test_customers, test_transactions)
    result = fe.calculate_rfm(as_of_date='2019-10-31')
    
    assert result.loc[result['customer_id'] == 'cust_1', 'days_since_last_purchase'].values[0] == 16 
    assert result.loc[result['customer_id'] == 'cust_1', 'num_purchases'].values[0] == 2
    assert result.loc[result['customer_id'] == 'cust_1', 'total_spent'].values[0] == 0.03
