import pytest
import pandas as pd
from src.customer_feature_engineering import CustomerFeatureEngineering
from pathlib import Path

data_path = Path("./data")
customers_path = data_path / 'customer_hm_cleaned.csv'
transactions_path = data_path / 'transactions_hm_cleaned.csv'

def test_customer_feature_engineering():
    cfe = CustomerFeatureEngineering(customers_path, transactions_path)
    assert isinstance(cfe.customers, pd.DataFrame)
    assert isinstance(cfe.transactions, pd.DataFrame)
