import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def real_data():
    """Load real H&M data for integration tests"""
    data_path = Path("./data/processed")
    
    if not data_path.exists():
        pytest.skip("Data directory not found - skipping integration test")
    
    customers_path = data_path / 'customer_hm_cleaned.csv'
    transactions_path = data_path/ 'transactions_hm_cleaned.csv'
    articles_path = data_path / 'articles_hm_cleaned.csv'
    
    if not all([customers_path.exists(), transactions_path.exists(), articles_path.exists()]):
        pytest.skip("Data files not found - skipping integration test")
    
    return {
        'customers': pd.read_csv(customers_path),
        'transactions': pd.read_csv(transactions_path),
        'articles': pd.read_csv(articles_path)
    }
