import pytest
import pandas as pd
from src.recommendation_training import RecommendationTrainingBuilder
from src.customer_features import CustomerFeatureEngineer
from src.product_features import ProductFeatureEngineer

def test_get_positive_examples():
    """Test extraction of actual purchases in prediction window"""
    transactions = pd.DataFrame({
        't_dat': pd.to_datetime(['2019-09-15', '2019-10-05', '2019-10-15', '2019-11-01']),
        'customer_id': ['c1', 'c1', 'c2', 'c1'],
        'article_id': [123, 456, 789, 111],
        'price': [0.01, 0.02, 0.03, 0.04]
    })
    
    customer_features = pd.DataFrame({
        'customer_id': ['c1', 'c2'],
        'days_since_last_purchase': [5, 10]
    })
    
    product_features = pd.DataFrame({
        'article_id': [123, 456, 789, 111],
        'avg_price': [0.01, 0.02, 0.03, 0.04]
    })
    
    builder = RecommendationTrainingBuilder(transactions, customer_features, product_features)
    positives = builder._get_positive_examples(
        prediction_start='2019-10-01',
        prediction_end='2019-10-31'
    )
    assert len(positives) == 2
    assert set(positives['customer_id'].values) == {'c1', 'c2'}
    assert set(positives['article_id'].values) == {456, 789}
    assert 'purchased' in positives.columns
    assert (positives['purchased'] == 1).all()


def test_get_positive_examples_with_features():
    """Test that positive examples include customer and product features"""
    transactions = pd.DataFrame({
        't_dat': pd.to_datetime(['2019-10-05', '2019-10-15', '2019-10-20', '2019-10-25']),
        'customer_id': ['c1', 'c2', 'c3', 'c1'],
        'article_id': [456, 789, 456, 111],
        'price': [0.02, 0.03, 0.02, 0.01]
    })

    customer_features = pd.DataFrame({
        'customer_id': ['c1', 'c2'],
        'days_since_last_purchase': [5, 10],
        'num_purchases': [3, 7]
    })
    
    product_features = pd.DataFrame({
        'article_id': [456, 789],
        'avg_price': [0.02, 0.03],
        'sales_last_7_days': [10, 20]
    })
    customer_fill_values = {'num_purchases': 0, 'days_since_last_purchase': 999, 'is_new_customer': 1}
    product_fill_values = {'avg_price': 0, 'sales_last_7_days': 0}
    builder = RecommendationTrainingBuilder(
        transactions, customer_features, product_features,
        customer_fill_values=customer_fill_values,
        product_fill_values=product_fill_values)
    
    positives = builder._get_positive_examples(
        prediction_start='2019-10-01',
        prediction_end='2019-10-31'
    )
    
    assert len(positives) == 4
    assert 'purchased' in positives.columns
    assert 'days_since_last_purchase' in positives.columns
    assert 'num_purchases' in positives.columns
    assert 'avg_price' in positives.columns
    assert 'sales_last_7_days' in positives.columns
    assert 'is_new_customer' in positives.columns

    c1_row = positives[positives['customer_id'] == 'c1'].iloc[0]
    assert c1_row['days_since_last_purchase'] == 5
    assert c1_row['avg_price'] == 0.02
    assert c1_row['purchased'] == 1
    assert c1_row['is_new_customer'] == 0

    c3_row = positives[positives['customer_id'] == 'c3'].iloc[0]
    assert c3_row['days_since_last_purchase'] == 999
    assert c3_row['avg_price'] == 0.02
    assert c3_row['purchased'] == 1
    assert c3_row['num_purchases'] == 0
    assert c3_row['is_new_customer'] == 1

    art111_row = positives[positives['article_id'] == 111].iloc[0]
    assert art111_row['avg_price'] == 0
    assert art111_row['sales_last_7_days'] == 0

def test_sample_negative_examples():
    """Test negative sampling maintains proper ratio per customer"""
    positives = pd.DataFrame({
        'customer_id': ['c1', 'c1', 'c2'],
        'article_id': [456, 789, 999],
        'purchased': [1, 1, 1]
    })
    
    transactions = pd.DataFrame({
        't_dat': pd.to_datetime(['2019-10-05', '2019-10-10', '2019-10-15']),
        'customer_id': ['c1', 'c1', 'c2'],
        'article_id': [456, 789, 999],
        'price': [0.02, 0.03, 0.04]
    })
    
    customer_features = pd.DataFrame({
        'customer_id': ['c1', 'c2'],
        'days_since_last_purchase': [5, 10]
    })
    
    product_features = pd.DataFrame({
        'article_id': [123, 456, 789, 999, 888, 777, 666],
        'avg_price': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    })
    
    builder = RecommendationTrainingBuilder(transactions, customer_features, product_features)
    
    negatives = builder._sample_negative_examples(positives, negative_ratio=2, random_state=42)
    assert len(negatives) == 6
    assert (negatives['purchased'] == 0).all()
    c1_negatives = negatives[negatives['customer_id'] == 'c1']
    c2_negatives = negatives[negatives['customer_id'] == 'c2']
    assert len(c1_negatives) == 4  # 2 positives × 2
    assert len(c2_negatives) == 2  # 1 positive × 2
    negative_pairs = set(zip(negatives['customer_id'], negatives['article_id']))
    positive_pairs = set(zip(positives['customer_id'], positives['article_id']))
    assert len(negative_pairs & positive_pairs) == 0
    assert 'days_since_last_purchase' in negatives.columns
    assert 'avg_price' in negatives.columns

def test_build_dataset_integration(real_data):
    """Integration test: create full training dataset with real H&M data"""
    
    sample_size = 10000
    transactions_sample = real_data['transactions'].sample(n=sample_size, random_state=42)
    cfe = CustomerFeatureEngineer(real_data['customers'], transactions_sample)
    customer_features = cfe.calculate_all_features(
        articles_df=real_data['articles'],
        as_of_date='2019-09-30'
    )
    
    pfe = ProductFeatureEngineer(real_data['articles'], transactions_sample)
    product_features = pfe.calculate_all_features(as_of_date='2019-09-30')
    
    builder = RecommendationTrainingBuilder(
        transactions_sample,
        customer_features,
        product_features,
        product_fill_values=pfe.get_fill_values(),
        customer_fill_values=cfe.get_fill_values()
    )
    training_data = builder.build_dataset(
        prediction_start='2019-10-01',
        prediction_end='2019-10-31',
        negative_ratio=5,
        random_state=42
    )
    assert len(training_data) > 0
    assert 'customer_id' in training_data.columns
    assert 'article_id' in training_data.columns
    assert 'purchased' in training_data.columns
    assert 'days_since_last_purchase' in training_data.columns
    assert 'num_purchases' in training_data.columns
    assert 'sales_last_7_days' in training_data.columns
    assert 'avg_price' in training_data.columns
    assert 'age' in training_data.columns
    assert 'age_x' not in training_data.columns
    assert 'age_y' not in training_data.columns
    feature_cols = [col for col in training_data.columns 
                    if col not in ['customer_id', 'article_id', 'purchased']]
    for col in feature_cols:
        nan_count = training_data[col].isna().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in {col}"
    num_positives = (training_data['purchased'] == 1).sum()
    num_negatives = (training_data['purchased'] == 0).sum()
    assert num_negatives / num_positives >= 4.5
    assert num_negatives / num_positives <= 5.5
