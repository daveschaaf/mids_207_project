import pandas as pd
from src.recommendation_training import RecommendationTrainingBuilder

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
        't_dat': pd.to_datetime(['2019-10-05', '2019-10-15']),
        'customer_id': ['c1', 'c2'],
        'article_id': [456, 789],
        'price': [0.02, 0.03]
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
    
    builder = RecommendationTrainingBuilder(transactions, customer_features, product_features)
    positives = builder._get_positive_examples(
        prediction_start='2019-10-01',
        prediction_end='2019-10-31'
    )
    
    assert len(positives) == 2
    assert 'purchased' in positives.columns
    assert 'days_since_last_purchase' in positives.columns
    assert 'num_purchases' in positives.columns
    assert 'avg_price' in positives.columns
    assert 'sales_last_7_days' in positives.columns
    c1_row = positives[positives['customer_id'] == 'c1'].iloc[0]
    assert c1_row['days_since_last_purchase'] == 5
    assert c1_row['avg_price'] == 0.02
    assert c1_row['purchased'] == 1

def test_sample_negative_examples():
    """Test negative sampling - customers paired with products they didn't buy"""
    positives = pd.DataFrame({
        'customer_id': ['c1', 'c2'],
        'article_id': [456, 789],
        'purchased': [1, 1]
    })
    
    transactions = pd.DataFrame({
        't_dat': pd.to_datetime(['2019-10-05', '2019-10-15']),
        'customer_id': ['c1', 'c2'],
        'article_id': [456, 789],
        'price': [0.02, 0.03]
    })
    
    customer_features = pd.DataFrame({
        'customer_id': ['c1', 'c2'],
        'days_since_last_purchase': [5, 10]
    })
    
    product_features = pd.DataFrame({
        'article_id': [123, 456, 789, 999],
        'avg_price': [0.01, 0.02, 0.03, 0.04]
    })
    
    builder = RecommendationTrainingBuilder(transactions, customer_features, product_features)
    
    negatives = builder._sample_negative_examples(positives, negative_ratio=2, random_state=42)
    assert len(negatives) == 4
    assert (negatives['purchased'] == 0).all()
    negative_pairs = set(zip(negatives['customer_id'], negatives['article_id']))
    positive_pairs = set(zip(positives['customer_id'], positives['article_id']))
    assert len(negative_pairs & positive_pairs) == 0
    assert 'days_since_last_purchase' in negatives.columns
    assert 'avg_price' in negatives.columns
