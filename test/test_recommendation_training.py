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
