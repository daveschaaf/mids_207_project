import pandas as pd


class RecommendationTrainingBuilder:
    def __init__(self, transactions_df, customer_features_df, product_features_df):
        self.transactions = transactions_df
        self.customer_features = customer_features_df
        self.product_features = product_features_df
    
    def _get_positive_examples(self, prediction_start=None, prediction_end=None):
        if prediction_start == None:
            prediction_start = self.transactions['t_dat'].min()
        prediction_start = pd.to_datetime(prediction_start)
        if prediction_end == None:
            prediction_end = self.transactions['t_dat'].max()
        prediction_end = pd.to_datetime(prediction_end)
        start_mask = self.transactions['t_dat']<= prediction_end
        end_mask = self.transactions['t_dat']>= prediction_start
        relevant_txn = self.transactions[start_mask & end_mask]
        
        positives = relevant_txn[['customer_id', 'article_id']].drop_duplicates()
        positives['purchased'] = 1
        return positives
