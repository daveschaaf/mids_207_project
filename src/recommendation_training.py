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

        positives = positives.merge(
            self.product_features,
            on='article_id',
            how='left'
        )

        positives = positives.merge(
            self.customer_features,
            on='customer_id',
            how='left'
        )
        return positives
    
    def _sample_negative_examples(self, positives, negative_ratio=2, random_state=67):
        customer_ids = self.customer_features['customer_id'].unique()
        article_ids = self.product_features['article_id'].unique()

        negative_examples = []
        for customer in customer_ids:
            bought = positives[positives['customer_id'] == customer].values

            non_purchase_products = [art_id for art_id in article_ids if art_id not in bought]
            sampled_products = pd.Series(non_purchase_products).sample(n=negative_ratio, random_state=random_state)

            for product in sampled_products:
                negative_examples.append({
                    'customer_id': customer,
                    'article_id': product,
                    'purchased': 0
                })
        negatives = pd.DataFrame(negative_examples)
        negatives = negatives.merge(
            self.product_features,
            on='article_id',
            how='left'
        )
        negatives = negatives.merge(
            self.customer_features,
            on='customer_id',
            how='left'
        )
        return negatives
