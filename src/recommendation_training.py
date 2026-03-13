import pandas as pd
import numpy as np

class RecommendationTrainingBuilder:
    def __init__(self, transactions_df, customer_features_df, product_features_df, customer_fill_values={}, product_fill_values={}):
        self.transactions = transactions_df
        self.customer_features = customer_features_df
        self.product_features = product_features_df
        self.customer_fill_values = customer_fill_values
        self.product_fill_values = product_fill_values

    def build_dataset(self, prediction_start=None, prediction_end=None, negative_ratio=2, random_state=67):
        """
        Creates a data set of customer-product pairs
        
        Outcome variable:
        purchased
        - 1 = purchased during the prediction period
        - 0 = did NOT purchase during the prediction period

        Method: build_dataset
        builds product and customer features
        prediction_start = Begin prediction period
        prediction_end = End of prediction period
        negative_ratio = Ratio of random pairs of customer-product that did not result in a transaction

        Returns:
            pd.DataFrame of features and outcome
        """
        positives = self._get_positive_examples(prediction_start, prediction_end)
        negatives = self._sample_negative_examples(positives, negative_ratio=negative_ratio, random_state=random_state)
        return pd.concat([positives, negatives], axis=0)
    
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
        if 'num_purchases' in positives.columns:
            positives['is_new_customer'] = positives['num_purchases'].isnull().astype(int)
        else:
            positives['is_new_customer'] = 0
        
        positives = self.fillna(positives)
        return positives
    
    def _sample_negative_examples(self, positives, negative_ratio=2, random_state=67):
        rng = np.random.default_rng(random_state)

        article_ids = self.product_features['article_id'].unique()
        customer_ids = positives['customer_id'].unique()

        positives_per_customer = positives.groupby('customer_id').size()

        negative_examples = []
        for customer in customer_ids:
            num_positives = positives_per_customer[customer]
            num_negatives = num_positives * negative_ratio

            bought = positives[positives['customer_id'] == customer]['article_id'].values
            buffer_size = min(num_negatives*2, len(article_ids))
            sampled_articles = rng.choice(article_ids, size=buffer_size, replace=False)
            sampled_articles = sampled_articles[~np.isin(sampled_articles, list(bought))][:num_negatives]
            
            negative_examples.extend(
                {
                    'customer_id': customer,
                    'article_id': product,
                    'purchased': 0
                }
                for product in sampled_articles
            )

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
        negatives = self.fillna(negatives)

        return negatives

    def fillna(self, df):
        if self.customer_fill_values:
            df = df.fillna(self.customer_fill_values)
        if self.product_fill_values:
            df = df.fillna(self.product_fill_values)
        return df
