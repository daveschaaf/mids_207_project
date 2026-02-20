import pandas as pd
import numpy as np

class CustomerFeatureEngineer:
    def __init__(self, customers_df, transactions_df):
        self.customers = customers_df
        self.transactions = transactions_df
        self.set_dtypes()
    
    def calculate_rfm_and_behaviors(self, as_of_date = None):
        """
        days_since_last_purchase
        num_purchases
        total_spent
        avg_transaction_value
        price_std
        avg_days_between purchases
        """
        if as_of_date == None:
            as_of_date = self.transactions['t_dat'].max()
        as_of_date = pd.to_datetime(as_of_date)

        relevant_txn = self.transactions[self.transactions['t_dat'] <= as_of_date]

        agg_data = relevant_txn.groupby('customer_id').agg(
            first_purchase_date = ('t_dat', 'min'),
            last_purchase_date = ('t_dat', 'max'),
            price_std = ('price', 'std'),
            num_purchases = ('t_dat', 'count'),
            total_spent = ('price','sum')
        ).reset_index()

        agg_data['days_since_last_purchase'] = (as_of_date - agg_data['last_purchase_date']).dt.days
        agg_data['avg_transaction_value'] = agg_data['total_spent'] / agg_data['num_purchases']
        agg_data['avg_days_between_purchases'] = (agg_data['last_purchase_date'] - agg_data['first_purchase_date']).dt.days / (agg_data['num_purchases']-1)
        
        agg_data = agg_data.drop('last_purchase_date', axis=1)
        agg_data = agg_data.drop('first_purchase_date', axis=1)
        agg_data = agg_data.rename(columns={'price_std': 'customer_price_std'})

        return agg_data


    def calculate_category_preferences(self, articles_df, as_of_date=None):
        """
        favorite_department
        favorite_garment_group
        category_diversity
        """

        def category_mode(category):
            return category.mode()[0] if not category.mode().empty else None

        if as_of_date == None:
            as_of_date = self.transactions['t_dat'].max()
        as_of_date = pd.to_datetime(as_of_date)

        
        relevant_txn = self.transactions[self.transactions['t_dat'] <= as_of_date]
        txn_with_category = relevant_txn.merge(
            articles_df[['article_id','department_name', 'garment_group_name']],
            on = 'article_id',
            how = 'left'
        )
        category_df = txn_with_category.groupby('customer_id').agg(
            primary_department = ('department_name', category_mode),
            primary_garment_group = ('garment_group_name', category_mode),
            category_diversity = ('department_name', 'nunique')
        ).reset_index()

        return category_df
    def calculate_all_features(self, articles_df, as_of_date=None):
        rfm_behaviors = self.calculate_rfm_and_behaviors(as_of_date)
        categories = self.calculate_category_preferences(articles_df, as_of_date)

        all_features = rfm_behaviors.merge(
            categories,
            on='customer_id',
            how = 'left'
        ).reset_index()

        return all_features

        

    def set_dtypes(self):
        self.transactions['t_dat'] = pd.to_datetime(self.transactions['t_dat'])
