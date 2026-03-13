import pandas as pd
import numpy as np

class ProductFeatureEngineer:
    def __init__(self, articles_df, transactions_df):
        self.articles = articles_df
        self.transactions = transactions_df
        self.set_dtypes()
    
    def set_dtypes(self):
        self.transactions['t_dat'] = pd.to_datetime(self.transactions['t_dat'])
    
    def calculate_popularity(self, as_of_date=None):
        """
        - sales_last_7_days
        - sales_last_30_days
        - days_since_first_sale
        """
        if as_of_date==None:
            as_of_date = self.transactions['t_dat'].max()
        as_of_date = pd.to_datetime(as_of_date)

        relevant_txn = self.transactions[self.transactions['t_dat'] <= as_of_date]
        
        last_7_days_txn = relevant_txn[relevant_txn['t_dat'] >= (as_of_date - pd.Timedelta(days=7))]
        last_30_days_txn = relevant_txn[relevant_txn['t_dat'] >= (as_of_date - pd.Timedelta(days=30))]

        sales_7d = last_7_days_txn.groupby('article_id').size().reset_index(name='sales_last_7_days')
        sales_30d = last_30_days_txn.groupby('article_id').size().reset_index(name='sales_last_30_days')
        sale_dates = relevant_txn.groupby('article_id').agg(
            first_sale_date=('t_dat', 'min'),
            last_sale_date=('t_dat', 'max')
        ).reset_index()
        
        popularity = sale_dates.merge(sales_7d, on='article_id', how='left')
        popularity = popularity.merge(sales_30d, on='article_id', how='left')
        
        popularity['sales_last_7_days'] = popularity['sales_last_7_days'].fillna(0).astype(int)
        popularity['sales_last_30_days'] = popularity['sales_last_30_days'].fillna(0).astype(int)

        # fillna with 0, treat it like a brand new product with no history
        popularity['days_since_first_sale'] = (as_of_date - popularity['first_sale_date']).dt.days.fillna(0).astype(int)
        
        # fillna with max, treat it like an inactive product
        popularity['days_since_last_sale'] = (as_of_date - popularity['last_sale_date']).dt.days.astype(int)
        max_last_sale = popularity['days_since_last_sale'].max() + 1
        popularity['days_since_last_sale'] = popularity['days_since_last_sale'].fillna(max_last_sale)

        popularity = popularity.drop(['first_sale_date', 'last_sale_date'], axis=1)

        return popularity
    
    def calculate_price_features(self, as_of_date=None):
        """
        - avg_price
        - product_price_std
        - min_price
        - max_price
        """
        if as_of_date==None:
            as_of_date = self.transactions['t_dat'].max()
        as_of_date = pd.to_datetime(as_of_date)

        relevant_txn = self.transactions[self.transactions['t_dat'] <= as_of_date]
        
        agg_data = relevant_txn.groupby('article_id').agg(
            avg_price = ('price', 'mean'),
            min_price = ('price', 'min'),
            max_price = ('price', 'max'),
            product_price_std = ('price', 'std'),
        ).reset_index()


        global_avg_price = relevant_txn['price'].mean()
        agg_data['avg_price'] = agg_data['avg_price'].fillna(global_avg_price)
        agg_data['min_price'] = agg_data['min_price'].fillna(global_avg_price)
        agg_data['max_price'] = agg_data['max_price'].fillna(global_avg_price)
        agg_data['product_price_std'] = agg_data['product_price_std'].fillna(0)

        return agg_data

    def get_fill_values(self):
        return {
            'sales_last_7_days': 0,
            'sales_last_30_days': 0,
            'days_since_first_sale': 999,
            'days_since_last_sale': 999,
            'avg_price': 0,
            'min_price': 0,
            'max_price': 0,
            'product_price_std': 0
        }

    def calculate_all_features(self, as_of_date=None):
        """Combine popularity and price features"""
        popularity = self.calculate_popularity(as_of_date=as_of_date)
        price_features = self.calculate_price_features(as_of_date=as_of_date)
        
        all_features = popularity.merge(price_features, on='article_id', how='left')
        
        return all_features
