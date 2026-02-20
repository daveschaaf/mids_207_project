import pandas as pd
import numpy as np

class CustomerFeatureEngineer:
    def __init__(self, customers_df, transactions_df):
        self.customers = customers_df
        self.transactions = transactions_df
        self.set_dtypes()
    
    def calculate_rfm(self, as_of_date = None):
        if as_of_date == None:
            as_of_date = self.transactions['t_dat'].max()
        as_of_date = pd.to_datetime(as_of_date)

        relevant_txn = self.transactions[self.transactions['t_dat'] <= as_of_date]
        rfm = relevant_txn.groupby('customer_id').agg(
            last_purchase_date = ('t_dat', 'max'),
            num_purchases = ('t_dat', 'count'),
            total_spent = ('price','sum')
        ).reset_index()

        rfm['days_since_last_purchase'] = (as_of_date - rfm['last_purchase_date']).dt.days
        rfm = rfm.drop('last_purchase_date', axis=1)
        return rfm
    def calculate_behavioral_features(self, as_of_date=None):
        if as_of_date == None:
            as_of_date = self.transactions['t_dat'].max()
        as_of_date = pd.to_datetime(as_of_date)

        relevant_txn = self.transactions[self.transactions['t_dat'] <= as_of_date]
        rfm = self.calculate_rfm(as_of_date=as_of_date)
        behaviors = relevant_txn.groupby('customer_id').agg(
            price_std = ('price', 'std'),
            first_purchase_date = ('t_dat', 'min'),
            last_purchase_date = ('t_dat', 'max')
        ).reset_index()

        behaviors['avg_transaction_value'] = rfm['total_spent'] / rfm['num_purchases']
        behaviors['avg_days_between_purchases'] = (behaviors['last_purchase_date'] - behaviors['first_purchase_date']).dt.days / (rfm['num_purchases']-1)
        
        behaviors = behaviors.drop('first_purchase_date', axis=1)
        behaviors = behaviors.drop('last_purchase_date', axis=1)

        return behaviors


    def set_dtypes(self):
        self.transactions['t_dat'] = pd.to_datetime(self.transactions['t_dat'])
