import pandas as pd
import numpy as np

class CustomerFeatureEngineer:
    def __init__(self, customers_df, transactions_df):
        self.customers = customers_df
        self.transactions = transactions_df
        self.set_dtypes()
    
    def calculate_rfm(self, as_of_date = None):
        if as_of_date == None:
            as_of_date == self.transactions['t_dat'].max()
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

    def set_dtypes(self):
        self.transactions['t_dat'] = pd.to_datetime(self.transactions['t_dat'])
