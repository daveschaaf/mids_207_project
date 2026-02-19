import pandas as pd
import numpy as np
from pathlib import Path

class CustomerFeatureEngineering:
    def __init__(self, customers_path, transactions_path):
        self.customers = pd.read_csv(customers_path)
        self.transactions = pd.read_csv(transactions_path)

