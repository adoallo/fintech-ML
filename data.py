from indicators import sma
import numpy as np
import pandas as pd

# We import the data
dataset = pd.read_csv('dataset.csv')
dataset = dataset[::-1]
price_list = dataset['Price']

# We add two more columns
dataset = dataset.assign(sma50 = sma(50, price_list))
dataset = dataset.assign(sma20 = sma(20, price_list))

print(sma(20, price_list))