import pandas as pd
import statistics as st

# Moving averages
def sma(period, price_list):
	sma_list = [round(st.mean(price_list[i - period: i]), 3) for i in range(len(price_list)) if i >= period]

	return pd.Series(sma_list)
