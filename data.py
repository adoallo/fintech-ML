from indicators import sma
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from datetime import datetime
import numpy as np
import pandas as pd

# Data preprocessing #

# We import the data
dataset = pd.read_csv('dataset.csv')
dataset = dataset[::-1]
price_list = dataset['Price']

# Adding SMAs columns and deleting change %
dataset = dataset.assign(sma50 = sma(50, price_list))
dataset = dataset.assign(sma20 = sma(20, price_list))
del dataset['Change %']

# Transforming the time series into a supervised learning dataset
dataset['t + 1']  = dataset['Price'].shift(-1)
dataset['t + 3']  = dataset['Price'].shift(-3)
dataset['t + 5']  = dataset['Price'].shift(-5)
dataset['t + 10'] = dataset['Price'].shift(-10)
dataset['sma50'].shift(1)
dataset['sma20 - 1'] = dataset['sma20'].shift(1)

# Getting our X and Y vectors
X = dataset.iloc[:, :-4]
Y = dataset.iloc[:, -6:-2]

# Splitting the date into 3 separate columns
date_list  = [datetime.strptime(X.values[i][0], '%b %d, %Y').date() for i in range(len(X))]
year_list  = [date_list[i].year for i in range(len(date_list))]
month_list = [date_list[i].month for i in range(len(date_list))]
day_list   = [date_list[i].day for i in range(len(date_list))]

X = X.assign(Year = year_list)
X = X.assign(Month = month_list)
X = X.assign(Day = day_list)

del X['Date']

# Adding SMA - 1 columns
X = X.assign(p_sma50 = dataset['sma50'].shift(1).values)
X = X.assign(p_sma20 = dataset['sma20'].shift(1).values)

# Reordering columns
column_titles = ['Year', 'Month', 'Day', 'Open', 'Low', 'High', 'p_sma20', 'p_sma50', 'sma20', 'sma50', 'Price']

X = X.reindex(columns = column_titles)

# Dealing with NaNs
X = X.dropna()
Y = Y.dropna()[26:]

# Is there a cross?
diff_list = [[round(X['p_sma50'][i] - X['p_sma20'][i], 3), round(X['sma50'][i] - X['sma20'][i], 3)] for i in range(len(X))]
def get_cross(sma_diff_list):
	for i in range(len(sma_diff_list)):
		if sma_diff_list[i][0] >= 0 and sma_diff_list[i][1] >= 0 or sma_diff_list[i][0] < 0 and sma_diff_list[i][1] < 0:  # No cross
			sma_diff_list[i] = 0
		elif sma_diff_list[i][0] >= 0 and sma_diff_list[i][1] < 0: # Bullish cross
			sma_diff_list[i] = 1
		elif sma_diff_list[i][0] < 0 and sma_diff_list[i][1] >= 0: # Bearish cross
			sma_diff_list[i] = -1

	return sma_diff_list

# We add the cross column to our vector X
X = X.assign(cross = get_cross(diff_list))

# Dividing dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size = 0.2, shuffle = False)

# Scaling data
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# Y_train = sc.fit_transform(Y_train)
# Y_test  = sc.transform(Y_test)

# Applying a linear regression
regression = LinearRegression()
regression.fit(X_train, Y_train)

Y_pred = regression.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))










