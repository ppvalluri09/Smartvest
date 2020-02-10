import pandas as pd
import os
from smartvest.preprocessing.preprocessing import *
from smartvest.analytics.analysis import *

if __name__ == '__main__':
	path = str(input('Company logo: ')).upper()
	data = load_data(path)
	# plot_each_timeseries(data)
	data = clean_data(data, ['Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'])
	data = feature_scale(data)

	x_t = data[['Open', 'High', 'Low', 'Volume']].iloc[:-30, :]
	x_v = data[['Open', 'High', 'Low', 'Volume']].iloc[-30:, :]
	y_t = data[['Close']].iloc[30:, :]

	# model = train_nn(x_t.values, y_t.values)
	model = train_MLP(x_t.values, y_t.values)
	y_pred = model.predict(x_v)
	fig = plt.figure()
	plt.plot([i for i in range(y_t.shape[0])], y_t.values, c='b', label='Existing Values')
	plt.plot([y_t.shape[0] + i for i in range(len(y_pred))], y_pred, c='r', label='Predicted Values')
	plt.legend()
	plt.show()
