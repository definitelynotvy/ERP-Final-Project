import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import datetime as dt
import yfinance as yf
import pandas_ta as ta
from plotly.subplots import make_subplots
from datetime import timedelta
from datetime import date
import datetime
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import requests
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import xgboost as xgb

start = '2017-01-01'
end = date.today()

st.title('Bitcoin Price Prediction')

user_input = st.text_input('Enter Bitcoin Ticker', 'BTC-USD')

# stock_info = yf.Ticker(user_input).info
# # stock_info.keys() for other properties you can explore
# company_name = stock_info['shortName']
# st.subheader(company_name)
# market_price = stock_info['regularMarketOpen']
# previous_close_price = stock_info['regularMarketPreviousClose']

#api
url = 'https://query1.finance.yahoo.com/v8/finance/chart/{}?&interval=1d'.format(user_input)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.79'}
response = requests.get(url, headers=headers)

meta = response.json()['chart']['result'][0]
previous_close_price = meta['meta']['chartPreviousClose']
market_price = meta['meta']['regularMarketPrice']


st.write('Regular Market Price : ', market_price)
st.write('Previous Close Price : ', previous_close_price)

#df = data.DataReader(user_input, 'yahoo', start, end)

#new dataframe
from pandas_datareader import data as pdr

yf.pdr_override() # <== that's all it takes :-)

# download dataframe
df = pdr.get_data_yahoo(user_input, start=start, end=end+ datetime.timedelta(days=1))
df_temp = pdr.get_data_yahoo("BTC-USD", start=start, end=end+ datetime.timedelta(days=1))
#data for prophet
df2 = pdr.get_data_yahoo("BTC-USD", start=start, end=end+ timedelta(days=1))
#data for assemble learning
data = pdr.get_data_yahoo("BTC-USD", start=start, end=end+ timedelta(days=1))
# describing data

st.subheader('Data from 2017-2023')
#df= df.reset_index()

st.dataframe(df.tail(10).style.set_properties(**{'background-color': '#D7BBF5'}, subset=['Close']))
# st.write(df.tail(10))
st.write(df.describe())
# rows = st.columns(2)
# rows[0].markdown("### Test1")
# rows[0].dataframe(df.tail(10))
# rows[1].markdown("### Test2")
# rows[1].dataframe(df.describe())


# Force lowercase (optional)
df.columns = [x.lower() for x in df.columns]

#candlestick chart
df= df.reset_index()
df_cd = df.tail(60)
# st.write(df.columns)
# st.line_chart(df_temp['Close'])
fig = go.Figure(data=go.Scatter(x=df_temp.index, y=df_temp['Close'], mode='lines'))

# Customize the chart layout
fig.update_layout(title='BTC Crypto Price', xaxis_title='Date', yaxis_title='Price')

# Render the chart in Streamlit
st.plotly_chart(fig)

d = st.slider('Choose day', value=30,min_value=7, max_value=60, step=1)
df_cd1 = df.tail(d)
fig = go.Figure(data=[go.Candlestick(
							x=df_cd1['Date'],
                			open=df_cd1['open'], 
							high=df_cd1['high'],
                			low=df_cd1['low'], 
							close=df_cd1['close'],
							increasing_line_color = 'green',
							decreasing_line_color = 'red')])

fig.update_layout(xaxis_rangeslider_visible=False)
# fig.show()
st.plotly_chart(fig)

st.subheader('Technical Analysis')
infoType = st.radio(
        "Choose Technical Analysis Type",
        ('Moving Average Chart', 'Market trend', 'RSI & CCI', 'Williams %R', 'Stochastic Oscillator')
    )
if infoType == 'Moving Average Chart':
	st.header('Moving Average Chart')
	st.write('Charted moving average allow trends to be visually evaluated faster than looking at price alone, as they smooth out random daily price fluctuations.')
	st.subheader('Closing Price vs Time Chart with 100 MA')
	
	ma100 = df.close.rolling(100).mean()
	fig = plt.figure(figsize = (12, 6))
	plt.plot(ma100, alpha = 0.5, label = 'MA100')
	plt.plot(df.close, alpha = 0.5, label = 'Close')
	plt.legend(loc = 'upper left')
	st.pyplot(fig)

	st.subheader('Closing Price vs Time Chart with 100 MA & 200MA')
	#signal
    

	ma100 = df.close.rolling(100).mean()
	ma200 = df.close.rolling(200).mean()
	df['Signal'] = np.where(ma100 > ma200, 1, 0)
	df['Position'] = df['Signal'].diff()
	df['Buy'] = np.where(df['Position'] == 1, df.close, np.NAN)
	df['Sell'] = np.where(df['Position'] == -1, df.close, np.NAN)

    
	fig = plt.figure(figsize = (12, 6))
	plt.plot(ma100, alpha = 0.5, label = 'MA100', color='g')
	plt.plot(ma200, alpha = 0.5, label = 'MA200', color='r')
	plt.plot(df.close, alpha = 0.5, label = 'Close', color='b')
	plt.scatter(df.index, df['Buy'], marker = '^', alpha=1, color = 'green', label = 'BUY SIGNAL')
	plt.scatter(df.index, df['Sell'], marker = 'v', alpha=1, color = 'red', label = 'SELL SIGNAL')
	plt.legend(loc = 'upper left')
	st.pyplot(fig)

elif infoType == 'Stochastic Oscillator':
	st.header('Stochastic Oscillator')
	st.write('This oscillator compares the closing price of a cryptocurrency or another security against the price spectrum over a specific time period. It is beneficial if you want to have uptrend or downtrend signals or even generate overbought and oversold trading signals.')
	st.markdown(':green[if %K > %D, then that is a buy signal.]')
	st.markdown(':green[if %K < %D, then that is a sell signal.]')
	st.markdown('_Remember, that these signals are indicators not predictors. If we look at the graph, there are a ton of signals related to buy and sell, so you have to have other factors to have a more reliable approach of when to buy and sell._')
	# st.subheader('Stochastic Oscillator')

	#short dataframe
	# pd.set_option('mode.chained_assignment', None)
	df2 = pdr.get_data_yahoo("BTC-USD", start='2022-1-1', end=end+ datetime.timedelta(days=1))

	# Find minimum of 14 consecutive values by rolling function
	df2['14-low'] = df2['Low'].rolling(14).min()
	df2['14-high'] = df2['High'].rolling(14).max()

	# Apply the formula
	df2['%K'] = (df2['Close'] -df2['14-low'] )*100/(df2['14-high'] -df2['14-low'] )
	df2['%D'] = df2['%K'].rolling(3).mean()
	fig, ax = plt.subplots(figsize=(12, 6))

    # Plot %K and %D
	df2[['%K', '%D']].plot(ax=ax)
    
    # Plot Adj Close on secondary y-axis
	df2['Close'].plot(ax=ax, secondary_y=True, label='Close')
    
    # Add horizontal lines
	ax.axhline(20, linestyle='--', color="r")
	ax.axhline(80, linestyle="--", color="r")
    
    # Set labels and title
	ax.set_xlabel('Date')
	ax.set_ylabel('Percentage')
	ax.right_ax.set_ylabel('Close Price')
	ax.set_title('Stochastic Oscillator')
	lines, labels = ax.get_legend_handles_labels()
	lines2, labels2 = ax.right_ax.get_legend_handles_labels()
	ax.legend(lines + lines2, labels + labels2, loc='upper left')
    # Display the chart
	st.pyplot(fig)



elif infoType == 'RSI & CCI':
	st.subheader('Relative Strength Index (RSI) & Comodity Channel Index (CCI)')

	df3 = pdr.get_data_yahoo("BTC-USD", start='2022-1-1', end=end+ datetime.timedelta(days=1))
	df3["RSI(2)"]= ta.rsi(df3['Close'], length= 2)
	df3["RSI(7)"]= ta.rsi(df3['Close'], length= 7)
	df3["RSI(14)"]= ta.rsi(df3['Close'], length= 14)
	df3["CCI(30)"]= ta.cci(close=df3['Close'],length=30, high= df3["High"], low =  df3["Low"])
	df3["CCI(50)"]= ta.cci(close= df3['Close'],length= 50, high= df3["High"], low =  df3["Low"])
	df3["CCI(100)"]= ta.cci(close= df3['Close'],length= 100, high= df3["High"], low =  df3["Low"])

	fig3=plt.figure(figsize=(15,15))
	ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
	ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
	ax1.plot( df3['Close'], linewidth = 2.5)
	ax1.set_title('CLOSE PRICE')
	ax2.plot(df3['RSI(14)'], color = 'orange', linewidth = 2.5)
	ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
	ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
	ax2.set_title('RELATIVE STRENGTH INDEX')
	st.pyplot(fig3)



	fig4= plt.figure(figsize=(15,15))
	ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
	ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
	ax1.plot(df3['Close'], linewidth = 2.5)
	ax1.set_title('CLOSE PRICE')
	ax2.plot(df3['CCI(30)'], color = 'orange', linewidth = 2.5)
	ax2.axhline(-100, linestyle = '--', linewidth = 1.5, color = 'grey')
	ax2.axhline(100, linestyle = '--', linewidth = 1.5, color = 'grey')
	ax2.set_title('COMMODITY CHANNEL INDEX')
	st.pyplot(fig4)

elif infoType == 'Williams %R':
	st.subheader('Williams %R')
	def get_wr(high, low, close, lookback):
		highh = high.rolling(lookback).max()
		lowl = low.rolling(lookback).min()
		wr = -100 * ((highh - close) / (highh - lowl))
		return wr
        
    	
	df['wr_14'] = get_wr(df['high'], df['low'], df['close'], 14)

	fig5= plt.figure(figsize=(15,12))
	ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
	ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
	ax1.plot(df['close'], linewidth = 2)
	ax1.set_title('CLOSING PRICE')
	ax2.plot(df['wr_14'], color = 'orange', linewidth = 2)
	ax2.axhline(-20, linewidth = 1.5, linestyle = '--', color = 'grey')
	ax2.axhline(-50, linewidth = 1.5, linestyle = '--', color = 'green')
	ax2.axhline(-80, linewidth = 1.5, linestyle = '--', color = 'grey')
	ax2.set_title('WILLIAMS %R 14')
	st.pyplot(fig5)
else:
        start = dt.datetime.today() - dt.timedelta(2 * 365)
        end = dt.datetime.today()
        df4 = yf.download(user_input, start, end)
        # df = df.reset_index()
        fig = go.Figure(
            data=go.Scatter(x=df4.index, y=df4['Adj Close'])
        )
        fig.update_layout(
            title={
                'text': "Bitcoin Prices Over Past Two Years",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig, use_container_width=True)
	
#deploy model lstm and prophet
st.subheader("Prediction of Crypto Price")

# splitting date into training and testing 
# data_training= pd.DataFrame(df['close'][0:int(len(df)*0.70)])
# data_testing = pd.DataFrame(df['close'][int(len(df)*0.70): int(len(df))])

# # print("training data: ",data_training.shape)
# # print("testing data: ", data_testing.shape)


# # scaling of data using min max scaler (0,1)
# # from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

# data_training_array = scaler.fit_transform(data_training)


# #Load model 
# model = load_model("LSTM_test.h5")

# #testing part
# past_100_days = data_training.tail(100)

# # final_df= past_100_days.append(data_testing, ignore_index =True)
# final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# input_data = scaler.fit_transform(final_df)


# x_test = []
# y_test = []

# for i in range (100, input_data.shape[0]):
#     x_test.append(input_data[i-100 : i])
#     y_test.append(input_data[i, 0])
# x_test, y_test = np.array(x_test), np.array(y_test)    


# y_predicted = model.predict(x_test)

# scaler = scaler.scale_

# scale_factor = 1/scaler[0]

# y_predicted = y_predicted * scale_factor

# y_test = y_test* scale_factor


# final Graph
# st.subheader("Predictions vs Original")
# fig2= plt.figure(figsize = (12,6))
# plt.plot(y_test, 'b', label = 'Original Price')
# plt.plot(y_predicted, 'r', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)

# predict by date

st.subheader('Crypto Price Prediction by Date')

df1=df.reset_index()['close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
#datemax="24/06/2022"
datemax=dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
x_input=df1[:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


date1 = st.date_input("Enter Date in this format yyyy-mm-dd")
option = st.selectbox(
    'Select the model that you want to use for prediction!',
    ('LSTM', 'Prophet', 'RandomForestRegressor + XGBoost'))
result = st.button("Predict")

#st.write(result)
if option == 'LSTM':
	if result:
		#scale
		data_training= pd.DataFrame(df['close'][0:int(len(df)*0.70)])
		data_testing = pd.DataFrame(df['close'][int(len(df)*0.70): int(len(df))])

		# print("training data: ",data_training.shape)
		# print("testing data: ", data_testing.shape)


		# scaling of data using min max scaler (0,1)
		# from sklearn.preprocessing import MinMaxScaler
		scaler1 = MinMaxScaler(feature_range=(0,1))

		data_training_array = scaler1.fit_transform(data_training)


		#Load model 
		model = load_model("LSTM_test.h5")

		#testing part
		past_100_days = data_training.tail(100)

		# final_df= past_100_days.append(data_testing, ignore_index =True)
		final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

		input_data = scaler1.fit_transform(final_df)


		x_test = []
		y_test = []

		for i in range (100, input_data.shape[0]):
			x_test.append(input_data[i-100 : i])
			y_test.append(input_data[i, 0])
		x_test, y_test = np.array(x_test), np.array(y_test)    


		y_predicted = model.predict(x_test)

		scaler1 = scaler1.scale_

		scale_factor = 1/scaler1[0]

		y_predicted = y_predicted * scale_factor

		y_test = y_test* scale_factor
		#---end-----
		from datetime import datetime
		my_time = datetime.min.time()
		date1 = datetime.combine(date1, my_time)
		#date1=str(date1)
		#date1=dt.datetime.strptime(time_str,"%Y-%m-%d")

		nDay=date1-datemax
		nDay=nDay.days

		date_rng = pd.date_range(start=datemax, end=date1, freq='D')
		date_rng=date_rng[1:date_rng.size]
		lst_output=[]
		n_steps=x_input.shape[1]
		i=0

		while(i<=nDay):
			if(len(temp_input)>n_steps):
				x_input=np.array(temp_input[1:])
				print("{} day input {}".format(i,x_input))
				x_input=x_input.reshape(1,-1)
				x_input = x_input.reshape((1, n_steps, 1))
				#print(x_input)
				yhat = model.predict(x_input, verbose=0)
				print("{} day output {}".format(i,yhat))
				temp_input.extend(yhat[0].tolist())
				temp_input=temp_input[1:]
				#print(temp_input)
				lst_output.extend(yhat.tolist())
				i=i+1
			else:
				x_input = x_input.reshape((1, n_steps,1))
				yhat = model.predict(x_input, verbose=0)
				print(yhat[0])
				temp_input.extend(yhat[0].tolist())
				print(len(temp_input))
				lst_output.extend(yhat.tolist())
				i=i+1
		res =scaler.inverse_transform(lst_output)
	#output = res[nDay-1]

		output = res[nDay]

		st.write("*Predicted Price for Date :*", date1, "*is*", np.round(output[0], 2))
		st.success('The Price is {}'.format(np.round(output[0], 2)))

		#st.write("predicted price : ",output)

		predictions=res[res.size-nDay:res.size]
		# print(predictions.shape)
		predictions=predictions.ravel()
		# print(type(predictions))
		# print(date_rng)
		# print(predictions)
		# print(date_rng.shape)

		@st.cache_data
		def convert_df(df):
			return df.to_csv().encode('utf-8')
		df = pd.DataFrame(data = date_rng)
		df['Predictions'] = predictions.tolist()
		df.columns =['Date','Price']
		st.write(df)
		csv = convert_df(df)
		# st.download_button(
		# 	"Press to Download",
		# 	csv,
		# 	"file.csv",
		# 	"text/csv",
		# 	key='download-csv'
		# )
	
elif option == 'Prophet':
	if result:
		#count day
		from datetime import datetime
		my_time = datetime.min.time()
		date1 = datetime.combine(date1, my_time)
		nDay=date1-datemax
		nDay=nDay.days
		#---
		# df2 = pdr.get_data_yahoo("BTC-USD", start=start, end=end+ timedelta(days=1))
		df2 = df2.tail(100).reset_index()
		new_df = df2[['Date', 'Close']]
		new_df = new_df.rename(columns={'Date':'ds', 'Close':'y'})
		#new_df["ds"] = pd.to_datetime(new_df["ds"])
		final_df = new_df.tail(1)
		
		
		# initialize prophet model
		# fp = Prophet(daily_seasonality=True)
		fp = Prophet()
		fp.fit(new_df)

		#make future predictions
		future = fp.make_future_dataframe(periods=nDay-1)
		forecast = fp.predict(future)

		predicted_prices = forecast[['ds', 'yhat']].tail(nDay-1)
		
		final_df = pd.concat([final_df.rename(columns={'ds':'Date','y':'Price'}), predicted_prices.rename(columns={'ds':'Date','yhat':'Price'})], ignore_index=True)
		# Print the predicted prices
		st.write("*Predicted Price for Date :*", date1, "*is*", np.round(predicted_prices['yhat'].iloc[-1], 2))
		st.success('The Price is {}'.format(np.round(predicted_prices['yhat'].iloc[-1], 2)))
		st.write(final_df)

elif option == 'RandomForestRegressor + XGBoost':
	
	data['Date'] = data.index
	data['Year'] = data['Date'].dt.year
	data['Month'] = data['Date'].dt.month
	data['Day'] = data['Date'].dt.day
	features = ['Year', 'Month', 'Day']
	target = 'Close'
	X = data[features]
	y = data[target]

	rf_model = RandomForestRegressor()
	xgb_model = xgb.XGBRegressor()

	# Initialize the VotingRegressor and specify the individual models
	voting_model = VotingRegressor(estimators=[('rf', rf_model), ('xgb', xgb_model)])

	# Train the model using the entire data
	voting_model.fit(X, y)

	last_date = data.index[-2]

	#count day
	from datetime import datetime
	my_time = datetime.min.time()
	date1 = datetime.combine(date1, my_time)
	nDay=date1-datemax
	nDay=nDay.days
	# Generate the next 10 dates
	next_dates = [last_date + timedelta(days=i) for i in range(1, nDay+1)]

	# Preprocess the next dates and extract features
	next_features = pd.DataFrame({'Date': next_dates})
	next_features['Year'] = next_features['Date'].dt.year
	next_features['Month'] = next_features['Date'].dt.month
	next_features['Day'] = next_features['Date'].dt.day
	next_features = next_features[features]

	# Predict the Bitcoin prices for the next 10 days
	predictions = voting_model.predict(next_features)
	final = pd.DataFrame({'Date': next_dates, 'Price':predictions })
	st.write("*Predicted Price for Date :*", date1, "*is*", np.round(predictions[-1], 2))
	st.success('The Price is {}'.format(np.round(predictions[-1], 2)))
	st.write(final.round(2))
	
