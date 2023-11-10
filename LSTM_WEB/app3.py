import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

key = "16cab08f9e458fb48a94b1f7646af8ce37f4dd9c"
start = '2018-01-01'
end = '2019-12-31'
st.title('Stock Trend Prediction')
userinput = st.text_input('Enter stock ticker', 'AAPL')

# Uncomment the following lines if you want to fetch data using Tiingo API
# df = data.get_data_tiingo('AAPL', api_key=key)
# df.to_csv(userinput + '.csv', index=False)

# Read data from the CSV file
df = pd.read_csv(userinput + '.csv')

# Describing header
st.subheader("Data from 2015 - 2023")
st.write(df.describe())
st.subheader('Closing Price vs Time chart')

fig = plt.figure(figsize=(12, 6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.close)
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200Ma')
ma200 = df.close.rolling(200).mean()

fig = plt.figure(figsize=(12, 6))
plt.plot(df.close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

data_training = pd.DataFrame(df['close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df) * 0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

model = load_model('kerasmodel.h5')
past_100_days = data_training.tail(100)
final_df = pd.concat([data_testing, past_100_days], ignore_index=True)
inputdata = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, inputdata.shape[0]):
    x_test.append(inputdata[i - 100: i])
    y_test.append(inputdata[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Orginal Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


