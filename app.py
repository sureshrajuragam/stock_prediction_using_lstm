import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
from keras.models import load_model
import streamlit as st

yf.pdr_override()
#ticker = ['AAPL']
start_date = dt.datetime(2010,1,1)
end_date = dt.datetime(2019,12,31)

st.title('Stock Price Prediction')

user_input = st.text_input('Enter stock  ticker','AAPL')
data = pdr.get_data_yahoo(user_input,start=start_date, end=end_date)

#Describing data
st.subheader('Data from 2010 to 2019')
st.write(data.describe())

#VISUALIZATION
st.subheader('Closing price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)


st.subheader('Closing price vs Time chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA & 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(data.Close,'b')
st.pyplot(fig)

#splitting data into training and testing
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])#start from 0 to 70% of tot vals(70% data is for training)
data_test = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))]) # 30% data is for testing

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_arr = scaler.fit_transform(data_train)

#training part deleted here as this model is trained already.
#(i.e; pre-trained model)
x_train = []
y_train = []

for i in range(100,data_train_arr.shape[0]):
    x_train.append(data_train_arr[i-100: i])
    y_train.append(data_train_arr[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

#Load model
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_train.tail(100)
final_data = past_100_days.append(data_test,ignore_index = True)
input_data = scaler.fit_transform(final_data)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original price')
plt.plot(y_predicted,'r',label = 'Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)