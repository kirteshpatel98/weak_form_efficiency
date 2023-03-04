#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
import yfinance as yf


# In[2]:


import pandas_datareader as pdr
import os


# In[3]:


#nifty = pdr.get_data_tiingo("INDF", api_key = "30f0d443a2a115c661a899929861dccfa07e7478" )
nifty = yf.download('^NSEI',start="2018-10-01", end="2022-09-30",interval='1d')


# In[4]:


nifty.to_csv("^NSEI.csv")


# In[5]:


nifty = pd.read_csv("^NSEI.csv")
nifty


# In[6]:


nifty.tail()


# In[7]:


nifty_1 = nifty.reset_index()["Close"]
nifty_1


# In[8]:


plt.plot(nifty_1)


# In[9]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1) )
nifty_2 = scaler.fit_transform(np.array(nifty_1).reshape(-1,1))


# In[10]:


plt.figure(figsize = (20,8))
plt.plot(nifty_2)


# In[142]:


train_size = int(len(nifty_2)*0.8)
test_size = len(nifty_2)-train_size
train_size,test_size


# In[143]:


train_data,test_data = nifty_2[0:train_size,:],nifty_2[train_size:,:]


# In[144]:


train_data.shape,test_data.shape


# In[145]:


def creat_dataset(data, time_stamp):
    data_x, data_y = [], []
    for i in range(len(data)-time_stamp-1):
        a = data[i: i + time_stamp, 0]
        data_x.append(a)
        b = data[i+time_stamp,0]
        data_y.append(b)
    return data_x, data_y
                


# In[146]:


x_train, y_train = creat_dataset(train_data,30)
x_test, y_test = creat_dataset(test_data,30)


# In[147]:


x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
x_train,x_test


# In[148]:


import seaborn as sns 
from sklearn.preprocessing import Normalizer
normalizedx_train = Normalizer().fit_transform(x_train)

normalizedx_train = pd.DataFrame(normalizedx_train)
normalizedx_train


# In[149]:


# plot correlation heatmap
plt.figure(figsize = (20,10))
sns.heatmap(normalizedx_train.corr(), annot = True)


# In[150]:


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)


# In[151]:


x_train.shape, x_test.shape


# In[152]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[153]:


model_1 = Sequential()
model_1.add(LSTM(50, return_sequences = True, input_shape = (30,1)))
model_1.add(LSTM(50, return_sequences= True))
model_1.add(LSTM(50))
model_1.add(Dense(1))
model_1.compile(loss= "mean_squared_error", optimizer = "adam")


# In[154]:


model_1.summary()


# In[155]:


model_1.fit(x_train, y_train, validation_data= (x_test, y_test), epochs= 50, batch_size = 64, verbose = 1)


# In[156]:


train_pred = model_1.predict(x_train)
test_pred = model_1.predict(x_test)
train_pred, test_pred


# In[157]:


train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)
train_pred, test_pred


# In[158]:


import math
from sklearn.metrics import mean_squared_error


# In[159]:


math.sqrt(mean_squared_error(y_train, train_pred))


# In[160]:


math.sqrt(mean_squared_error(y_test, test_pred))


# In[161]:


import numpy
### Plotting 
# shift train predictions for plotting
look_back = 30
trainPredictPlot = numpy.empty_like(nifty_2)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_pred)+look_back, :] = train_pred


# In[162]:


# shift test predictions for plotting
testPredictPlot = numpy.empty_like(nifty_2)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_pred)+(look_back*2)+1:len(nifty_2)-1, :] = test_pred


# In[163]:


# plot baseline and predictions
plt.figure(figsize = (18,10))
plt.plot(scaler.inverse_transform(nifty_2))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[164]:


len(train_data),len(test_data)


# In[165]:


x_input = test_data[167:].reshape(1,-1)
x_input.shape


# In[166]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[167]:


temp_input


# In[168]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=30
i=0
while(i<5):
    
    if(len(temp_input)>30):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model_1.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model_1.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[169]:


day_new=np.arange(1,31)
day_pred=np.arange(31,36)


# In[170]:


import matplotlib.pyplot as plt


# In[171]:


len(nifty_2)


# In[172]:


plt.plot(day_new,scaler.inverse_transform(nifty_2[955:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[173]:


nifty_3=nifty_2.tolist()
nifty_3.extend(lst_output)
plt.plot(nifty_3[955:])


# In[174]:


check_nifty = yf.download('^NSEI',start="2022-09-01", end="2022-10-05",interval='1d')


# In[175]:


check_nifty_1 = check_nifty.reset_index()["Close"]
check_nifty_1


# In[176]:


plt.plot(check_nifty_1)


# # Prediction for next 10 days

# In[177]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output_10=[]
n_steps=30
i=0
while(i<10):
    
    if(len(temp_input)>30):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model_1.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output_10.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model_1.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output_10.extend(yhat.tolist())
        i=i+1
    

print(lst_output_10)


# In[178]:


day_new_10=np.arange(1,31)
day_pred_10=np.arange(31,41)


# In[179]:


plt.plot(day_new_10,scaler.inverse_transform(nifty_2[955:]))
plt.plot(day_pred_10,scaler.inverse_transform(lst_output_10))


# In[180]:


plt.plot(check_nifty_1)


# # German Market

# In[181]:


sdax = yf.download('^SDAXI',start="2018-10-01", end="2022-09-30",interval='1d')


# In[182]:


sdax = sdax.reset_index()["Close"]


# In[183]:


sdax


# In[184]:


plt.figure(figsize = (18, 8))
plt.plot(sdax[900:])


# In[187]:


sdax_1 = scaler.fit_transform(np.array(sdax).reshape(-1,1))


# In[188]:


sdax_1


# In[190]:


train_len = int(len(sdax_1)*0.8)
test_len = 1- train_len
train_data, test_data = sdax_1[:train_len], sdax_1[train_len:]
train_data.shape, test_data.shape


# In[193]:


def creat_dataset(data, time_stamp):
    data_x, data_y = [], []
    for i in range(len(data)-time_stamp-1):
        a = data[i: i + time_stamp, 0]
        data_x.append(a)
        b = data[i+time_stamp,0]
        data_y.append(b)
    return data_x, data_y


# In[191]:


x_train_1, y_train_1 = creat_dataset(train_data,30)


# In[192]:


x_test_1, y_test_1 = creat_dataset(test_data,30)


# In[197]:


x_train_1, y_train_1, x_test_1, y_test_1


# In[198]:


x_train_1, y_train_1 = np.asarray(x_train_1), np.asarray(y_train_1)
x_test_1, y_test_1 = np.asarray(x_test_1), np.asarray(y_test_1)
x_train_1, x_test_1


# In[199]:


x_train_1.shape, x_test_1.shape


# In[203]:


x_train_1 = x_train_1.reshape(x_train_1.shape[0],x_train_1.shape[1],1)
x_test_1 = x_test_1.reshape(x_test_1.shape[0],x_test_1.shape[1],1)
x_train_1.shape, x_test_1.shape


# In[204]:


model_1.fit(x_train_1, y_train_1, validation_data= (x_test_1, y_test_1), epochs= 50, batch_size = 64, verbose = 1)


# In[213]:


x_test_1.shape


# In[214]:


x_input_sdax = x_test_1[142:,0].reshape(1,-1)
x_input_sdax.shape


# In[215]:


temp_input_sdax = list(x_input_sdax)


# In[226]:


# demonstrate prediction for next 2 days
from numpy import array

lst_output_sdax=[]
n_steps=30
i=0
while(i<2):
    
    if(len(temp_input_sdax)>30):
        #print(temp_input)
        x_input_sdax=np.array(temp_input_sdax[1:])
        print("{} day input {}".format(i,x_input_sdax))
        x_input_sdax=x_input_sdax.reshape(1,-1)
        x_input_sdax = x_input_sdax.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model_1.predict(x_input_sdax, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input_sdax.extend(yhat[0].tolist())
        temp_input_sdax=temp_input_sdax[1:]
        #print(temp_input)
        lst_output_sdax.extend(yhat.tolist())
        i=i+1
    else:
        x_input_sdax = x_input_sdax.reshape((1, n_steps,1))
        yhat = model_1.predict(x_input_sdax, verbose=0)
        print(yhat[0])
        temp_input_sdax.extend(yhat[0].tolist())
        print(len(temp_input_sdax))
        lst_output_sdax.extend(yhat.tolist())
        i=i+1
    

print(lst_output_sdax)


# In[227]:


day_new_sdax = np.arange(1,31)
day_pred_sdax = np.arange(31,33)


# In[228]:


len(sdax_1)


# In[229]:


plt.plot(day_new_sdax,scaler.inverse_transform(sdax_1[983:]))
plt.plot(day_pred_sdax,scaler.inverse_transform(lst_output_sdax))


# In[223]:


sdax_check = yf.download('^SDAXI',start="2022-09-01", end="2022-10-07",interval='1d')


# In[224]:


sdax_check = sdax_check.reset_index()["Close"]


# In[225]:


plt.plot(sdax_check)


# In[234]:


x_train_1.shape


# In[236]:


x_train_check = x_train_1.reshape(x_train_1.shape[0],x_train_1.shape[1])
x_train_check.shape


# In[237]:


normalizedx_train_sdax = Normalizer().fit_transform(x_train_check)

normalizedx_train_sdax = pd.DataFrame(normalizedx_train_sdax)
normalizedx_train_sdax


# In[238]:


plt.figure(figsize = (20,10))
sns.heatmap(normalizedx_train_sdax.corr(), annot = True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


sp500 = yf.download('^NSEI',start="2018-10-01", end="2022-09-30",interval='1d')


# In[4]:


sp500.head()


# In[5]:


# Clean the data and convert to daily returns
sp500["returns"] = np.log(sp500["Close"]/sp500["Close"].shift(1))
sp500 = sp500.dropna()


# In[6]:


# Calculate the moving average of the stock prices
window_size = 50
sp500["ma"] = sp500["Close"].rolling(window=window_size).mean()

# Plot the moving average and the stock prices
plt.figure(figsize=(12, 8))
plt.plot(sp500["Close"], label="Stock Prices")
plt.plot(sp500["ma"], label="Moving Average")
plt.xlabel("Year")
plt.ylabel("Price")
plt.title("S&P 500 Moving Average")
plt.legend()
plt.show()


# In[7]:


# Calculate EWMA
sp500["EWMA"] = sp500["Close"].ewm(span=30, adjust=False).mean()

# Plot the original closing prices and the EWMA
plt.figure(figsize=(15, 8))
plt.plot(sp500["Close"], label="Closing Prices")
plt.plot(sp500["EWMA"], label="EWMA")
plt.plot(sp500["ma"], label="Moving Average")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Price")
plt.title("S&P 500 Closing Prices and EWMA")
plt.show()


# In[8]:


result = adfuller(sp500["Close"])
print("Augmented Dickey-Fuller Test Results")
print("Test Statistic: {:.3f}".format(result[0]))
print("p-value: {:.3f}".format(result[1]))
print("Critical Values:")
for key, value in result[4].items():
    print("\t{}: {:.3f}".format(key, value))
    
if result[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is not stationary.")


# # Deep Learning

# In[9]:


df = sp500.reset_index()["Close"]
df


# In[10]:


plt.plot(df)


# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
df = scaler.fit_transform(np.array(df).reshape(-1,1))


# In[12]:


train_size = int(len(df)*0.65)
test_size = len(df) - train_size
train_data,test_data=df[0:train_size,:],df[train_size:len(df),:1]


# In[13]:


train_size,test_size


# In[14]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[15]:


import numpy
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[16]:


print(X_train.shape), print(y_train.shape)


# In[17]:


print(X_test.shape), print(ytest.shape)


# In[18]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[20]:


model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (100,1)))
model.add(LSTM(50, return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = "mean_squared_error", optimizer = "adam")


# In[21]:


model.summary()


# In[22]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[23]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[24]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[25]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[26]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[27]:


### Plotting 
# shift train predictions for plotting
plt.figure(figsize=(12, 8))
look_back=100
trainPredictPlot = numpy.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[28]:


len(test_data)


# In[29]:


x_input=test_data[253:].reshape(1,-1)
x_input.shape


# In[30]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[38]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
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
    

print(lst_output)


# In[39]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[43]:


import matplotlib.pyplot as plt


# In[44]:


len(df)


# In[45]:


plt.plot(day_new,scaler.inverse_transform(df[906:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[50]:


sp_pred = yf.download('^GSPC',start="2018-10-01", end="2022-10-30",interval='1d')
sp_pred


# In[51]:


df1 = sp_pred.reset_index()["Close"]


# In[53]:


plt.figure(figsize = (15,6))
plt.plot(df1)


# In[ ]:





# # Arima Model

# In[40]:


def adf_test(series):
    result=adfuller(series)
    print('ADF Statistics: {}'.format(result[0]))
    print('p- value: {}'.format(result[1]))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[41]:


result = adf_test(sp500['Close'].dropna())


# In[42]:


### First Shift 
## Use Techniques Differencing
sp500['Close_1']=sp500["Close"]-sp500["Close"].shift(1)


# In[43]:


result = adf_test(sp500['Close_1'].dropna())


# In[44]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[47]:


fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(211)
acf = sm.graphics.tsa.plot_acf(sp500['Close_1'].dropna(),lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
pacf = sm.graphics.tsa.plot_pacf(sp500['Close_1'].dropna(),lags=50, ax=ax2)


# In[19]:


# For non-seasonal data
#p=0, d=1, q=0 or 1
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm


# In[48]:


model = sm.tsa.arima.ARIMA(sp500["Close"],order=(45,1,45))
model_fit=model.fit()


# In[50]:


model_fit.summary()


# In[57]:


sp500['forecast']=model_fit.predict(start="2022-05-01",end="2022-05-28",dynamic=True)
sp500[['Close','forecast']].plot(figsize=(12,8))


# In[23]:


### 12 months 
## Use Techniques Differencing
sp500['Close 12 Difference']=sp500["Close"]-sp500["Close"].shift(24)


# # Sarimax

# In[27]:


model=sm.tsa.statespace.SARIMAX(sp500["Close"],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()


# In[33]:


sp500['forecast']=results.predict(start="2020-10-01",end="2020-11-30",dynamic=True)
sp500[['Close','forecast']].plot(figsize=(12,8))
plt.xlim(pd.Timestamp('2020-06-01'), pd.Timestamp('2021-03-01'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


result = adf_test(sp500['Close 12 Difference'].dropna())


# In[ ]:


### 2_12 months 
## Use Techniques Differencing
sp500['Close 2_12 Difference']=sp500["Close 12 Difference"]-sp500["Close 12 Difference"].shift(1)


# In[ ]:


result_1 = adf_test(sp500['Close 2_12 Difference'].dropna())


# In[ ]:





# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[ ]:


acf12 = plot_acf(sp500['Close 12 Difference'].dropna())


# In[ ]:


pacf12 = plot_pacf(sp500['Close 2_12 Difference'].dropna())


# In[ ]:


## create a SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:


model_SARIMA=SARIMAX(train_data['Thousands of Passengers'],order=(3,0,5),seasonal_order=(0,1,0,365))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Test for random walk hypothesis using Augmented Dickey-Fuller test
result = adfuller(sp500["returns"])
print("ADF Test:")
print("Test Statistic: ", result[0])
print("p-value: ", result[1])
print("Critical Values:")
for key, value in result[4].items():
    print("\t", key, ": ", value)


# In[ ]:


# Plot autocorrelation and partial autocorrelation
plt.figure(figsize=(12, 8))

plt.subplot(2,1,1)
acf_result = acf(sp500["returns"], nlags=30)
plt.plot(acf_result)
plt.axhline(y=0, linestyle="--", color="gray")
plt.axhline(y=-1.96/np.sqrt(len(sp500["returns"])), linestyle="--", color="gray")
plt.axhline(y=1.96/np.sqrt(len(sp500["returns"])), linestyle="--", color="gray")
plt.title("Autocorrelation")


# In[ ]:


plt.figure(figsize=(12, 8))
plt.subplot(2,1,2)
pacf_result = pacf(sp500["returns"], nlags=30, method="ols")
plt.plot(pacf_result)
plt.axhline(y=0, linestyle="--", color="gray")
plt.axhline(y=-1.96/np.sqrt(len(sp500["returns"])), linestyle="--", color="gray")
plt.axhline(y=1.96/np.sqrt(len(sp500["returns"])), linestyle="--", color="gray")
plt.title("Partial Autocorrelation")


# In[ ]:


plt.tight_layout()
plt.show()


# In[ ]:




