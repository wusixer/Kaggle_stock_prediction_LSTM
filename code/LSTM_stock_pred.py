from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
#import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
import keras
import subprocess
from utils import read_files_for_length, plot_open_high_low_close, data_zcore_norm, get_data, data_gen, build_model
import plotly.graph_objects as graph
import plotly
from scipy.stats.mstats import zscore
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# find the top 1 stock that has the highest entries
current_dir = "/Users/jiayicox/Documents/job_prep/kaggle_stock_market"
data_dir = current_dir + "/Data/Stocks/"
all_stocks = [file for file in os.listdir(data_dir)]
all_stock_len =   read_files_for_length(all_stocks, current_dir)    
studystock = max(all_stock_len, key =all_stock_len.get)

# read in file
studystock = pd.read_csv(data_dir + studystock)

# plot data
plot_open_high_low_close(studystock, current_dir, 'raw')

# split data
studystock = studystock.iloc[:14000,[0,1,2,3,4,5]]
otherdata = studystock.iloc[ : 11000, :]
test = studystock.iloc[11000:, :]

#zscore normalize data
otherdata_norm = data_zcore_norm(otherdata)
test_norm = data_zcore_norm(test)
all_norm = otherdata_norm.append(test_norm)


#plot
plot_open_high_low_close(test_norm, current_dir, 'znorm')

# split otherdata to train and validation, and get rid of the 'time' column
train = otherdata_norm.iloc[:9000, 1:]
validation = otherdata_norm.iloc[9000:, 1:]
test = test_norm.iloc[:, 1:]

# process data, generate features and labels by window size and save it as new variables
window_size = 100
batch_size =100
train_X, train_Y = get_data(train, batch_size, window_size)
validation_X, validation_Y = get_data(validation, batch_size, window_size)
test_X, test_Y = get_data(test, batch_size, window_size)

train_X.shape
train_Y.shape
validation_X.shape
validation_Y.shape
test_X.shape
test_Y.shape
    

# start training
n_epochs = 20
lr = 0.008
train_steps_per_epoch =  train_X.shape[0] //batch_size 
validation_steps_per_epoch = validation_X.shape[0]//batch_size 

model =  build_model(window_size = window_size,  lr = lr)
model.summary()

# train model
model.fit_generator(data_gen(train_X, train_Y, batch_size), 
                    steps_per_epoch = train_steps_per_epoch,
                     epochs= n_epochs, 
                     validation_data =  data_gen(validation_X, validation_Y, batch_size),
                     validation_steps =  validation_steps_per_epoch)

# validate
pred_validation_Y = np.zeros((validation_Y.shape[0], validation_Y.shape[1],1))


for window in range(validation_Y.shape[0]//window_size +1):
    print(window)
    pred_validation_Y[window*window_size: (window +1)* window_size,:, :] = model.predict(validation_X[window*window_size: (window +1)* window_size, :,:], batch_size=100, verbose=2)


plt.plot(validation_Y.reshape(validation_Y.shape[0]*validation_Y.shape[1],1), label = "actual")
plt.plot(pred_validation_Y.reshape(validation_Y.shape[0]*validation_Y.shape[1],1), label = "predicted")
plt.legend()
#plt.show()
plt.savefig(current_dir + '/validation_pred_true.png')
MAE = mean_absolute_error(validation_Y.reshape(validation_Y.shape[0],1), pred_validation_Y)
print('The Mean Absolute Error is: {}'.format(MAE))


# test 
pred_test_Y = np.zeros((test_raw_X.shape[0],1))

for window in range(test_raw_X.shape[0]- window_size - window_size  ) : 
    print(window)
    if window == 0:
        pred_test_Y[window: window + window_size, ] = model.predict(test_X[window : window+ window_size, :,:], batch_size=100, verbose=2)
    if window > 0:
        test_raw_X[window: window + window_size, 2:3] = pred_test_Y[window -1: window + window_size-1, :]
        test_X, _ = get_data(test_raw_X, test_raw_Y, window_size) 
        pred_test_Y[window: window +window_size, ] = model.predict(test_X[window :window + window_size,:  ,:], batch_size=100, verbose=2)

plt.plot(test_Y.reshape(test_Y.shape[0],1), label = "actual")
plt.plot(pred_test_Y, label = "predicted")
plt.legend()
#plt.show()
plt.savefig(current_dir + '/test_pred_true.png')
MAE = mean_absolute_error(test_Y.reshape(test_Y.shape[1],1), pred_test_Y)
print('The Mean Absolute Error is: {}'.format(MAE))
