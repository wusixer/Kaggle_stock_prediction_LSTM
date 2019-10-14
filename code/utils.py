import pandas as pd
import numpy as np
import plotly.graph_objects as graph
import plotly
from scipy.stats.mstats import zscore
from keras.models import Sequential
from keras.layers import Activation, Dense, TimeDistributed, LSTM, Dropout, Flatten, RepeatVector
from keras.optimizers import Adam
from numpy import array



def read_files_for_length(filenames, current_dir):
    '''
    Input
        filenames: a list of file names, e.g [a.txt, b.txt, ...]

    Returns
        a dictionary of filenames and file length
    '''
    path = current_dir + "/Data/Stocks/"
    file_dict = {}
    for filename in filenames:
        df = pd.read_csv(path+filename)
        fn_len = df.shape[0]
        file_dict[filename] = fn_len
    return file_dict


def plot_open_high_low_close(df, current_dir, outname):
    '''
    plot the open, high, low, close value across time

    Input
        pd dataframe with stock prices

    Output
        graph with open, high, low, close value across time
    '''
    # sort entries by time
    df = df.sort_values('Date')
    fig = graph.Figure()
    for col in ['Open', 'High', 'Low', 'Close']:
        #plot_color = str(hash(col))
        fig.add_trace(graph.Scattergl(x=df['Date'], y=df[col], name=col))
    fig.update_layout(showlegend=True, xaxis=graph.layout.XAxis(
        tickformat='%d %B (%a)<br>%Y'))
    plotly.offline.init_notebook_mode(connected=True)
    filename = current_dir + str(outname)+'png'
    plotly.offline.plot(fig, filename=filename)


def data_zcore_norm(data):
    '''z score normalise the ['Open', 'Low', 'Close','Volume'] of the stock data
    '''
    data_copy = pd.DataFrame.copy(data, deep=True)
    for col in ['High','Open', 'Low', 'Close', 'Volume']:
        data_copy[col] = zscore(data[col])
    return data_copy


def get_data(data, batch_size, window_size):
    """data is 2D array (input, features)"""
    # flatten data
    print("get_data")
    data = data.values
    X, y = list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + batch_size
        out_end = in_end + window_size
        print(in_start, in_end, out_end)
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return array(X), array(y)


def data_gen(out_X, out_Y, batch_size):
    while True:
        n_batch = out_X.shape[0] // batch_size
        for i in range(n_batch - 1):
            yield out_X[i*batch_size: (i + 1) * batch_size, :, :], out_Y[i*batch_size: (i + 1) * batch_size, ].reshape((batch_size, out_Y.shape[1], 1))


def build_model(window_size, lr):
    '''input dim (window_size, feature), output dim (window_size, 1)'''
    n_timesteps = window_size
    n_features = 4
    n_outputs = window_size
    model = Sequential()
    model.add(LSTM(200, activation='relu',
                   input_shape=(window_size, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mae', optimizer=Adam(lr=lr))
    return model

