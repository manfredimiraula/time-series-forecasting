# imports
from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import TimeDistributed
from matplotlib import pyplot

################## General functions##################################


def measure_rmse(actual, predicted):  # root mean squared error or rmse
    return sqrt(mean_squared_error(actual, predicted))


def measure_mape(y_true, y_pred):  # quick function to calculate the MAPE error
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def train_test_split(data, n_test):  # split a univariate dataset into train/test sets
    return data[:-n_test], data[-n_test:]


def difference(data, interval):  # difference dataset
    return [data[i] - data[i - interval] for i in range(interval, len(data))]


# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values


################## Modeling functions##################################

def mlp_fit(train, config):  # fit a MLP model
    """
    This function takes the data and the configuration parameters and fit the model.
    The model is a linear model (output) for determining a continuous variable. The parameters:
    - n_input: number of lags to use as input
    - n_nodes: number of nodes to use in the hidden layer
    - n_epochs: number of times to expose the model to the whole training dataset (i.e. repetitions)
    - n_batch: number of samples within each epoch for which the weights are updated
    """
    # unpack config
    n_input, n_nodes, n_epochs, n_batch, n_diff = config
    # prepare data
    if n_diff > 0:
        train = difference(train, n_diff)
     # transform series into supervised format
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    # define model
    model = Sequential()
    model.add(Dense(n_nodes, activation='relu', input_dim=n_input))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


def mlp_predict(model, history, config):  # forecast with a pre-fit MLP model
    # unpack config
    n_input, _, _, _, n_diff = config
    # prepare data
    correction = 0.0
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
    # shape input for model
    x_input = array(history[-n_input:]).reshape(1, n_input)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    # correct forecast if it was differenced
    return correction + yhat[0]


def cnn_fit(train, config):  # fit a CNN model
    """
    This function takes the data and the configuration parameters and fit the model.
    - n_input: number of lags to use as input
    - n_filters: number of parallel fields on which the weights are updated
    - n_kernel: number of time steps within each snaphot
    - n_epochs: number of times to expose the model to the whole training dataset (i.e. repetitions)
    - n_batch: number of samples within each epoch for which the weights are updated
    - n diff: The difference order (e.g. 0 or 12)
    """
    # unpack config
    n_input, n_filters, n_kernel, n_epochs, n_batch, n_diff = config
    # prepare data
    if n_diff > 0:
        train = difference(train, n_diff)
    # transform series into supervised format
    data = series_to_supervised(train, n_in=n_input)
    # separate inputs and outputs
    train_x, train_y = data[:, :-1], data[:, -1]
    # reshape input data into [samples, timesteps, features]
    n_features = 1
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
    # define model
    model = Sequential()  # multiple hidden layers
    # two convolutional layer as hidden layer
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernel,
                     activation='relu', input_shape=(n_input, 1)))
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))  # filter only important features
    model.add(Flatten())  # flatten the input to one vector
    model.add(Dense(1))  # dense layers
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


def cnn_predict(model, history, config):  # forecast with a pre-fit CNN model
    # unpack config
    n_input, _, _, _, _, n_diff = config
    # prepare data
    correction = 0.0
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return correction + yhat[0]


def lsmt_fit(train, config):  # fit a Recurrent NN model
    """
    Parameters:
    - n_input: number of lag observations to use as inputs
    - n_nodes: number of LSTM to use in the hidden layer
    - n_epochs: number of times to expose the model to the whole training dataset
    - n_atch: number of samples within each epoch after which the weights are updated
    - n_diff: the the order of differencing if it is used (0 if none) to make the ts stationary
    """
    # unpack config
    n_input, n_nodes, n_epochs, n_batch, n_diff = config
    # prepare data
    if n_diff > 0:
        train = difference(train, n_diff)
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


def lsmt_predict(model, history, config):  # forecast with a pre-fit RNN model
    # unpack config
    n_input, _, _, _, n_diff = config
    # prepare data
    correction = 0.0
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return correction + yhat[0]


def cnn_lstm_fit(train, config):  # fit a cnn-lstm model
    """
    - n_seq: The number of subsequences within a sample.
    - n_steps: The number of time steps within each subsequence.
    - n_filters: The number of parallel filters.
    - n_kernel: The number of time steps considered in each read of the input sequence.
    - n_nodes: The number of LSTM units to use in the hidden layer.
    - n_epochs: The number of times to expose the model to the whole training dataset.
    - n_batch: The number of samples within an epoch after which the weights are updated.
    """
    # unpack config
    n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
    n_input = n_seq * n_steps
    # prepare data
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], n_seq, n_steps, 1))
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel,
                                     activation='relu', input_shape=(None, n_steps, 1))))  # CNN
    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel,
                                     activation='relu')))  # ?
    # select only the most important features
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))  # prepare the vector
    model.add(LSTM(n_nodes, activation='relu'))  # LSTM layer
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))  # output interpreter
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit cnn-lstm model
def cnn_lstm_predict(model, history, config):
    # unpack config
    n_seq, n_steps, _, _, _, _, _ = config
    n_input = n_seq * n_steps
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def convlstm_fit(train, config):  # fit a convlstm model
    """
    Parameters:
    - n_seq: The number of subsequences within a sample.
    - n_steps: The number of time steps within each subsequence.
    - n_filters: The number of parallel filters.
    - n_kernel: The number of time steps considered in each read of the input sequence.
    - n_nodes: The number of LSTM units to use in the hidden layer.
    - n_epochs: The number of times to expose the model to the whole training dataset.
    - n_batch: The number of samples within an epoch after which the weights are updated.

    """
    # unpack config
    n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
    n_input = n_seq * n_steps
    # prepare data
    data = series_to_supervised(train, n_input)
    train_x, train_y = data[:, :-1], data[:, -1]
    train_x = train_x.reshape((train_x.shape[0], n_seq, 1, n_steps, 1))
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=n_filters, kernel_size=(1, n_kernel),
                         activation='relu', input_shape=(n_seq, 1, n_steps, 1)))
    model.add(Flatten())
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')  # fit
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit ConvLSTM model
def convlstm_predict(model, history, config):
    # unpack config
    n_seq, n_steps, _, _, _, _, _ = config
    n_input = n_seq * n_steps
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_seq, 1, n_steps, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]

################## Training functions ##################################


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg, model_type, measure_type):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # select the model
    if model_type == 'mlp':
        # fit model
        model = mlp_fit(train, cfg)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = mlp_predict(model, history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
    if model_type == 'cnn':
        # fit model
        model = cnn_fit(train, cfg)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = cnn_predict(model, history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
    if model_type == 'lsmt':
        # fit model
        model = lsmt_fit(train, cfg)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = lsmt_predict(model, history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
    if model_type == 'cnn_lstm':
        # fit model
        model = cnn_lstm_fit(train, cfg)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = cnn_lstm_predict(model, history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
    if model_type == 'convlstm':
        # fit model
        model = convlstm_fit(train, cfg)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = convlstm_predict(model, history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
    # select error measure
    if measure_type == 'rmse':
        # estimate prediction error
        error = measure_rmse(test, predictions)
        print(' > %.3f' % error)
    if measure_type == 'mape':
        error = measure_mape(test, predictions)
    return error


# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, model_type, measure_type, n_repeats=30):
    # fit and evaluate the model n times
    scores = [walk_forward_validation(data, n_test, config, model_type, measure_type)
              for _ in range(n_repeats)]
    return scores


def summarize_scores(name, scores):  # summarize model performance
    # print a summary
    scores_m, score_std = mean(scores), std(scores)
    print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
    # box and whisker plot
    pyplot.boxplot(scores)
    pyplot.show()

################## Grid-search functions ##################################


def mlp_configs():  # create a list of MLP configs to try
    # define scope of configs
    n_input = [12]
    n_nodes = [50, 100]
    n_epochs = [100]
    n_batch = [1, 150]
    n_diff = [0, 12]
    # create configs
    configs = list()
    for i in n_input:
        for j in n_nodes:
            for k in n_epochs:
                for l in n_batch:
                    for m in n_diff:
                        cfg = [i, j, k, l, m]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs


def cnn_configs():  # create a list of CNN configs to try
    # define scope of configs
    n_input = [12]
    n_filters = [64]
    n_kernels = [3, 5]
    n_epochs = [100]
    n_batch = [1, 150]
    n_diff = [0, 12]
    # create configs
    configs = list()
    for a in n_input:
        for b in n_filters:
            for c in n_kernels:
                for d in n_epochs:
                    for e in n_batch:
                        for f in n_diff:
                            cfg = [a, b, c, d, e, f]
                            configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs


# grid search a model, return None on failure
def grid_search_repeat_evaluate(data, config, n_test, model_type, measure_type, n_repeats=30):
    # convert config to a key
    key = str(config)
    # fit and evaluate the model n times
    scores = [walk_forward_validation(data, n_test, config, model_type, measure_type)
              for _ in range(n_repeats)]  # summarize score
    result = mean(scores)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)


def grid_search(data, cfg_list, n_test, model_type, measure_type):  # grid search configs
    # evaluate configs
    scores = scores = [grid_search_repeat_evaluate(
        data, cfg, n_test, model_type, measure_type) for cfg in cfg_list]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
