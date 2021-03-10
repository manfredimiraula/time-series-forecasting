# import libraries

# modeling
# grid search simple forecasts
import numpy as np
from numpy import mean
from numpy import median
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from numpy import array

# visualization
import matplotlib.pyplot as plt


def simple_forecast(history, config):
    """
    Apply naive and averaging simple forecasting.
    Arguments:
        history: Numpy or list of the time series
        config:
            - n -> define the number of observations to hold to determine the average and forecast forward
            - offset -> determines the seasonality, i.e. the number of observations to count backwards before collecting values from which to include in the average
            - avg_type -> either mean or median in the case of non-Gaussian distribution
    Returns:
        output sequence
    """
    n, offset, avg_type = config
    # persist value, ignore other config
    if avg_type == 'persist':
        return history[-n]
    # collect values to average
    values = list()
    if offset == 1:
        values = history[-n:]
    else:
        # skip bad configs
        if n*offset > len(history):
            raise Exception('Config beyond end of data: %d %d' % (n, offset))
        # try and collect n values using offset
        for i in range(1, n+1):
            ix = i * offset
            values.append(history[-ix])
    # check if we can average
    if len(values) < 2:
        raise Exception('Cannot calculate average')
    # mean of last n values
    if avg_type == 'mean':
        return mean(values)
    # median of last n values
    return median(values)


def simple_configs(max_length, offsets=[1]):
    """
    This function generates the matrix of parameters we can use for fitting the simple naive and average model. The models are selected 
    - persistent == naive
    - average either median or mean
    and the differencing of lags and seasonality is determined based on the lenght of the ts
    """
    configs = list()
    for i in range(1, max_length+1):
        for o in offsets:
            for t in ['persist', 'mean', 'median']:
                cfg = [i, o, t]
                configs.append(cfg)
    return configs

    # one-step Holt Winter’s Exponential Smoothing forecast (ETS)


def exp_smoothing_forecast(history, config):
    """
    Apply triple exponential smoothing forecasting.
    Arguments:
        history: Numpy or list of the time series
        config:
        - t == trend -> additive, multiplicative, None
        - d == damped_trend -> True, False
        - s == seasonal -> additive, multiplicative, None
        - p == sample_periods -> number of period in a complete seasonal cycle
        - b == use_boxcox -> True, False
        - r == remove_bias -> True, False
    Returns:
        forecast
    """
    t, d, s, p, b, r = config
    # define model
    history = array(history)
    model = ExponentialSmoothing(
        history, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p)
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]

# create a set of exponential smoothing configs to try


def exp_smoothing_configs(seasonal=[None]):
    """
    This function generates the matrix of parameters we can use for fitting the Triple Exponential smoothing model. From the statsmodels page:
    - t == trend -> additive, multiplicative, None
    - d == damped_trend -> True, False
    - s == seasonal -> additive, multiplicative, None
    - p == sample_periods -> number of period in a complete seasonal cycle
    - b == use_boxcox -> True, False
    - r == remove_bias -> True, False
    """
    models = list()
    # define config lists
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t, d, s, p, b, r]
                            models.append(cfg)
    return models


# create a set of sarima configs to try


def sarima_configs(seasonal=[0]):
    """
    This function generates the matrix of parameters we can use for fitting the SARIMAX model. From the statsmodels page:
    - order -> represented by the parametrs p, d, q for the model of the trend
    - seasonal_order -> represented by the parameters (P, D, Q)
    - trend -> to control the model deterministic trend (no trend 'n', 'c' constant, 't' linear, 'ct' constant with linear trend)
    """
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models


# one-step sarima forecast
def sarima_forecast(history, config):
    """
    This function forecast one step using SARIMAX model. From the statsmodels page:
    - order -> represented by the parametrs p, d, q for the model of the trend
    - seasonal_order -> represented by the parameters (P, D, Q)
    - trend -> to control the model deterministic trend (no trend 'n', 'c' constant, 't' linear, 'ct' constant with linear trend)
    """
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one-step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# root mean squared error or RMSE


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# quick function to calculate the MAPE error


def measure_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# split a univariate dataset into train/test sets


def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# score a model, return None on failure


def score_model(data, n_test, cfg, model_type, measure_type, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(
            data, n_test, cfg, model_type, measure_type)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(
                    data, n_test, cfg, model_type, measure_type)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)

# grid search configs


def grid_search(data, cfg_list, n_test, model_type, measure_type, parallel=True):
    """
    This uses parallel cpus to grid search the parameter space for hyperparameter optimization. It is based on the function using walk-forward cross-validation
    """
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg,
                                      model_type, measure_type) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg, model_type,
                              measure_type) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

# walk-forward validation for univariate data


def walk_forward_validation(data, n_test, cfg, model_type, measure_type):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    if model_type == 'simple':
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = simple_forecast(history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
    if model_type == 'ets':
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = exp_smoothing_forecast(history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
    if model_type == 'sarimax':
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = sarima_forecast(history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
    # estimate prediction error
    if measure_type == 'rmse':
        error = measure_rmse(test, predictions)
    if measure_type == 'mape':
        error = measure_mape(test, predictions)
    return error


"""
Visualization functions
"""


def ets_plot(history, config, n_outputs, n_sim):
    """
    This function should be used with the parameters optimized during training. It train the ETS model on the entire dataset and plots n outputs based on user input. Parameters: 
    - config 
        -
    - n_outputs --> number of steps to be forecasted
    - n_sim --> number of simulations to run for the forecast
    """

    trend, damped_trend, seasonal, seasonal_periods, use_boxcox, initialization_method = config

    # create the model
    model = ExponentialSmoothing(history, trend=trend, damped_trend=damped_trend, seasonal=seasonal,
                                 seasonal_periods=seasonal_periods, use_boxcox=use_boxcox, initialization_method=initialization_method)

    # fit the model and print the summary
    results = model.fit()
    print(results.summary())

    # simulate the probability
    simulations = results.simulate(n_outputs, repetitions=n_sim, error='mul')

    # create the time horizon of prediction plus forecast
    end = len(results.fittedvalues) + n_outputs

    # In-sample one-step-ahead predictions, and out-of-sample forecasts
    predict = results.predict(start=0, end=end)
    idx = np.arange(len(predict))

    # Graph
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.xaxis.grid()
    ax.plot(history, 'k.', color='black', label='Observations')

    # Plot
    ax.plot(idx[1:-n_outputs], predict[1:-n_outputs],
            color='blue', label='ETS predictions')

    ax.plot(idx[-n_outputs:], predict[-n_outputs:],
            'k--', color='blue', linestyle='--', linewidth=2, label="SARIMAX forecast")

    ax.plot(idx[-n_outputs:], simulations[-n_outputs:],
            'k--', color='gray', linestyle='--', linewidth=0.2,
            # label="SARIMAX sim"
            )

    ax.set(title='ETS predictions')
    ax.set_ylabel("Observations versus predictions/forecast")
    ax.set_xlabel("Time")

    ax.legend(loc='best')


def sarima_plot(history, config, n_outputs):
    """
    This function should be used with the parameters optimized during training. It train the SARIMAX model on the entire dataset and plots n outputs based on user input. Parameters: 
    - config 
        - order -->  (a,b,c)
        - seasonal_order --> (a,b,c,d)
        - trend --> string
    - n_outputs --> number of steps to be forecasted
    """

    order, seasonal_order, trend = config

    # create the model
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                    trend=trend, enforce_stationarity=False, enforce_invertibility=False)

    # fit the model and print the summary
    results = model.fit()
    print(results.summary())

    # make predictions on the entire dataset
    # pred = results.predict(start=1, end=len(data))
    # predict = results.get_prediction(end=model.nobs + n_outputs)

    # In-sample one-step-ahead predictions, and out-of-sample forecasts
    predict = results.get_prediction(end=model.nobs + n_outputs)
    idx = np.arange(len(predict.predicted_mean))
    predict_ci = predict.conf_int(alpha=0.5)

    # Graph
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.xaxis.grid()
    ax.plot(history, 'k.', color='black', label='Observations')

    # Plot
    ax.plot(idx[1:-n_outputs], predict.predicted_mean[1:-
                                                      n_outputs], color='blue', label='SARIMAX predictions')
    ax.plot(idx[-n_outputs:], predict.predicted_mean[-n_outputs:],
            'k--', color='blue', linestyle='--', linewidth=2, label="SARIMAX forecast")
    ax.fill_between(
        idx, predict_ci[:, 0], predict_ci[:, 1], alpha=0.15, color='red', label="Confidence levels")

    ax.set(title='SARIMAX predictions')
    ax.set_ylabel("Observations versus predictions/forecast")
    ax.set_xlabel("Time")

    ax.legend(loc='best')
