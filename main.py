import numpy
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.layers import Dense, LSTM
from keras.models import Sequential
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange
from scipy import stats
from sklearn.metrics import r2_score

# fix random seed for reproducibility
numpy.random.seed(7)

def load_dataset(datasource: str) -> (numpy.ndarray, MinMaxScaler):
    

    dataframe = pandas.read_csv(datasource, usecols=[6])
    dataframe = dataframe.fillna(method="pad")
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    #plt.plot(dataset)
    #plt.show()
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset, scaler

def create_dataset(dataset: numpy.ndarray, look_back: int=1) -> (numpy.ndarray, numpy.ndarray):
   
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return numpy.array(data_x), numpy.array(data_y)


def split_dataset(dataset: numpy.ndarray, train_size, look_back) -> (numpy.ndarray, numpy.ndarray):
    
    if not train_size > look_back:
        raise ValueError('train_size must be lager than look_back')
    train, test = dataset[0:train_size, :], dataset[train_size - look_back:len(dataset), :]
    print('train_dataset: {}, test_dataset: {}'.format(len(train), len(test)))
    return train, test

def build_model(look_back: int, batch_size: int=1) -> Sequential:
    
    model = Sequential()
    model.add(LSTM(64,
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   use_bias=True,
                   batch_input_shape=(batch_size, look_back, 1),
                   stateful=True,
                   return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def plot_data(dataset: numpy.ndarray,
              look_back: int,
              train_predict: numpy.ndarray,
              test_predict: numpy.ndarray,
              forecast_predict: numpy.ndarray):

    plt.plot(dataset)
    plt.plot([None for _ in range(look_back)] +
             [x for x in train_predict], label = "Train_predict")
    plt.plot([None for _ in range(look_back)] +
             [None for _ in train_predict] +
             [x for x in test_predict], label = "Test_predict")
    plt.plot([None for _ in range(look_back)] +
             [None for _ in train_predict] +
             [None for _ in test_predict] +
             [x for x in forecast_predict], label = "Prediksi")
    plt.savefig('lstm.png')
    plt.show()

def make_forecast(model: Sequential, look_back_buffer: numpy.ndarray, timesteps: int=1, batch_size: int=1):
    forecast_predict = numpy.empty((0, 1), dtype=numpy.float32)
    for _ in trange(timesteps, desc='predicting data\t', mininterval=1.0):
        # make prediction with current lookback buffer
        cur_predict = model.predict(look_back_buffer, batch_size)
        # add prediction to result
        forecast_predict = numpy.concatenate([forecast_predict, cur_predict], axis=0)
        # add new axis to prediction to make it suitable as input
        cur_predict = numpy.reshape(cur_predict, (cur_predict.shape[1], cur_predict.shape[0], 1))
        # remove oldest prediction from buffer
        look_back_buffer = numpy.delete(look_back_buffer, 0, axis=1)
        # concat buffer with newest prediction
        look_back_buffer = numpy.concatenate([look_back_buffer, cur_predict], axis=1)
    return forecast_predict


def main():
    datasource = 'desember.csv'
    dataset, scaler = load_dataset(datasource)

    # split into train and test sets
    look_back = 1
    train_size = int(len(dataset) * 0.70)
    train, test = split_dataset(dataset, train_size, look_back)

    # reshape into X=t and Y=t+1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    train_x = numpy.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = numpy.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    # create and fit Multilayer Perceptron model
    batch_size = 1
    model = build_model(look_back, batch_size=batch_size)
    for _ in trange(100, desc='fitting model\t', mininterval=1.0):
        model.fit(train_x, train_y, nb_epoch=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()

    # generate predictions for training
    train_predict = model.predict(train_x, batch_size)
    test_predict = model.predict(test_x, batch_size)

    # generate forecast predictions
    forecast_predict = make_forecast(model, test_x[-1::], timesteps=24, batch_size=batch_size)

    # invert dataset and predictions
    dataset = scaler.inverse_transform(dataset)
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    forecast_predict = scaler.inverse_transform(forecast_predict)

    ravelTestPredict = numpy.ravel(test_predict)
    #print(numpy.shape(ravelTestPredict))
    ravelTestY = numpy.ravel(test_y)

    ravelForecast = numpy.ravel(forecast_predict)

    print(train_predict)
    print(test_predict)
    print(forecast_predict)

    print("\n")
    print("\n")

    slope, intercept, r_value, p_value, std_err = stats.linregress(ravelTestY,ravelTestPredict)
    #slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(ravelTestPredict, ravelForecast)
    line = slope*ravelTestY+intercept
    r_squared = r2_score (ravelTestY, ravelTestPredict)
    #r_squared1 = r2_score(ravelTestPredict, ravelForecast)

    accuracy = r_squared
    #accuracy1 = r_squared1
    print ("Accuracy Testing: %f%%" % ((accuracy)*100))
    #print ("Accuracy Forecasting: %f%%" % ((accuracy1)*100))
    
    # calculate root mean squared error
    train_score = numpy.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    print('Train Score: %f RMSE' % (train_score))
    test_score = numpy.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    print('Test Score: %f RMSE' % (test_score))



    plot_data(dataset, look_back, train_predict, test_predict, forecast_predict)


    np.savetxt("trainpredict_desember.csv", train_predict, delimiter= ",")
    np.savetxt("testpredict_desember.csv", test_predict, delimiter= ",")
    np.savetxt("forecastpredict_desember.csv", forecast_predict, delimiter= ",")

    """
    plt.plot(dataset, label="Observasi")
    plt.plot(train_predict, label = "data training")
    plt.plot(test_predict, forecast_predict, label = "data testing")
    plt.title("Recurrent Neural Networks")
    plt.xlabel("Data ke-")
    plt.ylabel("Angin")
    plt.legend()
    plt.savefig('lstm.png')
    plt.show()

    """

if __name__ == '__main__':
    main()
