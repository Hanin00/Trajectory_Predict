import keras.layers
import pandas as pd
import numpy as np

import matplotlib
import glob, os
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler
import sys
import random

from pylab import mpl, plt

from datetime import datetime
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle


matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

#[전체 데이터 개수,300][2 - XY][2 - 0 : x, 1 : y]
# print(len(data.iloc[0]))
# data.iloc[0][:][0][0]
# print(data.iloc[0][:][0][1])
# print(len(data.iloc[0][:][0][0]))
# print(len(data.iloc[0][:][0][1]))
# print(len(data.iloc[0][:]))
# print(data.iloc[0][:][1])


# data = loadData의 data[i]
#
# def loadData2(data) :
#     x_x = data.iloc[:][0][0]
#     x_y = data.iloc[:][0][1]
#     y = data.iloc[:][1]
#
#     i = 0
#     data = {"x_x": x_x,
#             "x_y": x_y, }
#     dataPd = pd.DataFrame(data)
#     dataPd.loc[len(dataPd)] = y
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(dataPd[['x_x', 'x_y']].values)
#     reframed = series_to_supervised(scaled_data, 5,
#                                     1)  # t = 50 ;  # 12 -> step = 5 + predict = 1 <- feature = x_pos, y_pos
#
#     train_days = 50  # 50
#     # valid_days = 2
#     values = reframed.values
#     train = values[:train_days + 1, :, ]
#     # valid = values[-valid_days:, :] #<-전체 데이터에서 분류할 것
#     # return values, train, valid
#     return values, train
# def series_to_supervised2(data, n_in=1, n_out=1, dropnan=True):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = pd.DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#
#     # put it all together
#     agg = pd.concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg


def loadData(data) :
    x_x = data.iloc[:][0][0]
    x_y = data.iloc[:][0][1]
    y = data.iloc[:][1]

    i = 0
    data = {"x_x": x_x,
            "x_y": x_y, }
    dataPd = pd.DataFrame(data)
    dataPd.loc[len(dataPd)] = y

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataPd[['x_x', 'x_y']].values)
    reframed = series_to_supervised(scaled_data, 5,1)  # t = 50 ;  # 12 -> step = 5 + predict = 1 <- feature = x_pos, y_pos

    train_days = 50  # 50
    # valid_days = 2
    values = reframed.values
    train = values[:train_days + 1, :, ]
    # valid = values[-valid_days:, :] #<-전체 데이터에서 분류할 것
    # return values, train, valid
    return values, train, scaler

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]


    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 output_dim):  # num_layers : 2, hidden_dim : 32, input_dim : 1, self : LSTM(1,32,2,batch_firsttrue)
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # fc = nn.Linear(hidden_dim, output_dim)

        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop  all the way to the start even after going through another batch

        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm(x)

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        # out = self.fc(out[:, -1, :])
        out = self.fc(out[:, :])
        # out.size() --> 100, 10
        return out



def train(trainData) :
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
# 데이터 하나 당 epoch 씩 학습
    for i in range(len(trainData)) :

        train_values, train_data, scaler = loadData(trainData.iloc[i])
        train_X, train_y = train_data[:, :-2], train_data[:, -2:]  # 끝에 두 개가  Y의 x,y에 대한 예측값

        train_X = torch.Tensor(train_X)
        train_y = torch.Tensor(train_y)

        print(train_X.shape, train_y.shape) #(46, 10) (46, 2)
        print("train data Num : ",i)
        for t in range(num_epochs):

            train_X = torch.Tensor(train_X)
            train_y = torch.Tensor(train_y)

            y_train_pred = model(train_X)

            loss = loss_fn(y_train_pred, train_y)

            x_loss = loss_fn(y_train_pred[:, 0], train_y[:, 0])
            y_loss = loss_fn(y_train_pred[:, 1], train_y[:, 1])

            if t % 10 == 0 and t != 0:
                print("Epoch ", t, "MSE: ", loss.item())
                print("x_loss : ", x_loss.item())
                print("y_loss : ", y_loss.item())

            hist[t] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()
            train_predict = model(train_X)


    plt.figure(figsize=(24, 8))
    plt.xlabel('x')
    plt.ylabel('y')


    # train-values의 X값 비교
    plt.title(label="train-values의 X값 비교")
    plt.plot(list(range(len(train_values[:, 0]))), train_values[:, 0], label='raw_trajectory', c='b')
    plt.plot(list(range(len(train_predict[:, 0]))), train_predict[:, 0].detach().numpy(), label='test_predict', c='r')
    plt.legend()
    plt.show()

    plt.gca()
    # train-values의 Y값 비교
    plt.title(label="train-values의 Y값 비교")
    plt.plot(list(range(len(train_values[:, 1]))), train_values[:, 1], label='raw_trajectory', c='b')
    plt.plot(list(range(len(train_predict[:, 1]))), train_predict[:, 1].detach().numpy(), label='test_predict', c='r')
    plt.legend()
    plt.show()
    #
    # x_loss = loss_fn(train_values[:,0],train_predict[:,0])
    # y_loss = loss_fn(train_values[:,1],train_predict[:,1])
    #
    # print(x_loss)


def test(testData) :
    # 데이터 하나 당 epoch 씩 학습
    for i in range(len(testData)):
        test_values, test_data, scaler  = loadData(testData.iloc[i])
        test_X, test_y = test_data[:, :-2], test_data[:, -2:]  # 끝에 두 개가  Y의 x,y에 대한 예측값

        test_X = torch.Tensor(test_X)
        test_y = torch.Tensor(test_y)

        #todo - loss 가 이 위치 또는 더 상위에 있어야 하나?
        test_X = torch.Tensor(test_X)
        test_y = torch.Tensor(test_y)
        y_test_pred = model(test_X)

        loss = loss_fn(y_test_pred, test_y)
        print("test loss : ", loss.item())

        # x_loss = loss_fn(test_values[0], test_y[0])
        # y_loss = loss_fn(test_values[1], test_y[1])
        #
        # print("x_loss : ",x_loss.item())
        # print("y_loss : ",y_loss.item())


        test_predict = model(test_X)


    plt.figure(figsize=(24, 8))
    plt.xlabel('x')
    plt.ylabel('y')
    # for LSTM
    #test-values의 X값 비교
    plt.title(label="test-values의 X값 비교")
    plt.plot(list(range(len(test_values[:, 0]))),test_values[:, 0], label='raw_trajectory', c='b')
    plt.plot(list(range(len(test_values[:, 0]))), test_predict[:, 0].detach().numpy(), label='test_predict', c='r')
    plt.legend()
    plt.show()

    plt.gca()
    # test-values의 Y값 비교
    plt.title(label="test-values의 Y값 비교")
    plt.plot(list(range(len(test_values[:, 1]))), test_values[:, 1], label='raw_trajectory', c='b')
    plt.plot(list(range(len(test_values[:, 1]))), test_predict[:, 1].detach().numpy(), label='test_predict', c='r')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    path = "D:/Semester2201/LAB/Vessel_Trajectory_Prediction-main/"

    # with open('./data/data_prof.pickle', 'rb') as f :
    with open('./data/listTrainData.pickle', 'rb') as f:
        data = pickle.load(f)

    # flag = int(len(data) * 0.7) # 210
    # print(flag)
    trainData = data.iloc[:30]
    testData = data.iloc[30:50]

    #INIT - model
    #####################
    num_epochs = 200
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    # seq_dim = look_back - 1
    input_dim = 10
    hidden_dim = 128
    num_layers = 2
    output_dim = 2

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    train(trainData)
    test(testData)

