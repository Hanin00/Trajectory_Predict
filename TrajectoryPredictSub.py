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

'''
    ver 1 : 50개의 데이터를 5일마다의 간격으로 학습하고 이를 300번 반복
    ver 2 : 300*50 개의 데이터를 50일 마다

'''


pd.set_option('display.max_rows', None)


matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

#[전체 데이터 개수,300][2 - XY][2 - 0 : x, 1 : y]


def preprocessing(dataX, dataY) :

#todo data scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataX[['x_x','x_y']])

    reframed = series_to_supervised(scaled_data, 50,1)  # t = 50 ;  # 12 -> step = 5 + predict = 1 <- feature = x_pos, y_pos


    train_days = 210
    # valid_days = 2
    values = reframed.values
    train = values[:train_days + 1, :, ]
    # valid = values[-valid_days:, :] #<-전체 데이터에서 분류할 것

    return values, train, scaler


    # names = []
    # [names.append('t{}'.format(i))for i in range(len(data['feature'][0]), 0,-1)]
    # # names = 't50', 't49', 't48', 't47', 't46',  ... , 't4', 't3', 't2', 't1'
    #
    # # ref) https://blog.naver.com/nomadgee/220857820094
    # num = len(data['feature'][0])
    # # for k in range(num):  # 50번
    # #     globals()['t{}'.format(num - k)] = [data['feature'][c][k] for c in range(len(data))]
    # #     print(t50)
    # #     print(len(t50))
    # #
    # # dataX = {
    # #     't1' : t1,
    # #     't2': t2,
    # #     't3': t3,
    # #     't4': t4,
    # #     't5': t5,
    # #     't1': t1,
    # #     't1': t1,
    # #     't1': t1,'t1' : t1,'t1' : t1,
    # # #     't1': t1,
    # # # }
    # #
    # # dataPd = pd.DataFrame(, columns = names)
    # #
    # # scaled_data = scaler.fit_transform(data[[names]].values)
    # # # scaled_data = scaler.fit_transform(data[['X', 'Y']].values)
    # # print(scaled_data.head())
    # #
    # # sys.exit()
    # #
    # # reframed = series_to_supervised(scaled_data, 50,1)  # t = 50 ;  # 12 -> step = 5 + predict = 1 <- feature = x_pos, y_pos
    # #
    # #
    # # train_days = 300*50  # 50
    # # # valid_days = 2
    # # values = reframed.values
    # # train = values[:train_days + 1, :, ]
    # # # valid = values[-valid_days:, :] #<-전체 데이터에서 분류할 것
    # # # return values, train, valid
    # return values, train, scaler

#tran dataset 생성
def series_to_supervised(dataX, n_in=1, n_out=1, dropnan=True):
    # dataX = dataX.values
    n_vars = 1 if type(dataX) is list else dataX.shape[1]
    df = pd.DataFrame(dataX)
    #dataX(50*210)개에 대해 50(n_in)개씩 이동
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    # 전체 데이터 수에서 50개씩 나눈 것 만큼 움직임(300개 기준 210번)
    #

    for i in range(n_in, 0, -1):
        print(i)
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)] #var1(t-50)

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    print(cols)
    print(names)
    print(len(cols))
    print(len(names))
    sys.exit()


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


# 50개의 [X,Y] 궤적을 갖는 데이터 당 한 개의 [X,Y]좌표 Y label 300개
def train(trainX, trainY) :
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
# 데이터 하나 당 epoch 씩 학습
    train_values, train_data, scaler = preprocessing(trainX, trainY)



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
        # x_loss = loss_fn(y_train_pred[:, 0], train_y[:, 0])
        # y_loss = loss_fn(y_train_pred[:, 1], train_y[:, 1])

        if t % 10 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())
            # print("x_loss : ", x_loss.item())
            # print("y_loss : ", y_loss.item())

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


def test(testX, testY) :
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

#ratio : train-test 비율 / time = 궤적 개수
def loadData(data, ratio=0.7, time=50) :

    x_x = []
    [x_x.extend(data['feature'].iloc[i][0]) for i in range(len(data))] #50개마다 다른 차량
    x_y = []
    [x_y.extend(data['feature'].iloc[i][1]) for i in range(len(data))]  # 50개마다 다른 차량

    xDict = {'x_x' : x_x,
                'x_y' : x_y}
    xDf = pd.DataFrame(xDict)

    flag = int(len(data) * ratio)  # 210
    trainX = xDf[:flag*time]
    testX = xDf[flag*time:]


    y_x = []
    [y_x.append(data['label'][i][0]) for i in range(len(data['label']))]
    y_y = []
    [y_y.append(data['label'][i][1]) for i in range(len(data['label']))]

    yDict = { 'y_x' : y_x,
              'y_y' : y_y}

    yDf = pd.DataFrame(yDict)

    trainY = yDf[:flag]
    testY = yDf[flag:]

    return trainX, trainY, testX, testY

if __name__ == '__main__':

    path = "D:/Semester2201/LAB/Vessel_Trajectory_Prediction-main/"

    # with open('./data/data_prof.pickle', 'rb') as f :
    # with open('./data/data_ver2_1025.pickle', 'rb') as f :
    with open('./data/listTrainData.pickle', 'rb') as f:
        data = pickle.load(f)

    trainX, trainY, testX, testY = loadData(data)

    # print(trainX.head())
    # print(trainY.head())
    # print(testX.head())
    # print(testY.head())
    #
    # print(len(trainX)//50)
    # print(len(trainY))
    # print(len(testX)//50)
    # print(len(testY))


    #INIT - model
    #####################
    num_epochs = 200
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    # seq_dim = look_back - 1
    input_dim = 100
    hidden_dim = 128

    num_layers = 2
    output_dim = 2

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    train(trainX, trainY)
    test(testX, testY)

