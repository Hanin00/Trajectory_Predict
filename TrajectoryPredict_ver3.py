# ref) https://coding-yoon.tistory.com/190 \
# https://colab.research.google.com/drive/19T8bOq3MLvBGAKYouZpY7FoU0tAH3SH7#scrollTo=YWTCiXLdghFI

import pandas as pd
import numpy as np

import matplotlib
import glob, os
import seaborn as sns
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
from sklearn.preprocessing import minmax_scale

'''
    ver 1 : 50개의 데이터를 5일마다의 간격으로 학습하고 이를 300번 반복
    ver 2 : 300*50 개의 데이터를 50일 마다

'''
import scipy.stats as ss

pd.set_option('display.max_rows', None)

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

#[전체 데이터 개수,300][2 - XY][2 - 0 : x, 1 : y]

#train dataset 생성
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
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)] #var1(t-50)

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
        h0 = torch.zeros(self.num_layers, self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Index hidden state of last time step
        out = self.fc(out[:, :])
        # out.size() --> 100, 10
        return out

def train(trainData):
    for t in range(num_epochs):
        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        a = []
        for i in range(len(trainData)) :
            b = []
            for j in range(len(trainData['feature'].iloc[i])) :
                b.extend(trainData['feature'].iloc[i][j])
            a.append(b)
        train_X = torch.Tensor(a)
        train_y = torch.Tensor(trainData['label'])

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
    plt.plot(list(range(len(train_y[:, 0]))), train_y[:, 0], label='raw_trajectory', c='b')
    plt.plot(list(range(len(train_predict[:, 0]))), train_predict[:, 0].detach().numpy(), label='predict', c='r')
    plt.legend()
    plt.show()

    plt.gca()
    # train-values의 Y값 비교
    plt.title(label="train-values의 Y값 비교")
    plt.plot(list(range(len(train_y[:, 1]))), train_y[:, 1], label='raw_trajectory', c='b')
    plt.plot(list(range(len(train_predict[:, 1]))), train_predict[:, 1].detach().numpy(), label='predict', c='r')
    plt.legend()
    plt.show()

def test(testData) :
    loss_fn = torch.nn.MSELoss()
    # optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    a = []
    for i in range(len(testData)):
        b = []
        for j in range(len(testData['feature'].iloc[i])):
            b.extend(testData['feature'].iloc[i][j])
        a.append(b)

    test_X = torch.Tensor(a)
    test_y = torch.Tensor(testData['label'])
    # y_test_pred = model(test_X)




    with torch.no_grad():
        preds = []
        for _ in range(len(test_X)):
            # model.reset_hidden_state()
            y_test_pred = model(test_X)
            # pred = torch.flatten(y_test_pred).item()
            preds.append(y_test_pred)
            loss = loss_fn(y_test_pred, test_y)
            print("loss : ", loss)

    plt.plot(np.array(test_y)*MAX, label = 'True')
    plt.plot(np.array(preds)*MAX, label = 'Pred')
    plt.legend()


#ratio : train-test 비율 / time = 궤적 개수
'''
    scaler, train-test split
'''

def loadData(data, ratio=0.7, time=50) :
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    # x_data = [minmax_scale(data['feature'].iloc[i]) for i in range(len(data))]
    x_data = [x_scaler.fit_transform(data['feature'].iloc[i]) for i in range(len(data))]

    xDf = {'feature' : x_data}
    trainDummDf = pd.DataFrame(xDf)

    y_data_x = [data['label'].iloc[i][0] for i in range(len(data['label']))]
    y_data_y = [data['label'].iloc[i][1] for i in range(len(data['label']))]

    yData = *y_data_x, *y_data_y
    yData = torch.Tensor(list(yData)).unsqueeze(dim=1)

    yData = y_scaler.fit_transform(yData)

    scaled_yx = yData[:300]
    scaled_yy = yData[300:]

    scaledY = [[*scaled_yx[i], *scaled_yy[i]] for i in range(len(scaled_yy))]

    scaledYDict = {'label' : scaledY}
    scaledY = pd.DataFrame(scaledYDict)

    #todo y 값은 전체 값에 대해서 normalize 한 후 변경하던가 해야 할 듯.
    totalDf =  pd.concat([trainDummDf, scaledY], axis = 1)
    flag = int(len(totalDf)*ratio)
    train = totalDf.iloc[:flag]
    test = totalDf.iloc[flag:].reset_index()

    return train, test, x_scaler, y_scaler


if __name__ == '__main__':

    path = "D:/Semester2201/LAB/Vessel_Trajectory_Prediction-main/"

    with open('./data/data_prof.pickle', 'rb') as f :
        data = pickle.load(f)

    trainData, testData, x_scaler, y_scaler = loadData(data)

    #INIT - model
    #####################
    num_epochs = 20000
    hist = np.zeros(num_epochs)

    input_dim = 100
    hidden_dim = 128
    num_layers = 2
    output_dim = 2

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    train(trainData)
    test(testData)