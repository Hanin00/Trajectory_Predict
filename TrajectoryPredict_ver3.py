# ref) https://coding-yoon.tistory.com/190 \

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

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_layers,
                 output_dim):  # num_layers : 2, hidden_dim : 32, input_dim : 1, self : LSTM(1,32,2,batch_firsttrue)
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        self.c1 = nn.Conv1d(in_channels=input_1d.shape[1], out_channels=input_1d.shape[2],
                            kernel_size=input_1d.shape[2], stride=input_1d.shape[1])
        # self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1)  # 1D CNN 레이어 추가
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.seq_len = seq_len

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.seq_len - 1, self.hidden_dim),
            torch.zeros(self.num_layers, self.seq_len - 1, self.hidden_dim)
        )

    def forward(self, sequences):
        #[50, 49, -1]

        sequences = self.c1(sequences.view(len(sequences), 1, -1))

        sequences = self.c1(in_channels=input_1d.shape[1], out_channels=input_1d.shape[2],
                            kernel_size=input_1d.shape[2], stride=input_1d.shape[1])

        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len - 1, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len - 1, len(sequences), self.hidden_dim)[-1]
        y_pred = self.fc(last_time_step)
        return y_pred


def train_model(model, trainX, trainY, val_data=None, val_labels=None, num_epochs=100, verbose=10,
                patience=10):

    trainX_x = [trainX[i][0] for i in range(len(trainX))]
    trainX_y = [trainX[i][1] for i in range(len(trainX))]

    trainXdict = { 'x' : trainX_x,
                   'y' : trainX_y}
    Xdf = pd.DataFrame(trainXdict)

    trainY_x = [trainY[i][0] for i in range(len(trainY))]
    trainY_y = [trainY[i][1] for i in range(len(trainY))]

    trainYdict = { 'x' : trainY_x,
                   'y' : trainY_y}
    Ydf = pd.DataFrame(trainYdict)

    #todo - Xdf.columns = ['x','y']
    loss_fn = torch.nn.L1Loss()  #
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    train_hist = []
    val_hist = []

    xDumm = torch.Tensor(trainX)
    xNpArray = np.array(trainX)


    # xDumm = {'x' : trainX_x}
    # trainx_dumm = pd.DataFrame(xDumm)
    # yDumm = {'x' : trainY_x}
    # trainy_dumm = pd.DataFrame(yDumm)
    # print(len(xDumm)) #210
    # print(len(xDumm[0])) #50

    input_1d = xDumm
    print(input_1d.shape)
    sys.exit()



    # cnn1d_4 = nn.Conv1d(in_channels=input_1d.shape[1], out_channels=input_1d.shape[2], kernel_size=input_1d.shape[2], stride=input_1d.shape[1])
    # print("cnn1d_4: \n")
    # print(cnn1d_4(input_1d).shape, "\n") #torch.Size([210, 2, 1])
    # print(cnn1d_4(input_1d)) #t
    # print(len(cnn1d_4(input_1d))) #210 -> 두 번 째

    for t in range(num_epochs):
        epoch_loss = 0

        for idx, seq in enumerate(xNpArray):  # sample 별 hidden state reset을 해줘야 함

            model.reset_hidden_state()
            # train loss
            # seq = torch.unsqueeze(seq, 0)

            y_pred = model(seq)
            loss = loss_fn(y_pred[0].float(), train_labels[idx])  # 1개의 step에 대한 loss

            # update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        train_hist.append(epoch_loss / len(train_data))

        if val_data is not None:

            with torch.no_grad():

                val_loss = 0
                for val_idx, val_seq in enumerate(val_data):
                    model.reset_hidden_state()  # seq 별로 hidden state 초기화

                    val_seq = torch.unsqueeze(val_seq, 0)
                    y_val_pred = model(val_seq)
                    val_step_loss = loss_fn(y_val_pred[0].float(), val_labels[val_idx])

                    val_loss += val_step_loss

            val_hist.append(val_loss / len(val_data))  # val hist에 추가

            ## verbose 번째 마다 loss 출력
            if t % verbose == 0:
                print(f'Epoch {t} train loss: {epoch_loss / len(train_data)} val loss: {val_loss / len(val_data)}')

            ## patience 번째 마다 early stopping 여부 확인
            if (t % patience == 0) & (t != 0):
                ## loss가 커졌다면 early stop
                if val_hist[t - patience] < val_hist[t]:
                    print('\n Early Stopping')
                    break

        elif t % verbose == 0:
            print(f'Epoch {t} train loss: {epoch_loss / len(train_data)}')

    return model, train_hist, val_hist

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
'''
    scaler, train-test split
'''
def loadData(data, ratio=0.7, time=50) :

    x_x = []
    [x_x.extend(data['feature'].iloc[i][0]) for i in range(len(data))] #50개마다 다른 차량
    x_y = []
    [x_y.extend(data['feature'].iloc[i][1]) for i in range(len(data))]  # 50개마다 다른 차량

    xDict = {'x_x' : x_x,
                'x_y' : x_y}
    xDf = pd.DataFrame(xDict)

    y_x = []
    [y_x.append(data['label'][i][0]) for i in range(len(data['label']))]
    y_y = []
    [y_y.append(data['label'][i][1]) for i in range(len(data['label']))]

    yDict = { 'y_x' : y_x,
              'y_y' : y_y}

    yDf = pd.DataFrame(yDict)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_x_total = scaler.fit_transform(xDf[['x_x', 'x_y']]).tolist()
    scaled_y_total = scaler.fit_transform(yDf[['y_x', 'y_y']]).tolist()

    yflag = int(len(yDf) * ratio)
    xflag = int(len(xDf)//len(yDf)) * yflag  # 300


    trainX = scaled_x_total[:xflag]
    trainY = scaled_y_total[:yflag]
    testX = scaled_x_total[xflag:]
    testY = scaled_y_total[yflag:]

    trainX_x = [trainX[i][0] for i in range(len(trainX))]
    trainX_y = [trainX[i][1] for i in range(len(trainX))]

    trainXdict = {'x': trainX_x,
                  'y': trainX_y}
    Xdf = pd.DataFrame(trainXdict)

    trainY_x = [trainY[i][0] for i in range(len(trainY))]
    trainY_y = [trainY[i][1] for i in range(len(trainY))]

    trainYdict = {'x': trainY_x,
                  'y': trainY_y}
    Ydf = pd.DataFrame(trainYdict)

    # todo - Xdf.columns = ['x','y']

    xDumm = torch.Tensor(trainX)
    print(xDumm.shape)
    sys.exit()



    return trainX, trainY, testX, testY




if __name__ == '__main__':

    path = "D:/Semester2201/LAB/Vessel_Trajectory_Prediction-main/"

    # with open('./data/data_prof.pickle', 'rb') as f :
    # with open('./data/data_ver2_1025.pickle', 'rb') as f :
    with open('./data/listTrainData.pickle', 'rb') as f:
        data = pickle.load(f)

    # trainX, trainY, testXmodel, testY = loadData(data)
    # 나중에 denomalize 할 때 사용하려면, 전체 데이터에 대해 scale 해야 하는 것 아닌가? 어차피 상관 없나?

    trainX, trainY, testX, testY = loadData(data)

    #trainX를 50개씩 끊어서 모델에 넣어야 할 듯. 그래야 sequential이 될 것 같음
    print(len(trainX))
    trainXList = []
    a = []
    cnt = 0
    for i in range(len(trainX)) :
        a.append(trainX[i])
        cnt += 1
        if cnt == 50 :
            trainXList.append(a)
            a = []
            cnt = 0

    #INIT - model
    #####################
    num_epochs = 200
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    # seq_dim = look_back - 1
    input_dim = trainXList.shape
    hidden_dim = 128

    num_layers = 2
    output_dim = 2
    seq_len = 50


    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, seq_len = 50, num_layers=num_layers, output_dim=output_dim)
    # model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, seq_len = 50, num_layers=num_layers, output_dim=output_dim)


    # trainXList : 50개씩 나뉨
    train_model(model, trainXList, trainY)
    # train_model(model, trainX, trainY)
    test(testX, testY)

