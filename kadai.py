from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np
import numpy.random as rnd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

import os.path
import time
import datetime
import sys
import traceback
import shutil
import os
import csv

import io_utils
import test
import constant_value as const

import LSTM_model as lstm

import matplotlib.pyplot as plt

##
PREDICT_SEQUENCE_NUM = 100

def make_datas(seq_data):
    seq_data = seq_data.tolist()
    train_datas = []
    target_datas = []
    for i in range(len(seq_data)-PREDICT_SEQUENCE_NUM):
        train_datas.append(seq_data[i:i+PREDICT_SEQUENCE_NUM])
        target_datas.append(seq_data[i+PREDICT_SEQUENCE_NUM])
    return np.reshape(np.array(train_datas)[:,:,1],(-1,100,1)).astype(np.float64),np.reshape(np.array(target_datas)[:,1],(-1,1)).astype(np.float64)

def predict(model, X_test):
    # test
    predict = model.predict(X_test, batch_size=1, verbose=1)

    return predict

##get files
infile = "nyc_taxi.csv"
tmp = np.loadtxt(infile, delimiter=',',dtype="string",skiprows=1)
##normalize
mean_value = np.mean(tmp[:,1].astype(np.float64))
std_value = np.std(tmp[:,1].astype(np.float64))
normal_data_seq = np.c_[tmp[:,0],(tmp[:,1].astype(np.float64)-mean_value)/std_value]
##make dataset
train_datas,target_datas = make_datas(normal_data_seq)
##standardization

##make model
model = Sequential()
model.add(LSTM(20, batch_input_shape=(None, PREDICT_SEQUENCE_NUM, 1), return_sequences=True))
model.add(LSTM(10, batch_input_shape=(None, PREDICT_SEQUENCE_NUM, 1), return_sequences=False))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="adam")

early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
history = model.fit(train_datas, target_datas,nb_epoch=10, batch_size=32, validation_split=0.1, callbacks=[early_stopping])#

##test prediction
predict_result = predict(model,train_datas)

##show
date_row = normal_data_seq[PREDICT_SEQUENCE_NUM:,0].reshape(1,-1)[0]
real_row = tmp[PREDICT_SEQUENCE_NUM:,1].astype(np.float64)
#real_row = normal_data_seq[PREDICT_SEQUENCE_NUM:,1].reshape(1,-1).astype(np.float64)[0]
predict_row = predict_result.reshape(1,-1)[0]
##anti_normalization##
predict_row = predict_row*std_value + mean_value
######################
diff_row = np.absolute(real_row-predict_row)
anomaly_row = np.absolute(real_row-predict_row)
anomaly_row[anomaly_row<10000]=0
anomaly_row[anomaly_row!=0]=1

output_data = np.c_[date_row,real_row,predict_row,diff_row,anomaly_row]
output_data =np.r_[np.array([["timestamp","true","pred","AE","anomal"]]),output_data]
#print(np.array([date_row,real_row,predict_row]).T)
#kakikomi
flout = open("result.csv", 'w')
writer = csv.writer(flout, lineterminator='\n')
writer.writerows(output_data)
flout.close()
