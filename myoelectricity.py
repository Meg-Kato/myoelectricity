#!/usr/bin/env python
# -*- coding: utf-8 -*-

#必要なmoduleの読み込み
import os
import glob
import numpy as np
from scipy.fftpack import fft
# import matplotlib.pyplot as plt
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import BatchNormalization, Add
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.pooling import MaxPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import random


# datasetsフォルダ内にあるcsvファイルの一覧を取得
files1 = glob.glob("./datasets/inside/muscle*.txt")
files2 = glob.glob("./datasets/outside/muscle*.txt")
files3 = glob.glob("./datasets/fist/muscle*.txt")

#inside/outside/fistの配列化
f1_np =[]
fa1 = []
for i in range(50):
    f1 = open(files1[i])
    f1_array = f1.read().split()
    f1_np = np.append(f1_np, f1_array)
    for j in range(1000):
        a1 = f1_np[j].split(",")
        fa1 = np.append(fa1, a1)
fr1 = []
for i in range(fa1.shape[0]):
    fr1 = np.append(fr1,float(fa1[i]))
inside = fr1.reshape(50,1000,2)

f2_np =[]
fa2 = []
for i in range(50):
    f2 = open(files2[i])
    f2_array = f2.read().split()
    f2_np = np.append(f2_np, f2_array)
    for j in range(1000):
        a2 = f2_np[j].split(",")
        fa2 = np.append(fa2, a2)
fr2 = []
for i in range(fa2.shape[0]):
    fr2 = np.append(fr2,float(fa2[i]))
outside = fr2.reshape(50,1000,2)

f3_np =[]
fa3 = []
for i in range(50):
    f3 = open(files3[i])
    f3_array = f1.read().split()
    f3_np = np.append(f3_np, f3_array)
    for j in range(1000):
        a3 = f3_np[j].split(",")
        fa3 = np.append(fa3, a3)
fr3 = []
for i in range(fa3.shape[0]):
    fr3 = np.append(fr3,float(fa3[i]))
fist = fr3.reshape(50,1000,2)

#ホールドアウト法による検査と検証データへの振り分け＆ラベルづけ
x_train = []
for i in range(0,35):
    x_train = np.append(x_train, inside[i])
    x_train = np.append(x_train, outside[i])
    x_train = np.append(x_train, fist[i])
x_test = []
for j in range(35,50):
    x_test = np.append(x_test, inside[j])
    x_test = np.append(x_test, outside[j])
    x_test = np.append(x_test, fist[j])

x_train = x_train.reshape(105,1000,2)
x_test = x_test.reshape(45,1000,2)
y_train = np.array([0,1,2]*35)
y_test = np.array([0,1,2]*15)


#FFT
X_train = np.zeros((105,1000))
for i in range(105):
    for j in range(1000):
        X_train_pre = x_train[i][j][1]
        X_train[i][j] = X_train_pre

X_test = np.zeros((45,1000))
for i in range(45):
    for j in range(1000):
        X_test_pre = x_test[i][j][1]
        X_test[i][j] = X_test_pre

x_train_pre = []
for i in range(105):
    x_train_pre = np.append(x_train_pre,fft(X_train[i]))

X_train_fft = x_train_pre.reshape(105,1000)
x_test_pre = []
for i in range(45):
    x_test_pre = np.append(x_test_pre,fft(X_test[i]))

X_test_fft = x_test_pre.reshape(45,1000)

#学習を回したデータの読み込み
model = load_model('myoelectricity.h5')

#予想と正解の比較
index = random.randint(0,44)
sample = X_test_fft[index]
sample = sample.reshape(1,1000,1)
predict = model.predict(sample)

y_pred = np.argmax(predict, axis=1)
print('predict: ', y_pred[0])
print('answer: ', y_test[index])