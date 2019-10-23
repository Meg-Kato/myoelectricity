#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import RPi.GPIO as GPIO
import numpy as np
import keras
from keras.models import load_model
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# GPIOの番号の定義。
# 上記回路図では、Raspberry Pi上の特定のピンに接続するかのように説明しましたが、
# たぶんここで定義する番号と接続するGPIOの番号が合っていれば動く気がする。
spi_clk  = 11
spi_mosi = 10
spi_miso = 9
spi_ss   = 8

# RPiモジュールの設定
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# GPIOデバイスの設定
GPIO.setup(spi_mosi, GPIO.OUT)
GPIO.setup(spi_miso, GPIO.IN)
GPIO.setup(spi_clk,  GPIO.OUT)
GPIO.setup(spi_ss,   GPIO.OUT)

#学習を回したデータの読み込み
model = load_model('myoelectricity3.h5')

num = 0

# def wave_plot_fft(f):
#     global num
#     N =  1000
#     dt = 0.01
#     freq = np.linspace(0, 1.0/dt, N)
#     yf = f/(N/2)
    
#     plt.figure(2)
#     plt.plot(freq, np.abs(yf))
#     plt.xlabel('frequency')
#     plt.ylabel('amplitude')
#     plt.tight_layout()
#     num += 1
#     file_name = str(num)+'.png'
#     plt.savefig(file_name)


def predict():
    print("start!!!!!!!!")

    array = []

    # 0.1秒インターバルの永久ループ
    for i in range(1000):

        time.sleep(0.001)

        # 8 チャンネル分のループ
        for ch in range(1):
            GPIO.output(spi_ss,   False)
            GPIO.output(spi_clk,  False)
            GPIO.output(spi_mosi, False)
            GPIO.output(spi_clk,  True)
            GPIO.output(spi_clk,  False)

            # 測定するチャンネルの指定をADコンバータに送信
            cmd = (ch | 0x18) << 3
            for i in range(5):
                if (cmd & 0x80):
                    GPIO.output(spi_mosi, True)
                else:
                    GPIO.output(spi_mosi, False)
                cmd <<= 1
                GPIO.output(spi_clk, True)
                GPIO.output(spi_clk, False)
            GPIO.output(spi_clk, True)
            GPIO.output(spi_clk, False)
            GPIO.output(spi_clk, True)
            GPIO.output(spi_clk, False)

            # 12ビットの測定結果をADコンバータから受信
            value = 0
            for i in range(12):
                value <<= 1
                GPIO.output(spi_clk, True)
                if (GPIO.input(spi_miso)):
                    value |= 0x1
                GPIO.output(spi_clk, False)

            # 測定結果を標準出力
            if ch > 0:
                sys.stdout.write(" ")

            GPIO.output(spi_ss, True)
            sys.stdout.write(str(value))

            #arrayに信号をを追加
            array.append(value)
        sys.stdout.write("\n")

    #arrayをnp.array型に変換
    array = np.array(array)

    # wave_plot_fft(array)

    X_test_fft = fft(array)

    #予想を出力
    sample = X_test_fft.reshape(1,1000,1)
    predict = model.predict(sample)

    y_pred = np.argmax(predict, axis=1)

    if y_pred[0] == 0:
        print("inside")
    elif y_pred[0] == 1:
        print("outside")
    else:
        print("fist")

    print('predict: ', y_pred[0])

while True:
    print("3\n")
    time.sleep(1)
    print("2\n")
    time.sleep(1)
    print("1\n")
    time.sleep(1)
    predict()

