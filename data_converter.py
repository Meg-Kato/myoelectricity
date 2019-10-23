#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import RPi.GPIO as GPIO


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

# 0.1秒インターバルの永久ループ
for j in range(1000):

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

        #測定結果を標準出力
        if ch > 0:
            sys.stdout.write(" ")

        GPIO.output(spi_ss, True)
        sys.stdout.write(str(value))
        
        dt = round(0.001 * j,3)
        path = './datasets/inside/muscle347.txt'
        with open(path, mode='a') as f:
            f.write("{},{}".format(str(dt),str(value)) )
            f.write("\n")
        f.close()
        
        
    sys.stdout.write("\n")


