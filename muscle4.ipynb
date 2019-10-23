{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/pandas/compat/__init__.py:84: UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.\n",
      "  warnings.warn(msg)\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/pandas/compat/__init__.py:84: UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.\n",
      "  warnings.warn(msg)\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import glob\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# datasetsフォルダ内にあるcsvファイルの一覧を取得\n",
    "files1 = glob.glob(\"../../../Desktop/GHELIA/myoelectricity/datasets4/inside/muscle*.txt\")\n",
    "files2 = glob.glob(\"../../../Desktop/GHELIA/myoelectricity/datasets4/outside/muscle*.txt\")\n",
    "files3 = glob.glob(\"../../../Desktop/GHELIA/myoelectricity/datasets4/fist/muscle*.txt\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "f1_np =[]\n",
    "fa1 = []\n",
    "for i in range(150):\n",
    "    f1 = open(files1[i])\n",
    "    f1_array = f1.read().split()\n",
    "    f1_np = np.append(f1_np, f1_array)\n",
    "    for j in range(1000):\n",
    "        a1 = f1_np[j].split(\",\")\n",
    "        fa1 = np.append(fa1, a1)\n",
    "fr1 = []\n",
    "for i in range(fa1.shape[0]):\n",
    "    fr1 = np.append(fr1,float(fa1[i]))\n",
    "inside = fr1.reshape(150,1000,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "f2_np =[]\n",
    "fa2 = []\n",
    "for i in range(150):\n",
    "    f2 = open(files2[i])\n",
    "    f2_array = f2.read().split()\n",
    "    f2_np = np.append(f2_np, f2_array)\n",
    "    for j in range(1000):\n",
    "        a2 = f2_np[j].split(\",\")\n",
    "        fa2 = np.append(fa2, a2)\n",
    "fr2 = []\n",
    "for i in range(fa2.shape[0]):\n",
    "    fr2 = np.append(fr2,float(fa2[i]))\n",
    "outside = fr2.reshape(150,1000,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "f3_np =[]\n",
    "fa3 = []\n",
    "for i in range(150):\n",
    "    f3 = open(files3[i-1])\n",
    "    f3_array = f3.read().split()\n",
    "    f3_np = np.append(f3_np, f3_array)\n",
    "    for j in range(1000):\n",
    "        a3 = f3_np[j-1].split(\",\")\n",
    "        fa3 = np.append(fa3, a3)\n",
    "fr3 = []\n",
    "fa3 = np.array(fa3)\n",
    "fr3 = np.array(fr3)\n",
    "for k in range(fa3.shape[0]):\n",
    "    fr3 = np.append(fr3,float(fa3[k]))\n",
    "fist = fr3.reshape(150,1000,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = []\n",
    "for i in range(0,105):\n",
    "    x_train = np.append(x_train, inside[i])\n",
    "    x_train = np.append(x_train, outside[i])\n",
    "    x_train = np.append(x_train, fist[i])\n",
    "x_test = []\n",
    "for j in range(105,150):\n",
    "    x_test = np.append(x_test, inside[j])\n",
    "    x_test = np.append(x_test, outside[j])\n",
    "    x_test = np.append(x_test, fist[j])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = x_train.reshape(315,1000,2)\n",
    "x_test = x_test.reshape(135,1000,2)\n",
    "y_train = np.array([0,1,2]*105)\n",
    "y_test = np.array([0,1,2]*45)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from scipy.fftpack import fft\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = np.zeros((315,1000))\n",
    "for i in range(315):\n",
    "    for j in range(1000):\n",
    "        X_train_pre = x_train[i][j][1]\n",
    "        X_train[i][j] = X_train_pre\n",
    "\n",
    "X_test = np.zeros((135,1000))\n",
    "for i in range(135):\n",
    "    for j in range(1000):\n",
    "        X_test_pre = x_test[i][j][1]\n",
    "        X_test[i][j] = X_test_pre"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def wave_plot_fft(f):\n",
    "    N =  1000\n",
    "    dt = 0.01\n",
    "    freq = np.linspace(0, 1.0/dt, N)\n",
    "    yf = f/(N/2)\n",
    "    \n",
    "    plt.figure(2)\n",
    "    plt.plot(freq, np.abs(yf))\n",
    "    plt.xlabel('frequency')\n",
    "    plt.ylabel('amplitude')\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_pre = []\n",
    "for i in range(315):\n",
    "    x_train_pre = np.append(x_train_pre,fft(X_train[i]))\n",
    "\n",
    "X_train_fft = x_train_pre.reshape(315,1000)\n",
    "x_test_pre = []\n",
    "for i in range(135):\n",
    "    x_test_pre = np.append(x_test_pre,fft(X_test[i]))\n",
    "\n",
    "X_test_fft = x_test_pre.reshape(135,1000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n"
     ]
    }
   ],
   "source": [
    "import keras\n",
    "from keras.models import Model, Sequential\n",
    "from keras.layers import Dense, Dropout, Activation, Flatten, Input\n",
    "from keras.layers import BatchNormalization, Add\n",
    "from keras.layers.convolutional import Conv1D, MaxPooling1D\n",
    "from keras.layers.pooling import MaxPool1D\n",
    "from keras.callbacks import EarlyStopping, ModelCheckpoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 64\n",
    "num_classes = 3\n",
    "epochs = 50"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_tf = np.reshape(X_train_fft, (X_train_fft.shape[0], -1, 1))\n",
    "X_test_tf = np.reshape(X_test_fft, (X_test_fft.shape[0], -1, 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_tf = keras.utils.to_categorical(y_train, num_classes)\n",
    "y_test_tf = keras.utils.to_categorical(y_test, num_classes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "\n",
    "model.add(Conv1D(filters=128, input_shape=(X_train_fft.shape[1],1), kernel_size=2, strides=1, padding='same'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool1D(pool_size=2))\n",
    "model.add(Activation('relu'))\n",
    "\n",
    "model.add(Conv1D(filters=128, kernel_size=2, strides=1, padding='same'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool1D(pool_size=2))\n",
    "model.add(Activation('relu'))\n",
    "\n",
    "model.add(Conv1D(filters=128, kernel_size=2, strides=1, padding='same'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool1D(pool_size=2))\n",
    "model.add(Activation('relu'))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(512))\n",
    "model.add(Activation('relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(num_classes))\n",
    "model.add(Activation('softmax'))\n",
    "\n",
    "opt = keras.optimizers.rmsprop()\n",
    "model.compile(loss = 'categorical_crossentropy',\n",
    "              optimizer = opt,\n",
    "              metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "es = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W1017 15:43:33.123351 4649948608 deprecation.py:323] From /Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.where in 2.0, which has the same broadcast rule as np.where\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 315 samples, validate on 135 samples\n",
      "Epoch 1/50\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/megu.kato/.pyenv/versions/3.6.1/lib/python3.6/site-packages/numpy/core/_asarray.py:85: ComplexWarning: Casting complex values to real discards the imaginary part\n",
      "  return array(a, dtype, copy=False, order=order)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "315/315 [==============================] - 3s 10ms/step - loss: 5.9861 - acc: 0.4571 - val_loss: 3.8099 - val_acc: 0.6667\n",
      "Epoch 2/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 1.7227 - acc: 0.8254 - val_loss: 3.8742e-05 - val_acc: 1.0000\n",
      "Epoch 3/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 0.0103 - acc: 0.9968 - val_loss: 4.4717e-06 - val_acc: 1.0000\n",
      "Epoch 4/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 2.5465e-04 - acc: 1.0000 - val_loss: 2.9683e-06 - val_acc: 1.0000\n",
      "Epoch 5/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 1.8737e-04 - acc: 1.0000 - val_loss: 1.4314e-06 - val_acc: 1.0000\n",
      "Epoch 6/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 3.1997e-04 - acc: 1.0000 - val_loss: 5.9605e-07 - val_acc: 1.0000\n",
      "Epoch 7/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 2.9789e-04 - acc: 1.0000 - val_loss: 2.3842e-07 - val_acc: 1.0000\n",
      "Epoch 8/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 8.3549e-05 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 9/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 3.2309e-05 - acc: 1.0000 - val_loss: 1.3908e-07 - val_acc: 1.0000\n",
      "Epoch 10/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 2.9483e-04 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 11/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 3.1507e-05 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 12/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 5.4449e-05 - acc: 1.0000 - val_loss: 1.5895e-07 - val_acc: 1.0000\n",
      "Epoch 13/50\n",
      "315/315 [==============================] - 2s 6ms/step - loss: 9.8148e-05 - acc: 1.0000 - val_loss: 1.9868e-07 - val_acc: 1.0000\n"
     ]
    }
   ],
   "source": [
    "history_fft = model.fit(X_train_tf, y_train_tf,\n",
    "                       batch_size = batch_size,\n",
    "                       epochs = epochs,\n",
    "                       verbose =1,\n",
    "                       validation_data = (X_test_tf, y_test_tf),\n",
    "                       callbacks = [es])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test loss: 1.986821767256212e-07\n",
      "Test accuracy: 1.0\n"
     ]
    }
   ],
   "source": [
    "score = model.evaluate(X_test_tf, y_test_tf, verbose=0)\n",
    "print('Test loss:', score[0])\n",
    "print('Test accuracy:', score[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXgAAAEWCAYAAABsY4yMAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3deXxV5b3v8c8vAwmQHcZAEkADVkgoSKIBnHDqXKfWodZrVazK8ZzWtrfW2t72tKi9p31Ve+yp9h6LE2iR49habWsHFcGhaKCgSECtgAYCBGQIhECG3/1j76SIDHsne2Vnr/19v155ZWfvNfyWwpcnz/OsZ5m7IyIi4ZOV6gJERCQYCngRkZBSwIuIhJQCXkQkpBTwIiIhpYAXEQkpBbwIYGazzexHcW67xsw+3t3jiARNAS8iElIKeBGRkFLAS9qIdY1cb2avmdkuM7vHzIab2R/NrNHM/mpmg/bZ/hwze8PMtpnZfDOr2OezKjNbEtvvISB/v3OdZWZLY/u+ZGbHdLHmq83sbTN738x+Z2alsffNzG4zs01mtsPMXjezCbHPPmtmK2K1rTOzb3XpP5hkPAW8pJvzgU8AY4GzgT8C/wcoIvrn+WsAZjYWmAd8I/bZH4AnzayPmfUBfgs8AAwGHokdl9i+VcC9wL8AQ4BfAb8zs7xECjWzM4AfA18ASoC1wP/EPv4kcErsOgbEttkS++we4F/cPQJMAJ5N5LwiHRTwkm5ud/eN7r4OWAgscve/u3sz8BugKrbdRcDv3f0v7t4C3Ar0BU4EjgdygZ+7e4u7Pwq8us85ZgC/cvdF7t7m7nOAPbH9EnEJcK+7L3H3PcB3gRPMrAxoASJAOWDuXuvu9bH9WoDxZlbo7lvdfUmC5xUBFPCSfjbu83r3AX4uiL0uJdpiBsDd24H3gBGxz9b5B1faW7vP6yOB62LdM9vMbBswKrZfIvavYSfRVvoId38WuAP4JbDJzGaZWWFs0/OBzwJrzex5MzshwfOKAAp4Ca/1RIMaiPZ5Ew3pdUA9MCL2Xocj9nn9HvB/3X3gPl/93H1eN2voT7TLZx2Au//C3Y8DxhPtqrk+9v6r7n4uMIxoV9LDCZ5XBFDAS3g9DJxpZh8zs1zgOqLdLC8BLwOtwNfMLNfMzgOm7LPvXcA1ZjY1Nhja38zONLNIgjXMA64ws8pY//1/EO1SWmNmk2PHzwV2Ac1Ae2yM4BIzGxDrWtoBtHfjv4NkMAW8hJK7rwK+BNwObCY6IHu2u+91973AecB04H2i/fWP77NvDXA10S6UrcDbsW0TreGvwL8DjxH9reEo4IuxjwuJ/kOylWg3zhbglthnlwJrzGwHcA3RvnyRhJke+CEiEk5qwYuIhJQCXkQkpBTwIiIhpYAXEQmpnFQXsK+hQ4d6WVlZqssQEUkbixcv3uzuRQf6rFcFfFlZGTU1NakuQ0QkbZjZ2oN9pi4aEZGQUsCLiISUAl5EJKR6VR+8iCSmpaWFuro6mpubU12KBCw/P5+RI0eSm5sb9z4KeJE0VldXRyQSoaysjA8ujilh4u5s2bKFuro6Ro8eHfd+gXbRmNlAM3vUzFaaWa3WtRZJrubmZoYMGaJwDzkzY8iQIQn/phZ0C/6/gKfd/YLYY9L6BXw+kYyjcM8MXfn/HFgL3swGEH3m5D0AsWVatyX7PC1t7fy/+W+z4M2GZB9aRCStBdlFMxpoAO4zs7+b2d2xJ9p8gJnNMLMaM6tpaEg8pHOyjFkL3uGPy+sPv7GIJF1BQcHhN+qChoYGpk6dSlVVFQsXLuzSMWbPns369esT3u/OO+/k/vvvP+Q2NTU1fO1rX+tSXT0lyC6aHOBY4Fp3X2Rm/wV8h+gDEDq5+yxgFkB1dXXCi9ObGeXFEWrrG5NQsoj0Fs888wwTJ07k7rvvjnuftrY2srOzO3+ePXs2EyZMoLT0w4/T3X/bfV1zzTWHPVd1dTXV1dVx15YKQbbg64A6d18U+/lRooGfdOXFhaza0Eh7ux5eIpIq7s7111/PhAkTmDhxIg899BAA9fX1nHLKKVRWVjJhwgQWLlxIW1sb06dP79z2tttu+8Cxli5dyre//W2eeOIJKisr2b17N/PmzWPixIlMmDCBG264oXPbgoICrrvuOiZNmsTLL7/c+f6jjz5KTU0Nl1xySecxysrKuOGGGzj22GN55JFHuOuuu5g8eTKTJk3i/PPPp6mpCYCZM2dy6623AnDaaadxww03MGXKFMaOHdv528T8+fM566yzOrf/8pe/zGmnncaYMWP4xS9+0VnHzTffzLhx4zj55JO5+OKLO4/bEwJrwbv7BjN7z8zGxR6f9jFgRRDnGl9SyO6WNt59v4myoR/qBRLJCDc++QYr1u9I6jHHlxbyw7M/Gte2jz/+OEuXLmXZsmVs3ryZyZMnc8opp/Dggw/yqU99iu9973u0tbXR1NTE0qVLWbduHcuXLwdg27YPDs9VVlZy0003UVNTwx133MH69eu54YYbWLx4MYMGDeKTn/wkv/3tb/nc5z7Hrl27mDp1Kj/72c8+cIwLLriAO+64g1tvvfUDLe0hQ4awZMkSALZs2cLVV18NwPe//33uuecerr322g9dW2trK6+88gp/+MMfuPHGG/nrX//6oW1WrlzJc889R2NjI+PGjeNf//VfWbp0KY899hjLli2jpaWFY489luOOOy6u/57JEPSdrNcCc83sNaCS6EOHk668JPos5JUbkvuHW0Ti98ILL3DxxReTnZ3N8OHDOfXUU3n11VeZPHky9913HzNnzuT1118nEokwZswY3nnnHa699lqefvppCgsLD3nsV199ldNOO42ioiJycnK45JJLWLBgAQDZ2dmcf/75cdd50UUXdb5evnw506ZNY+LEicydO5c33njjgPucd955ABx33HGsWbPmgNuceeaZ5OXlMXToUIYNG8bGjRt58cUXOffcc8nPzycSiXD22WfHXWcyBDpN0t2XAoF3Uh09LEKWQW19I5+eUBL06UR6pXhb2j3tlFNOYcGCBfz+979n+vTpfPOb3+Syyy5j2bJl/OlPf+LOO+/k4Ycf5t577+3S8fPz8w/al34g/fv/87f86dOn89vf/pZJkyYxe/Zs5s+ff8B98vLygOg/Jq2trYfc5nDb9aRQrEXTt082ZUP7qwUvkkLTpk3joYceoq2tjYaGBhYsWMCUKVNYu3Ytw4cP5+qrr+aqq65iyZIlbN68mfb2ds4//3x+9KMfdXaZHMyUKVN4/vnn2bx5M21tbcybN49TTz31sDVFIhEaGw8+AaOxsZGSkhJaWlqYO3duwtd8OCeddBJPPvkkzc3N7Ny5k6eeeirp5ziU0CxVUFFcyPL121NdhkjG+vznP8/LL7/MpEmTMDN++tOfUlxczJw5c7jlllvIzc2loKCA+++/n3Xr1nHFFVfQ3t4OwI9//ONDHrukpISf/OQnnH766bg7Z555Jueee+5ha5o+fTrXXHMNffv2/cAAbIebb76ZqVOnUlRUxNSpUw/5j0FXTJ48mXPOOYdjjjmG4cOHM3HiRAYMGJDUcxyKufeemSfV1dXe1Qd+3P7MW/zsL2+y/MZPUZAXmn+3RA6ptraWioqKVJchh7Bz504KCgpoamrilFNOYdasWRx7bNcmFB7o/7eZLXb3A3aFhyYJy0uigzSrNjRy3JGDUlyNiEjUjBkzWLFiBc3NzVx++eVdDveuCE/AF/9zJo0CXkR6iwcffDBl5w7FICvAyEF9ieTlsFJ3tIqIACEKeDOjvCSimTQiIjGhCXiILlmwsr6R3jRwLCKSKuEK+JIIjXtaWbdtd6pLERFJuXAFfHF0Jo364UV6Tm9eLjhR06dP59FHHwXgqquuYsWKDy+fNXv2bL761a8e8jjz58/npZde6vw5nuWHgxCaWTQA42IzaWrrd/Dx8cNTXI2IdEcylgvujkTOu7/58+dTUFDAiSeeCMS3/HAQQtWCL8jL4YjB/Vi5QS14kZ7W25YLXrlyJVOmTOn8ec2aNUycOBGAm266icmTJzNhwgRmzJhxwHG70047jY4bL++77z7Gjh3LlClTePHFFzu3efLJJzt/y/j4xz/Oxo0bWbNmDXfeeSe33XYblZWVLFy48APLDy9dupTjjz+eY445hs9//vNs3bq183wHWpa4O0LVgofofPhazaSRTPTH78CG15N7zOKJ8JmfxLVpb1suuLy8nL1797J69WpGjx7NQw891LmS5Fe/+lV+8IMfAHDppZfy1FNPHXSlx/r6en74wx+yePFiBgwYwOmnn05VVRUAJ598Mn/7298wM+6++25++tOf8rOf/YxrrrmGgoICvvWtbwHR30Y6XHbZZdx+++2ceuqp/OAHP+DGG2/k5z//ORDfssSJCFULHqCipJA1m3exe29bqksRySi9cbngL3zhC52/Sewb8M899xxTp05l4sSJPPvsswddJhhg0aJFnefu06fPB5Ybrqur41Of+hQTJ07klltuOeRxALZv3862bds6F0q7/PLLO68D4luWOBGha8FXlERod3hrUyPHjByY6nJEek6cLe2elsrlgi+66CIuvPBCzjvvPMyMo48+mubmZv7t3/6NmpoaRo0axcyZM2lubu7Sua+99lq++c1vcs455zB//nxmzpzZpeN0iGdZ4kSErgWvmTQiqdEblws+6qijyM7O5uabb+5seXeE+dChQ9m5c2fnrJmDmTp1Ks8//zxbtmyhpaWFRx55pPOz7du3M2LECADmzJnT+f7BlikeMGAAgwYN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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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NoRu54Wy+Q8+38wz//6S607uxGldWpW9gohIgkILAne/uoz5DnwvrP1XqYI8yHoZup0P9Sv/2sCub/K5a+qndGrZkHvOq6YXyqXc8vLyyM7OZt++fckuRWqRtLQ0OnToQN26ibca1IiLxdXe57Nh7xboHU6z0MP/WsbabXuZcvPJNKyn/2W1RXZ2Nk2aNKFz586YrspLJXB3Nm/eTHZ2Nscdd1zC66nTucqwZCqkNYMThlf6pmev2MikD79m3OnHM6Bzi0rfviTPvn37aNmypUJAKo2Z0bJly3KfZSoIjtT+PbDsVeh5CaTWK3v5cti2Zz93T11E1zaNuePsrpW6bakeFAJS2SryN6V2hiP12euQtxt6j670TT84I4stu/fz5+sHklY3pdK3LyICOiM4counQZO20OmUSt3s64vX8/LCdYw/60R6d2haqdsWqS3mzp1Lr169OOmkk9i7d2+FtvHLX/6yQuvdeOONLF269LDLPPnkkzz//PMV2n5VqnHdUFerJ4v3boVfd4HBN8O5D1faZjft+oZz/ncO7ZqlMf22U6mboryujZYtW0aPHj2SXUaVyM/PJzW18hsgbrnlFk477TSuueaaCtfRuHFjdu3addCy7o67U6dOzfv3V9rflpl95O4DSlu+5h1hdbLsn1CYV6l3C7k7P35pMbv25fPbK09SCEioLr30Uvr370+vXr2YOHFi8fQ33niDfv36kZGRwfDhsZsgdu3axQ033EDv3r3p06cP06ZNA2IfpEWmTp3K2LFjARg7diy33HILgwcPZsKECXz44YecfPLJ9O3bl1NOOYUVK1YAUFBQwJ133kl6ejp9+vThD3/4A2+//TaXXnpp8XZnzpzJqFGjDqj96aefZvLkydx///2MGTMGd+euu+4iPT2d3r178+KLsQHb3nnnHU4//XQuvvhievbsecA27rnnHvbu3ctJJ53EmDFjWL16Nd26deO6664jPT2dNWvWcOuttzJgwAB69erFAw88ULzu0KFDKfpS2rhxY37yk5+QkZHBkCFD2LBhAwAPPvggjz32WPHyd999N4MGDaJr167MnTsXgD179nDllVfSs2dPRo0axeDBg6nqL7u6RnAkFk+BFidA25MqbZPTP1nLm0s3cO953enaJpGhc6U2+Nk/s1i6bkelbrNnu6N44KJeh13mmWeeoUWLFuzdu5eBAwdy+eWXU1hYyE033cScOXM47rjj2LIl1onwz3/+c5o2bcrixYsB2Lp1a5k1ZGdn8/7775OSksKOHTuYO3cuqampzJo1ix//+MdMmzaNiRMnsnr1ahYuXEhqaipbtmyhefPm3HbbbeTm5tK6dWv+8pe/8J3vfOeAbd9444289957XHjhhVxxxRVMmzaNhQsX8umnn7Jp0yYGDhzIGWecAcDHH3/MkiVLDrql8pFHHuGPf/wjCxcuBGD16tWsXLmS5557jiFDhgDw8MMP06JFCwoKChg+fDiLFi2iT58+B2xn9+7dDBkyhIcffpgJEybwpz/9ifvuu++g30d+fj4ffvghr732Gj/72c+YNWsWTzzxBM2bN2fp0qUsWbKEk06qvM+TROnrZkXtWA9fzo1dJK6kOz/Wb9/LAzOyGNCpOTeefnylbFPkcH7/+98Xf4tds2YNK1euZN68eZxxxhnFH5otWsRuW541axbf+95/nwFt3rx5mdsfPXo0KSmxGx22b9/O6NGjSU9P54477iArK6t4uzfffHNxk02LFi0wM6699lr+9re/sW3bNj744APOO+/wT+2/9957XH311aSkpNCmTRvOPPNM5s+fD8CgQYMSvq++U6dOxSEAMHnyZPr160ffvn3Jysoq9bpAvXr1uPDCCwHo378/q1evLnXbl1122UHLvPfee1x11VUAxWdFVU1nBBWVNR3wSmsWcncmTF1EfoHz2OgMUurotsIoKeubexjeeecdZs2axQcffEDDhg0ZOnRohZ5yjr9dseT6jRo1Kn59//33M2zYMKZPn87q1asZOnToYbd7ww03cNFFF5GWlsbo0aOP6BpDfB3lWfbLL7/kscceY/78+TRv3pyxY8eW+juqW7du8e8hJSWF/Pz8Urddv379MpdJBp0RVNTiKdA2A1p1qZTN/f3Dr5m7chM/Pr87nVsl/kcrUlHbt2+nefPmNGzYkOXLlzNv3jwAhgwZwpw5c/jyyy8BipuGzj77bB5//PHi9Yuahtq0acOyZcsoLCxk+vTph91f+/btAXj22WeLp5999tk89dRTxR+MRftr164d7dq14xe/+AU33FB2V2Snn346L774IgUFBeTm5jJnzhwGDRpU5np169YlLy+v1Hk7duygUaNGNG3alA0bNvD666+Xub3yOvXUU5k8eTIAS5cuLW56q0oKgorY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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_history_lost(history):\n",
    "    plt.plot(history.history['loss'], label='loss for training')\n",
    "    plt.plot(history.history['val_loss'], label='loss for validation')\n",
    "    plt.title('model loss')\n",
    "    plt.xlabel('epoch')\n",
    "    plt.ylabel('loss')\n",
    "    plt.legend(loc='best')\n",
    "    plt.show()\n",
    "    \n",
    "def plot_history_acc(history):\n",
    "    plt.plot(history.history['acc'], label='accuracy for training')\n",
    "    plt.plot(history.history['val_acc'], label='accuracy for validation')\n",
    "    plt.title('model accuracy')\n",
    "    plt.xlabel('epoch')\n",
    "    plt.ylabel('accuracy')\n",
    "    plt.legend(loc='best')\n",
    "    plt.ylim([0,1])\n",
    "    plt.show()\n",
    "    \n",
    "plot_history_lost(history_fft)\n",
    "plot_history_acc(history_fft)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('myoelectricity4.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
