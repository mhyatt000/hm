import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow.keras.initializers import HeNormal, Ones, RandomNormal
from tensorflow.keras.layers import Dense, Flatten, InputLayer, BatchNormalization, LSTM
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import L1, L1L2, L2

from visualize import scatter
import preprocess

'''TODO do people buy different stuff depending on what they already bought'''


def build_recurrent(norm):
    '''TODO feature:
    - time since last purchase
    - total number of purchases
    - total ammount of money spent
    (RFM)
    '''

    model = Sequential(
        [
            norm,
            LSTM(105542, recurrent_activation='relu', return_sequences=True),
        ]
    )

    model = compile_model(model)
    return model

def compile_model(model):

    model.compile(
        optimizer="adam",
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()],
    )
    return model

def build_model(*, norm, shape=(4), regularizer=None):
    """conveniently builds model for different tests"""

    model = Sequential(
        [
            norm,
            # InputLayer(input_shape=shape),
            Dense(4096, activation="relu", kernel_initializer="glorot_uniform"),
            BatchNormalization(),
            Dense(4096, activation="relu", kernel_initializer="glorot_uniform"),
            BatchNormalization(),
            Dense(128, activation="relu", kernel_initializer="glorot_uniform"),
            BatchNormalization(),
            # LSTM or GRU ... ouput 12 of em
            Dense(105542),
        ]
    )

    model = compile_model(model)
    return model


def timed_eval(train_x, train_y, test_x, test_y, *, model):

    start = time.time()

    hist = model.fit(train_x, train_y, batch_size=32, epochs=50, verbose=0)
    results = model.evaluate(test_x, test_y, batch_size=32, verbose=0)

    stop = time.time()

    pprint(
        {
            "train accuracy": round(hist.history["sparse_categorical_accuracy"][-1], 4),
            "test accuracy": round(results[1], 4),
            "time": int(stop - start),
        }
    )


def main():

    x,y,norm = preprocess.load()

    x = x.reshape(-1,32,18)
    y = y.reshapre(-1,32,1)

    # try:
    #     x, y, norm = preprocess.load()
    #     print('loaded data')
    # except:
    #     x, y, norm = preprocess.process()


    print(x.shape)
    print(y.shape)
    quit()

    model = build_model(norm=norm)
    model.summary()
    hist = model.fit(x, y, batch_size=128, epochs=50, verbose=1)  # was 32 batch


if __name__ == "__main__":
    main()
