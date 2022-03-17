import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow.keras.initializers import HeNormal, Ones, RandomNormal
from tensorflow.keras.layers import Dense, Flatten, InputLayer, BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import L1, L1L2, L2

from visualize import scatter
from preprocess import article_mapper

'''TODO do people buy different stuff depending on what they already bought'''


def build_model(*, norm, shape=(4), regularizer=None):
    """conveniently builds model for different tests"""

    model = Sequential(
        [
            norm,
            # InputLayer(input_shape=shape),
            # BatchNormalization(),
            *[
                Dense(
                    100,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=regularizer,
                    name=f"hidden{i+1}",
                )
                for i in range(3)
            ],
            # BatchNormalization(),
            Dense(105542),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()],
    )
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


def task1(data):
    """
    Task 1: Compare the performance of [2,3,4] layer MLPs on MNIST dataset

    Settings:
    Adam optimizer with 2e-4 learning rate.
    """

    for i in [2, 3, 4]:
        model = build_model(hidden=i, optimizer=Adam(learning_rate=2e-4))
        print(f"\n{i} hidden layers")
        timed_eval(*data, model=model)


def to_days(date_string):
    """returns number of days given a date_string ex: 2018-09-20"""

    dates = [int(item) for item in date_string.split("-")]
    return (dates[0] * 365) + (dates[1] * 30) + dates[2]


def sandbox():
    '''testing things out'''

    path_articles = "articles.csv"
    # articles = pd.read_csv(path_articles)

    path_customers = "customers.csv"
    # customers = pd.read_csv(path_customers)

    path_transactions = "transactions_train.csv"
    transactions = pd.read_csv(path_transactions)
    # transactions['t_dat'] = transactions['t_dat'].apply(lambda x: to_days(x))

    # transactions["customer_id"] = transactions["customer_id"].apply(
    #     lambda x: int(x, 16)
    # )

    # print(transactions['customer_id'])
    # print(transactions['article_id'])
    # print(transactions['price'])
    # print(transactions['sales_channel_id'])

    # remove for now?
    # articles = articles.drop(labels='detail_desc', axis=1)
    transactions = transactions.drop(labels=['t_dat', 'customer_id', 'sales_channel_id'], axis=1)

    print(transactions)
    print(transactions.columns)

    quit()

    fig, ax = plt.subplots()
    ax.scatter(
        articles["product_type_no"].tolist(), articles["garment_group_no"].tolist()
    )
    plt.show()

    # scatter(data)


def process():
    path_articles = "articles.csv"
    articles = pd.read_csv(path_articles)

    path_transactions = "head_transactions_train.csv"
    transactions = pd.read_csv(path_transactions)

    transactions['t_dat'] = transactions['t_dat'].apply(lambda x: to_days(x))
    transactions["customer_id"] = transactions["customer_id"].apply(
        lambda x: int(x, 16)
    )

    map = article_mapper()
    transactions['article_id'] = transactions["article_id"].apply(lambda x: map[x])

    y = transactions['article_id'].to_numpy().astype(float)

    x = transactions.drop(labels='article_id', axis=1).to_numpy().astype(float)
    norm = tf.keras.layers.experimental.preprocessing.Normalization()
    norm.adapt(x)

    # [print(type(x[0][i])) for i in range(4)]
    # quit()
    return x, y, norm


def main():

    x, y, norm = process()

    model = build_model(norm=norm)
    model.summary()
    hist = model.fit(x, y, batch_size=32, epochs=50, verbose=1)


if __name__ == "__main__":
    main()
