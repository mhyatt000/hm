import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def process():

    mode = input('do you want to overwrite previous data? (probably no) (YES/NO): ')
    if mode != 'YES':
        quit()

    # articles.csv
    path_articles = "articles.csv"
    articles = pd.read_csv(path_articles)

    article_ids = articles["article_id"].tolist()

    article_columns = [
        "product_code",
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
        "department_no",
        "index_group_no",
        "garment_group_no",
    ]

    article_columns = [(c, articles[c].tolist()) for c in article_columns]

    article_map = article_hash_map()

    article_info = {
        article_map[x]: {c[0]: c[1][i] for c in article_columns}
        for i, x in enumerate(article_ids)
    }

    # customers.csv
    path_customers = "customers.csv"
    customers = pd.read_csv(path_customers)

    customer_ids = customers["customer_id"].tolist()

    zip_code_map = zip_hash_map()

    """TODO replace null values with 0?? (theres a lot) ... print data"""
    customers["age"] = customers["age"].fillna(0)
    customers["postal_code"] = customers["postal_code"].fillna(0).apply(lambda x: zip_code_map[x])

    customer_columns = [
        # 'FN',
        # 'Active',
        # 'club_member_status',
        # 'fashion_news_frequency',
        "age",
        "postal_code",
    ]

    customer_columns = [(c, customers[c].tolist()) for c in customer_columns]

    customer_map = customer_hash_map()

    customer_info = {
        customer_map[x]: {c[0]: c[1][i] for c in customer_columns}
        for i, x in enumerate(customer_ids)
    }

    # transactions.csv
    path_transactions = "transactions_train.csv"
    transactions = pd.read_csv(path_transactions)

    t_dat = [item.split("-") for item in transactions["t_dat"].tolist()]
    transactions["year"] = [item[0] for item in t_dat]
    transactions["month"] = [item[1] for item in t_dat]
    transactions["day"] = [item[2] for item in t_dat]
    transactions["t_dat"] = transactions["t_dat"].apply(
        lambda x: [int(item) * [365, 30, 1][i] for i, item in enumerate(x.split("-"))]
    )

    # transfer article info to transactions
    transactions["article_id"] = transactions["article_id"].apply(
        lambda x: article_map[x]
    )

    print('adding article info...')
    for c in tqdm(article_columns):
        transactions[c[0]] = transactions["article_id"].apply(
            lambda x: article_info[x][c[0]]
        )

    # transfer customer info to transactions
    transactions["customer_id"] = transactions["customer_id"].apply(
        lambda x: customer_map[x]
    )

    print('adding customer info...')
    for c in tqdm(customer_columns):
        transactions[c[0]] = transactions["customer_id"].apply(
            lambda x: customer_info[x][c[0]]
        )

    transactions.to_csv('data.csv', index=False)


# def load():
#
#     x, y = np.load("data.npy", allow_pickle=True)
#     print(x.shape)
#     norm = tf.keras.layers.experimental.preprocessing.Normalization()
#     norm.adapt(x)
#
#     return x, y, norm


def article_hash_map():
    return generate_map(file="articles.csv", column="article_id")


def customer_hash_map():
    return generate_map(file="customers.csv", column="customer_id")


def zip_hash_map():
    return generate_map(file="customers.csv", column="postal_code")


def generate_map(file, column):
    "indexes "

    data = pd.read_csv(file)

    keys = set(data[column].fillna(0).to_list())
    values = [i for i in values]

    map = {k:v for k,v in zip(keys, values)}
    return map



def main():
    pass
    # process()

if __name__ == "__main__":
    main()
