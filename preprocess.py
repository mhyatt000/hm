import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

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

    transactions.to_csv('data.csv')

def batch():

    print('reading in data')
    transactions = pd.read_csv('data.csv')

    print('batching data...')
    y, x = np.array([]), np.array([])
    customers = tqdm(list(set(transactions["customer_id"].tolist())))

    # batch on n CPUs
    with ProcessPoolExecutor() as executor:

        '''TODO:
        for item in transactions
            put in dict by id as key
        then for key
            same same
            o(2n) vs o(n*n)
        '''

        data = list(tqdm(executor.map(_batch_customer, customers, repeat(transactions)), total=len(customers)))

    x = [item[0] for item in data if item[0]]
    y = [item[1] for item in data if item[1]]
    # x,y = x[:-r,:].reshape(-1,32,18) ,y[:-r,:].reshape(-1,32,1)

    # print('normalizing data')
    # norm = tf.keras.layers.experimental.preprocessing.Normalization()
    # norm.adapt(x)

    data = np.array([x, y])
    np.save("data.npy", data)

    return x, y, # norm

def _batch_customer(id, transactions):
    '''helper for parallel batching'''

    temp = transactions[transactions["customer_id"] == id]

    timesteps = 32
    length = temp.shape[0]
    n, r = length // timesteps, length % timesteps

    if length >= timesteps:

        temp = temp.sort_values(by=["t_dat"])

        yi = temp["article_id"].to_numpy().astype(float)
        xi = temp.drop(labels=["t_dat", "article_id"], axis=1).to_numpy().astype(float)

        return (xi,yi)

    return None,None

def load():

    x, y = np.load("data.npy", allow_pickle=True)
    print(x.shape)
    norm = tf.keras.layers.experimental.preprocessing.Normalization()
    norm.adapt(x)

    return x, y, norm


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
    # process()
    batch()


if __name__ == "__main__":
    main()
