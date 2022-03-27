import pandas as pd
import numpy as np
import tensorflow as tf


def process():

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

    '''TODO replace null values with 0?? (theres a lot) ... print data'''
    customers['age'] = customers['age'].fillna(0)
    customers['postal_code'] = customers['postal_code'].fillna(0)

    customer_columns = [
        # 'FN',
        # 'Active',
        # 'club_member_status',
        # 'fashion_news_frequency',
        'age',
        'postal_code'
    ]

    customer_columns = [(c, customers[c].tolist()) for c in customer_columns]

    customer_map = customer_hash_map()

    customer_info = {
        customer_map[x]: {c[0]: c[1][i] for c in customer_columns}
        for i, x in enumerate(customer_ids)
    }

    # transactions.csv
    path_transactions = "head_transactions_train.csv"
    transactions = pd.read_csv(path_transactions)

    t_dat = transactions["t_dat"].tolist()
    t_dat = [item.split("-") for item in t_dat]
    transactions["year"] = [item[0] for item in t_dat]
    transactions["month"] = [item[1] for item in t_dat]
    transactions["day"] = [item[2] for item in t_dat]
    transactions = transactions.drop(labels="t_dat", axis=1)

    # transfer article info to transactions
    transactions["article_id"] = transactions["article_id"].apply(
        lambda x: article_map[x]
    )

    for c in article_columns:
        transactions[c[0]] = transactions["article_id"].apply(
            lambda x: article_info[x][c[0]]
        )

    # transfer customer info to transactions
    transactions["customer_id"] = transactions["customer_id"].apply(
        lambda x: customer_map[x]
    )

    for c in customer_columns:
        transactions[c[0]] = transactions["customer_id"].apply(
            lambda x: customer_info[x][c[0]]
        )

    zip_code_map = zip_hash_map()
    transactions["postal_code"] = transactions["postal_code"].apply(lambda x: zip_code_map[x])

    # prepare for export
    y = transactions["article_id"].to_numpy().astype(float)
    x = transactions.drop(labels="article_id", axis=1).to_numpy().astype(float)

    norm = tf.keras.layers.experimental.preprocessing.Normalization()
    norm.adapt(x)

    return x, y, norm


def article_hash_map():
    """maps articles ids to their corresponding index"""

    data = pd.read_csv("articles.csv")
    map = [i for i in data.index]

    articles = data["article_id"].to_list()

    map = {a: m for a, m in zip(articles, map)}
    return map


def customer_hash_map():
    """like the article mapper but for customers"""

    data = pd.read_csv("customers.csv")
    map = [i for i in data.index]

    articles = data["customer_id"].to_list()

    map = {a: m for a, m in zip(articles, map)}
    return map


def zip_hash_map():
    """like the article mapper but for customers"""

    data = pd.read_csv("customers.csv")

    data['postal_code'] = data['postal_code'].fillna(0)
    zip_code = data["postal_code"].to_list()
    map = [i for i in range(len(zip_code))]

    map = {a: m for a, m in zip(zip_code, map)}
    return map


def main():
    process()


if __name__ == "__main__":
    main()
