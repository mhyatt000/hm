import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualize import scatter


def main():

    path_articles = 'articles.csv'

    articles = pd.read_csv(path_articles)

    # remove words for now?
    articles = articles.drop(labels='detail_desc', axis=1)

    print(articles)
    print(articles.columns)

    data = {'product_type_no': articles['product_type_no'].tolist(),
            'garment_group_no': articles['garment_group_no'].tolist()}

    print(data_dict)

    fig, ax = plt.subplots()
    ax.scatter(articles['product_type_no'].tolist(), articles['garment_group_no'].tolist())
    plt.show()

    scatter(data)


if __name__ == '__main__':
    main()
