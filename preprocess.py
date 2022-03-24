import pandas as pd
import numpy as np


def article_mapper():
    '''maps articles ids to their corresponding index'''

    data = pd.read_csv('articles.csv')
    map = [i for i in data.index]

    articles = data['article_id'].to_list()

    map = {a: m for a, m in zip(articles, map)}
    return map


def customer_mapper():
    '''like the article mapper but for customers'''

    data = pd.read_csv('customers.csv')
    map = [i for i in data.index]

    articles = data['customer_id'].to_list()

    map = {a: m for a, m in zip(articles, map)}
    return map


def main():
    customer_mapper()


if __name__ == '__main__':
    main()
