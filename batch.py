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


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self):
        print('reading dataset...')
        data = pd.read_csv('docs/data.csv')

        print('building customer list... ETA 20 min')
        customers = set(data["customer_id"].tolist())
        col = data.columns
        self.transactions = {cust : pd.DataFrame({c:[] for c in col}) for cust in tqdm(customers)}

        with ProcessPoolExecutor() as executor:

            # for every row in df, append it to its own df O(n) vs O(n2) time
            print('splitting transactions by customer...')
            rows = [item[1] for item in list(data.iterrows())]
            def add_transaction(t):
                id = t["customer_id"]
                self.transactions[id] = self.transactions[id].append(t)
            tqdm(executor.map(add_transaction, rows), total=len(customers))

            print('building transaction batches...')
            tqdm(executor.map(self._batch_customer, customers), total=len(customers))

        print('normalizing torch tensors...')
        y, x = np.array([]), np.array([])
        for id in customers:
            xi,yi = transactions[id]

            if xi or yi: # transactions for index 0? idk
                x = np.append(x,xi)
                y = np.append(y,yi)

        norm = lambda x: torch.nn.functional.normalize(x)
        self.x = norm(torch.tensor(x.reshape(-1,32,18)))
        self.y = norm(torch.tensor(y.reshape(-1,32,1)))


    def _batch_customer(self, id):
        '''helper for parallel batching'''

        temp = self.transactions[self.transactions["customer_id"] == id]

        timesteps = 32
        length = temp.shape[0]
        n, r = length // timesteps, length % timesteps

        if length >= timesteps:

            temp = temp.sort_values(by=["t_dat"])

            yi = temp["article_id"].to_numpy().astype(float)
            xi = temp.drop(labels=["t_dat", "article_id"], axis=1).to_numpy().astype(float)

            self.transactions[id] = (xi,yi)

        self.transactions[id] = (None,None)


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def main():

    data = CustomDataset()
    print(len(data))

    print(data.x.shape)


if __name__ == '__main__':
    main()
