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
import time
from argparse import ArgumentParser

def get_args():

    ap = ArgumentParser()

    ap.add_argument('-i','--input',type=str)

    return ap.parse_args()

class CustomDataset(torch.utils.data.Dataset):

    '''split csvs with natural language d1 .. d15 csv'''

    def __init__(self, timesteps=16, build=False):

        self.timesteps = timesteps

        if build:

            args = get_args()
            print('reading dataset...')
            input = args.input if args.input else [print('need input -i'),quit()]
            data = pd.read_csv(input) # data[data.columns[1:]].to_csv(input, index=False)

            print('split data... ETA 70 hrs')
            col = data.columns
            bar = tqdm(total=data.shape[0])
            for _,row in data.iterrows():
                id = row['customer_id']

                '''TODO
                if the next one has the same id dont close file yet'''

                try:
                    temp = pd.read_csv(f'dataset/cust_{id}.csv')
                except:
                    temp = pd.DataFrame({c:[] for c in col})
                finally:
                    temp = temp.append(row)
                    temp.to_csv(f'dataset/cust_{id}.csv', index=False)
                    bar.update(1)

        # print('reading dataset...')
        # data = pd.read_csv('docs/data.csv')
        # # tensor = torch.Tensor(data.values)
        # print(f'Finished in{round(time.perf_counter() - timer)} secs')
        #
        # print('building customer list... ETA 20 min')
        # customers = set(data["customer_id"].tolist())
        # col = data.columns
        # self.transactions = {cust : pd.DataFrame({c:[] for c in col}) for cust in tqdm(customers)}
        #
        # with ProcessPoolExecutor() as executor:

        #     print('building transaction batches...')
        #     tqdm(executor.map(self._batch_customer, customers), total=len(customers))
        #
        # print('normalizing torch tensors...')
        # y, x = np.array([]), np.array([])
        # for id in customers:
        #     xi,yi = transactions[id]
        #
        #     if xi or yi: # transactions for index 0? idk
        #         x = np.append(x,xi)
        #         y = np.append(y,yi)
        #
        # norm = lambda x: torch.nn.functional.normalize(x)
        # self.x = norm(torch.tensor(x.reshape(-1,32,18)))
        # self.y = norm(torch.tensor(y.reshape(-1,32,1)))


    # def _batch_customer(self, id):
    #     '''helper for parallel batching'''
    #
    #     temp = self.transactions[self.transactions["customer_id"] == id]
    #
    #     timesteps = 32
    #     length = temp.shape[0]
    #     n, r = length // timesteps, length % timesteps
    #
    #     if length >= timesteps:
    #
    #         temp = temp.sort_values(by=["t_dat"])
    #
    #         yi = temp["article_id"].to_numpy().astype(float)
    #         xi = temp.drop(labels=["t_dat", "article_id"], axis=1).to_numpy().astype(float)
    #
    #         self.transactions[id] = (xi,yi)
    #
    #     self.transactions[id] = (None,None)


    def __len__(self):
        customers = pd.read_csv('docs/customers.csv')
        return customers.shape[0]

    def __getitem__(self, i):

        zeros = [
            torch.zeros(size=(1,self.timesteps,18)),
            torch.zeros(size=(1,self.timesteps,1))
        ]

        try:
            data = pd.read_csv(f'dataset/cust_{i}.csv')
        except:
            return zeros[0][0], zeros[1][0]


        length = data.shape[0]
        n, r = length // self.timesteps, length % self.timesteps



        if length >= self.timesteps:

            try:
                data = data.sort_values(by=["t_dat"])

                y = data["article_id"].to_numpy().astype(float)
                x = data.drop(labels=["t_dat", "article_id"], axis=1).to_numpy().astype(float)

                # take 1st 32 steps only rn
                x = x.flatten()[:32*18]
                y = y.flatten()[:32]

                if x.shape[0] < 576:
                    return zeros[0][0], zeros[1][0]
                    'todo count how many zeros there are and drop them?'


                norm = lambda x: torch.nn.functional.normalize(x)
                x = norm(torch.tensor(x.reshape(-1,self.timesteps,18)))
                y = norm(torch.tensor(y.reshape(-1,self.timesteps,1)))

                '''ONLY RETURN FIRST BATCH FOR CUSTOMER RN'''
            except ValueError as ve:
                print(ve)
                print(f'error in file: dataset/cust_{i}.csv')
                return zeros[0][0], zeros[1][0]


        return zeros[0][0], zeros[1][0]


class CustomDataLoader(DataLoader):

    def __init__(self, batch_size=64):
        super().__init__(CustomDataset(timesteps=32, build=False),
            batch_size=batch_size, shuffle=True)

def try_again(func, **kwargs):
    try:
        func()
    except:
        try_again(func)

def main():

    dataset = CustomDataset(timesteps=32, build=False)

    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    dataloader = CustomDataLoader()

    for i in range(10):
        print(f'trial {i}')
        train_features, train_labels = next(iter(dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        time.sleep(0.25)

    # print(len(data))
    #
    # print(data.x.shape)


if __name__ == '__main__':
    main()
