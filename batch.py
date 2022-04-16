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
        self.z = 0 # number of times returning zeros

        customers = pd.read_csv('docs/customers.csv')
        self.length = customers.shape[0]
        print(type(self.length))


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

    def __len__(self):
        return self.length


    def return_zeros(self, i):

        self.z += 1
        print(f'{self.z} : dataset/cust_{i}.csv')
        zeros = [
            torch.zeros(size=(1,self.timesteps,18)),
            torch.zeros(size=(1,self.timesteps,1))
        ]
        return zeros[0][0], zeros[1][0]


    def __getitem__(self, i):

        file = f'dataset/cust_{i}.csv'

        try:
            data = pd.read_csv(file)
        except:
            self.z += 1
            print(f'{self.z} : file doesnt exist', file)
            return self.__getitem__(i+1)

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
                    self.z += 1
                    print(f'{self.z} : less than 576', file)
                    return self.__getitem__(i+1)
                    'todo count how many zeros there are and drop them?'


                norm = lambda x: torch.nn.functional.normalize(x)
                x = norm(torch.tensor(x.reshape(-1,self.timesteps,18)))
                y = norm(torch.tensor(y.reshape(-1,self.timesteps,1)))

                '''ONLY RETURN FIRST BATCH FOR CUSTOMER RN'''
            except ValueError as ve:
                print(ve)
                print(f'error in file: dataset/cust_{i}.csv')
                return self.__getitem__(i+1)

        self.z += 1
        print(f'{self.z} : too small', file)
        return self.__getitem__(i+1)


class CustomDataLoader(DataLoader):

    def __init__(self, batch_size=64, timesteps=16):
        super().__init__(CustomDataset(timesteps=timesteps, build=False),
            batch_size=batch_size, shuffle=True)

        print('batch_size', batch_size)
        print('timesteps', timesteps)

def try_again(func, **kwargs):
    try:
        func()
    except:
        try_again(func)

def main():

    dataset = CustomDataset(build=False)

    print(len(dataset))
    for i in range(len(dataset)):
        print(f'item {i}')
        print(dataset[i])

    quit()

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
