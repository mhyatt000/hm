import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat

import torch
from torch.nn import functional as F

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

        customers = pd.read_csv('docs/customers.csv')
        self.length = customers.shape[0]


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


    def return_zeros(self):

        zeros = [
            torch.zeros(size=(1,self.timesteps,18)),
            torch.zeros(size=(1,self.timesteps,1))
        ]
        return zeros[0][0], zeros[1][0]


    def __getitem__(self, i):

        # i = i%8

        'TODO ... make a list of x that are too short and put in a file (8,16,32)'
        while True:
            file = f'dataset/cust_{i}.csv'

            try:
                data = pd.read_csv(file)
            except:
                # print(f'file doesnt exist', file)

                i += 1
                continue
                # return self.return_zeros()

            length = data.shape[0]
            n, r = length // self.timesteps, length % self.timesteps

            if length >= self.timesteps:

                try:
                    return self.pd_to_tensor(data)

                except Exception as ex:
                    print(ex)
                    # print(f'error in file: dataset/cust_{i}.csv')
                    # return self.__getitem__(i+1)
                    # return self.return_zeros()
                    i += 1
                    continue

            # print(f'too small', file)
            # return self.__getitem__(i+1)
            # return self.return_zeros()
            i += 1
            continue


    def pd_to_tensor(self, data):

        data = data.sort_values(by=["t_dat"])
        y = data["article_id"].to_numpy().astype(float)
        x = data.drop(labels=["t_dat", "article_id"], axis=1).to_numpy().astype(float)

        # take 1st 32 steps only rn
        # seed the first transaction ...
        # given 0, predict item
        seed = np.zeros(18)
        x = np.append(seed,x.flatten())[:32*18]
        y = y.flatten()[:32]

        norm = lambda x: torch.nn.functional.normalize(x)
        x = norm(torch.tensor(x.reshape(-1,self.timesteps,18)))
        y = norm(torch.tensor(y.reshape(-1,self.timesteps,1)))

        # raise Exception('err')

        return x[0], y[0]


    def count_valid_files(self):

        self.pbar = tqdm(total=len(self))
        def is_valid(i):
            file = f'dataset/cust_{i}.csv'
            self.pbar.update(1)
            try:
                data = pd.read_csv(file)
                length = data.shape[0]
                if length >= self.timesteps:
                    return 1
                else:
                    return 0
            except:
                return None


        with ThreadPoolExecutor() as executor:
            files = list(tqdm(executor.map(is_valid, range(len(self)))))
            val = sum([i for i in files if i])
            non = sum([1 for i in files if i is None])
            length = len(files)

        print(f'length == len : {length==len(self)}')
        print(f'{val/len(self)} % valid')
        print()
        print('val, inv, none')
        print(val, length-val, non)


class CustomDataLoader(DataLoader):

    def __init__(self, batch_size=64, timesteps=16, shuffle=True):
        super().__init__(CustomDataset(timesteps=timesteps, build=False),
            batch_size=batch_size, shuffle=shuffle)

        print('batch_size', batch_size)
        print('timesteps', timesteps)


def main():

    ''' 8 timesteps
    0.5324625723407047 % valid
    val, inv, none
    730528 641452 37878
    '''

    dataset = CustomDataset(build=False, timesteps=8)
    # dataset.count_valid_files()
    # quit()

    print(len(dataset))
    for i in range(8):
        print(f'\nitem {i}')
        item = dataset[i]
        print(item[0].shape, item[1].shape)
        print(item[0].dtype, item[1].dtype)

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
