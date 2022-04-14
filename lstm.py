import functools
import os
import time
from argparse import ArgumentParser

import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as Parallel
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def get_args():

    ap = ArgumentParser(description='args for hw4')

    ap.add_argument('-s', '--save', action='store_true')
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('-t', '--train', action='store_true')
    ap.add_argument('-e', '--eval', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('-n', '--num_epochs', type=int)

    args = ap.parse_args()

    if not args.num_epochs:
        args.num_epochs = 5
        print('default 5 epochs')

    return args


class Environment():
    'environment variables to be passed around'

    def __init__(self):
        pass


class RNNModel(torch.nn.Module):

    def __init__(self):
        super(RNNModel, self).__init__()

        self.rnn = torch.nn.LSTM(input_size=18,hidden_size=256,num_layers=3)

        self.hidden = torch.nn.Linear(256 , 512)
        self.out = torch.nn.Linear(512 , 105542)

    def forward(self, x):
        print('test')
        out, hidden_ = self.rnn(input)
        print('test')
        out = self.out(self.linear(out))
        return out


def train(net, train_iter, optimizer, *, env):

    timer = time.perf_counter()

    num_epochs = env.args.num_epochs

    device = env.device
    net.to(device)

    loss = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(num_epochs):
        print(f'\nepoch: {epoch+1} of {num_epochs}')

        net.train()
        for (features, target) in tqdm(train_iter):

            optimizer.zero_grad()
            X, Y = features.to(device), target.to(device)

            fx = net(X)
            l = loss(fx, Y)
            l.mean().backward()
            optimizer.step()

            losses.append(l.mean())

            # save it ... after batch cuz it takes a while
            if env.args.save and l.mean() < min(losses):
                torch.save(net.state_dict(), env.file)

        print(f'loss: {l.mean()}')

    timer = int(time.perf_counter() - timer)
    print(f'Finished in {timer} seconds')
    print(f'{len(train_iter.dataset) / timer:.1f} examples/sec on {str(device)}')


def main():

        args = get_args()

        env = Environment()
        env.file = __file__.split('/')[-1].split('.')[0] + '.pt'
        env.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env.args = args

        net = RNNModel()
        print(net)

        train_iter = DataLoader(training_data, batch_size=64, shuffle=True)

        if args.load:
            try:
                net.load_state_dict(torch.load(env.file))
            except:
                pass

        if torch.cuda.device_count() > 1:
            net = Parallel(net)

        train_iter = torch.rand(size=(64,2,32,18))

        # train
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-3)

        if args.train:
            train(net, train_iter, optimizer, env=env)

        # do some predictions
        if args.eval:
            print('evaluation...')

            n, imgs = 4, []
            for i in tqdm(range(n)):
                pass

            print('predictions here...')


if __name__ == '__main__':
    main()
