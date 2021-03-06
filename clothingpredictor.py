from argparse import ArgumentParser
import functools
from functools import reduce
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DataParallel as Parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import torchvision
from tqdm import tqdm

from batch import CustomDataLoader


def get_args():

    ap = ArgumentParser()

    ap.add_argument('-s', '--save', action='store_true')
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('-t', '--train', action='store_true')
    ap.add_argument('-e', '--eval', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('-n', '--num-epochs', type=int)
    ap.add_argument('-b', '--batch-size', type=int)
    ap.add_argument('--steps', type=int)

    args = ap.parse_args()

    if not args.num_epochs:
        args.num_epochs = 5
        print('default 5 epochs')
    if not args.steps:
        args.steps = 8
        print('default to 8 steps')
    if not args.batch_size:
        args.batch_size = 64
        print('default to batch_size 128')

    return args


class ClothingPredictor(torch.nn.Module):

    def __init__(self, dropout=0, **kwargs):
        super(ClothingPredictor, self).__init__()

        self.lstm = nn.LSTM(input_size=18,hidden_size=256,num_layers=3, dropout=dropout)
        self.dense = nn.Linear(256 , 105542)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, X, state=None, *args):
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2).float()
        # When state is not mentioned, it defaults to zeros
        output, state = self.lstm(X)

        output = self.activation(self.dense(output))
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        # state[-1] is the only useful one
        return output, state


def train(net, train_iter):

    timer = time.perf_counter()

    num_epochs = net.args.num_epochs
    optimizer = net.optimizer

    device = net.device
    net.to(device)
    net.train()

    len_iter = len(train_iter)
    ce_loss = nn.CrossEntropyLoss()
    ctc_loss = nn.CTCLoss()
    adalog_loss = nn.AdaptiveLogSoftmaxWithLoss(n_classes=105542, in_features=18, cutoffs = [10, 100, 1_000, 10_000])
    losses = []

    pbar = tqdm(total=num_epochs*len_iter, desc='Training ')
    for epoch in range(num_epochs):

        for i, batch in enumerate(train_iter):

            # X, Y = torch.randn(32,64,18), torch.tensor([[random.randint(0,105542) for step in range(32)] for iter in range(64)]).float()
            # while True:

                optimizer.zero_grad()

                X, Y = [x.to(device) for x in batch]
                # X,Y = X.to(device), Y.to(device)

                Y_hat, _ = net(X)

                # reshape Y to match predictions
                Y = Y.to(torch.int64)
                Y = F.one_hot(torch.flatten(Y.permute(1,0,2)), num_classes=105542)
                Y = torch.reshape(Y, (Y_hat.shape)).to(torch.float32)

                # backprop
                l = ce_loss(Y_hat, Y)
                l.sum().backward()

                nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()

                if not i%100: # display

                    correct = float(torch.sum(torch.argmax(Y,dim=-1) == torch.argmax(Y_hat,dim=-1)))
                    acc = correct / (Y.shape[0]*Y.shape[1])

                    pbar.set_postfix({
                        'batch' : f'{i+1}/{len_iter}',
                        'loss' : f'{round(float(l.sum()),4)}',
                        'epoch' : f'{epoch+1}/{num_epochs}',
                        'accuracy' : acc,
                        'correct' : correct,
                    })
                pbar.update(1)


    # save it ... after all of training
    if net.args.save:
        torch.save(net.state_dict(), net.file)

    # print(f'loss: {l.sum()}')

    timer = int(time.perf_counter() - timer)
    print(f'Finished in {timer} seconds')
    print(f'loss: {round(losses[-1],4)}')

    return losses


def main():

    print()
    args = get_args()

    net = ClothingPredictor()
    if torch.cuda.device_count() > 1:
        net = Parallel(net)

    print(net)

    # hparam
    net.file = __file__.split('/')[-1].split('.')[0] + '.pt'
    net.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.args = args
    net.optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-3)

    # try to load data
    train_iter = CustomDataLoader(batch_size=net.args.batch_size, timesteps=net.args.steps)

    # try to load weights
    if args.load:
        try:
            net.load_state_dict(torch.load(net.file))
        except:
            pass # maybe set a default?? xavier_init ?

    # train
    if args.train:
        train(net, train_iter)

    # do some predictions
    if args.eval:
        print('evaluation...')

        print('no eval yet')
        quit()

        train_iter = torch.rand(size=(64,32,18))
        cust_iter = torch.rand(size=(1,32,18))

        print(cust_iter.shape)

        x, state = net(cust_iter)
        print(x.shape, state[-1].shape)
        print(x[-1].shape)

        print('predicted items:',[i.argmax() for i in x[-1]])

        net.eval()
        # load in the data for a customer somehow
        print('\n\n\n')
        print(f'customer _____')
        for i in range(12):
            print(f'iter: {i+1}')

            pred, state = net(cust_iter)
            pred_item = pred[-1]
            print(f'predicted article: {int(pred_item.argmax())}')
            quit()


if __name__ == '__main__':
    main()
