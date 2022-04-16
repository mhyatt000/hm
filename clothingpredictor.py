import functools
import os
import time
from argparse import ArgumentParser
from functools import reduce
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

from batch import CustomDataLoader

def get_args():

    ap = ArgumentParser()

    ap.add_argument('-s', '--save', action='store_true')
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('-t', '--train', action='store_true')
    ap.add_argument('-e', '--eval', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('-n', '--num_epochs', type=int)

    args = ap.parse_args()

    print(args.load)

    if not args.num_epochs:
        args.num_epochs = 5
        print('default 5 epochs')

    return args


class ClothingPredictor(torch.nn.Module):

    def __init__(self, dropout=0, **kwargs):
        super(ClothingPredictor, self).__init__()

        self.lstm = nn.LSTM(input_size=18,hidden_size=256,num_layers=3, dropout=dropout)
        self.dense = nn.Linear(256 , 105542)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, X, state=None, *args):
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.lstm(X)

        output = self.activation(self.dense(output))
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        # state[-1] is the only useful one
        return output, state

class CustomCELoss(nn.CrossEntropyLoss):

    def __init__(self):
        super().__init__()
        pass

    def __call__(self, src,tgt):

        src = torch.reshape(torch.argmax(src.permute(1,0,2), dim=-1), (tgt.shape))
        score = torch.sum(torch.eq(src,tgt))
        num = reduce(lambda x,y: x*y, tgt.shape, 1)
        return score / num


def train(net, train_iter):

    timer = time.perf_counter()

    num_epochs = net.args.num_epochs
    optimizer = net.optimizer

    device = net.device
    net.to(device)
    net.train()

    loss = nn.CrossEntropyLoss()
    loss = CustomCELoss()
    losses = []
    pbar = tqdm(range(num_epochs), desc='Training ')
    for epoch in range(num_epochs):

        for batch in train_iter:

            optimizer.zero_grad()

            X, Y = [x.to(device) for x in batch]

            Y_hat, _ = net(X)
            # Y_hat = torch.reshape(torch.argmax(Y_hat.permute(1,0,2), dim=-1), (Y.shape))
            l = loss(Y_hat, Y)
            l.sum().backward()

            nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            losses.append(l.sum())
        pbar.update(1)
        pbar.set_postfix_str(f'loss : {round(float(l.sum()),4)}')


    # save it ... after all of training
    if net.args.save:
        torch.save(net.state_dict(), net.file)

    # print(f'loss: {l.sum()}')

    timer = int(time.perf_counter() - timer)
    print(f'Finished in {timer} seconds')
    print(f'loss: {round(losses[-1],4)}')

    return losses


def main():

    args = get_args()

    net = ClothingPredictor()
    if torch.cuda.device_count() > 1:
        net = Parallel(net)

    print(net)

    net.file = __file__.split('/')[-1].split('.')[0] + '.pt'
    net.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.args = args
    net.optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-3)

    train_iter = CustomDataLoader()

    if args.load:
        try:
            net.load_state_dict(torch.load(net.file))
        except:
            pass



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
