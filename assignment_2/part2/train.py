#! /usr/bin/env python

# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import sys
    sys.path.append("..")

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

import random

################################################################################

def train(config):

    # Initialize the device which to run the model on   <- copypasta?
    device = torch.device('cuda:0')                   

    # Initialize the dataset and data loader (note the +1)   <- copypasta?
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                                config.lstm_num_hidden, config.lstm_num_layers, device) 

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    loss_list = torch.empty(config.train_steps+1, device=device)
    accuracy_list = torch.empty(config.train_steps+1, device=device)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = torch.nn.functional.one_hot(batch_inputs, num_classes=dataset.vocab_size).float().to(device)
        batch_targets = batch_targets.to(device)

        # Only for time measurement of step through network
        t1 = time.time()

        out = model.forward(batch_inputs)

        loss = criterion(out.transpose(2, 1), batch_targets)
        accuracy = (out.argmax(-1) == batch_targets).float().mean()

        loss_list[step] = loss.detach()
        accuracy_list[step] = accuracy.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step % config.sample_every == 0:
            with torch.no_grad():
                pred = torch.randint(dataset.vocab_size, (1, 1)).long()
                x = torch.nn.functional.one_hot(pred, num_classes=dataset.vocab_size).float().to(device)
                keys = [pred.cpu().item()]

                for i in range(100):
                    y = model.forward(x)
                    pred = y.argmax(-1)
                    keys.append(pred.cpu()[:, -1].item())
                    y = torch.nn.functional.one_hot(pred, num_classes=dataset.vocab_size).float().to(device)
                    x = torch.cat((x, y[:, [-1], :]), 1)

                print(dataset.convert_to_string(keys))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    from matplotlib import pyplot as plt

    loss_list = loss_list.cpu()
    accuracy_list = accuracy_list.cpu()

    plt.plot(loss_list[:step], label='Loss')
    plt.plot(accuracy_list[:step], label='Accuracy')

    plt.title('Generative LSTM')
    plt.xlabel('Train step')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig('generative_lstm_loss_accuracy.svg', format='svg')
    plt.close()


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
