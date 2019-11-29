################################################################################
# MIT License
#
# Copyright (c) 2019
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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.device = device

        self.W_hx = nn.Parameter(nn.init.xavier_normal_(torch.empty(input_dim, num_hidden, device=device)))
        self.W_hh = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden, device=device)))
        self.W_ph = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_hidden, num_classes, device=device)))

        self.b_h = nn.Parameter(torch.zeros(1, num_hidden, device=device))
        self.b_p = nn.Parameter(torch.zeros(1, num_classes, device=device))

        self.tanh = nn.Tanh()


    def forward(self, x):
        h = torch.zeros(x.shape[0], self.num_hidden, device=self.device)

        for seq_n in range(self.seq_length):
            x_n = x[:, [seq_n]]
            h = self.tanh(x_n @ self.W_hx + h @ self.W_hh + self.b_h)
        
        p = h @ self.W_ph + self.b_p
        return p
