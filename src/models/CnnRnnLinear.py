import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor

def print_shapes(layer_name, x):
    print_shapes = True
    if print_shapes: print(layer_name, "x.shape", x.shape)


class DilatedNet(nn.Module):
    def __init__(self, num_inputs=4, kernel_size=6,
                                     channels=[4, 4, 8],
                                     dilation=None,
                                     h1=None,
                                     h2=None,
                                     dilations=None,
                                     activation="ReLU"
                                     ):
        super(DilatedNet, self).__init__()

        self.kernel_size = len(channels) * [kernel_size]
        self.num_inputs = num_inputs
        self.channels = channels
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.2)
        self.padding = [int(self.kernel_size[i] / 2) for i in range(len(self.kernel_size))]
        
        self.conv1 = nn.Conv1d(self.num_inputs, self.channels[0], 
          padding=self.padding[0], 
          kernel_size=self.kernel_size[0])

        
        self.conv2 = nn.Conv1d(self.channels[0], self.channels[1], 
          padding=self.padding[1], 
          kernel_size=self.kernel_size[1])
        self.conv3 = nn.Conv1d(self.channels[1], self.channels[2], 
          padding=self.padding[2], 
          kernel_size=self.kernel_size[2])
        # self.conv4 = nn.Conv1d(self.channels[2], self.channels[3], 
        #   padding=self.padding[3], 
        #   kernel_size=self.kernel_size[3])

        self.lstm_input_size = 3 * channels[-1]
        
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(input_size = self.lstm_input_size, 
          hidden_size = self.lstm_hidden_size, 
          num_layers = 2,
          batch_first = True, 
          bidirectional = False)

        self.linear1_input = self.lstm_hidden_size
        self.linear1_output = 256 # self.linear1_input * 2 
        self.linear1 = nn.Linear(self.linear1_input, self.linear1_output)
        self.batch1 = nn.BatchNorm1d(self.linear1_output)

        self.linear2_input = self.linear1_output
        self.linear2_output = 32 #self.linear2_input // 2
        self.linear2 = nn.Linear(self.linear2_input, self.linear2_output)
        self.batch2 = nn.BatchNorm1d(self.linear2_output)

        self.linear3 = nn.Linear(self.linear2_output, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.max_pool(x)
        x = self.relu(x)        
        
        x = x.view(x.shape[0], 1, x.shape[1] * x.shape[2])
        x, (h, c) = self.lstm(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.batch1(x)

        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.batch2(x)
        
        out = self.linear3(x)
        
        return out