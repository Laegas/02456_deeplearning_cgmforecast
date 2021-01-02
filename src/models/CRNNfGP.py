import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor

def print_shapes(layer_name, x):
    print_shapes = True
    if print_shapes: print(layer_name, "x.shape", x.shape)


class DilatedNet(nn.Module):
    def __init__(self, num_inputs=4, kernel_size=8,
                                     channels=[8,16,32]):
        
        super(DilatedNet, self).__init__()

        self.kernel_size = 3 * [kernel_size]
        self.num_inputs = num_inputs
        self.channels = channels
        # print("PARAMS\nnum_inputs", num_inputs, "\nkernel_size", self.kernel_size, "\nchannels", channels)
        
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(2)

        # self.lli = len(self.kernel_size) - 1 # last layer index
        self.padding = [int(self.kernel_size[i] / 2) for i in range(len(self.kernel_size))]
        
        self.conv1 = nn.Conv1d(self.num_inputs, self.channels[0], padding=self.padding[0], kernel_size=self.kernel_size[0])
        self.conv2 = nn.Conv1d(self.channels[0], self.channels[1], padding=self.padding[1], kernel_size=self.kernel_size[1])
        self.conv3 = nn.Conv1d(self.channels[1], self.channels[2], padding=self.padding[2], kernel_size=self.kernel_size[2])
        
        self.lstm = nn.LSTM(input_size = 96, hidden_size = 64, batch_first = True, bidirectional = True)

        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 32)
        self.linear3 = nn.Linear(32, 1)

        # arrays for layers - convolutions, batch normalization
        # self.conv = [] 
        # self.batch_norm = []
        
        # first conv layer
        # self.conv.append(nn.Conv1d(self.num_inputs, self.hidden_units[0], padding=self.padding[0], kernel_size=self.kernel_size[0], dilation=self.dilations[0]))
        # self.batch_norm.append(nn.BatchNorm1d(self.hidden_units[0]))

        # conv layers in between
        # for i in range (1, self.lli):
          # self.conv.append(nn.Conv1d(self.hidden_units[i-1], self.hidden_units[i], padding=self.padding[i], kernel_size=self.kernel_size[i], dilation=self.dilations[i]))
          # self.batch_norm.append(nn.BatchNorm1d(self.hidden_units[i]))
        
        # self.conv2 = nn.Conv1d(self.hidden_units[0], self.hidden_units[1], padding=self.padding[1], kernel_size=self.kernel_size[1], dilation=self.dilations[1])
        # self.conv3 = nn.Conv1d(self.hidden_units[1], self.hidden_units[2], padding=self.padding[2], kernel_size=self.kernel_size[2], dilation=self.dilations[2])
        # self.conv4 = nn.Conv1d(self.hidden_units[2], self.hidden_units[3], padding=self.padding[3], kernel_size=self.kernel_size[3], dilation=self.dilations[3])
        
        # last conv layer
        # self.conv.append(nn.Conv1d(self.hidden_units[self.lli - 1], 1, padding=self.padding[self.lli], kernel_size=self.kernel_size[self.lli], dilation=self.dilations[self.lli]))

        # convert arrays to ModuleList, necessary
        # self.conv = nn.ModuleList(self.conv)
        # self.batch_norm = nn.ModuleList(self.batch_norm)

    def forward(self, x):
        #print_shapes("input", x)
        x = self.relu(self.max_pool(self.conv1(x)))
        #print_shapes("conv1", x)
        x = self.relu(self.max_pool(self.conv2(x)))
        #print_shapes("conv2", x)
        x = self.relu(self.max_pool(self.conv3(x)))
        #print_shapes("conv3", x)
        x = x.view(x.shape[0], 1, x.shape[1] * x.shape[2])
        #print_shapes("view", x)
        x, (h, c) = self.lstm(x)
        #print_shapes("lstm", x)
        x = x.reshape(x.shape[0], -1)
        #print_shapes("reshape", x)
        x = self.relu(self.linear1(x))
        #print_shapes("linear1", x)
        x = self.relu(self.linear2(x))
        #print_shapes("linear2", x)
        x = self.linear3(x)
        #print_shapes("linear3", x)
        
        #input("Wait for input")
        
        return x
