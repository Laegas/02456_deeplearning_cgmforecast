import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor


class DilatedNet(nn.Module):
    def __init__(self, num_inputs=4, dilations=[1,1,2,4,8],
                                     h1=2,
                                     h2=3,
                                     activation="ReLU"):
        super(DilatedNet, self).__init__()
        
        self.hidden_units = [h1,h1,h1,h2,h2]
        self.dilations = dilations

        self.num_inputs = num_inputs
        self.receptive_field = sum(dilations) + 1
        if activation == "ReLU":
          self.activation = nn.ReLU()
        elif activation == "ELU":
          self.activation = nn.ELU()
        elif activation == "CELU":
          self.activation = nn.CELU()
        else:
          raise Exception("Undefined activation function")
        
        self.conv1 = nn.Conv1d(self.num_inputs, self.hidden_units[0], kernel_size=2, dilation=self.dilations[0])
        self.conv2 = nn.Conv1d(self.hidden_units[0], self.hidden_units[1], kernel_size=2, dilation=self.dilations[1])
        self.conv3 = nn.Conv1d(self.hidden_units[1], self.hidden_units[2], kernel_size=2, dilation=self.dilations[2])
        self.conv4 = nn.Conv1d(self.hidden_units[2], self.hidden_units[3], kernel_size=2, dilation=self.dilations[3])
        self.conv_final = nn.Conv1d(self.hidden_units[4], 1, kernel_size=2, dilation=self.dilations[4])

        self.cnn = nn.Sequential(
            self.conv1,
            self.activation,
            self.conv2, 
            self.activation,
            self.conv3,
            self.activation,  
            self.conv4,
            self.activation,               
            self.conv_final
        )

    def forward(self, x):
        # First layer
        current_width = x.shape[2]
        pad = max(self.receptive_field - current_width, 0)
        input_pad = nn.functional.pad(x, [pad, 0], "constant", 0)
        x = self.cnn(input_pad)
        
        # Remove redundant dimensions
        out_final = x[:,:,-1]
      
        return out_final

