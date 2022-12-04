import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
import torch

from model.layers import FullyConnectedLayer, ConvolutionPoolLayer

class ConvNet(torch.nn.Module):
    """
    Simple convolutional neural network. 
    """
    def __init__(self, feature_map_sizes, filter_sizes, output_fc, num_classes, img_size, activation=torch.nn.LeakyReLU()):
        """
        Constructor for ConvNet.
        :param feature_map_sizes: list of out_channels for the convolution layers.
        :param filter_sizes: list of filter_size for each convolution layers.
        :param activation: Activation function used in the network. Default: ReLU.
        """   
        super().__init__()
        # Flatten layer
        self.flatten = torch.nn.Flatten() 
        # Softmax
        self.softmax = torch.nn.Softmax()
        # Convolutions
        in_channels = 1
        self.convolutions = torch.nn.ModuleList()
        for i, (out_channels, filter_size) in enumerate(zip(feature_map_sizes, filter_sizes)):
            self.convolutions.append(ConvolutionPoolLayer(in_channels, filter_size, out_channels, f'conv{i}_layer', activation))
            in_channels = out_channels

        #Fully Connected
        in_dim = int(int(img_size / 2**(len(filter_sizes)))**2 * out_channels)
        self.fcs = torch.nn.ModuleList()
        for i, output_dim in enumerate(output_fc):
            if i == len(output_fc)-1: self.fcs.append(FullyConnectedLayer(in_dim, output_dim, f'dens{i}_layer', activation=None))
            else: self.fcs.append(FullyConnectedLayer(in_dim, output_dim, f'dens{i}_layer', activation=activation))
            in_dim = output_dim

    def get_probabilities(self, x):
        """
        Returns a softmax of the forward pass.
        """
        logits = self.forward(x)
        probs = self.softmax(logits)
        return probs 

    def forward(self, x):
        """
        Compute the forward pass of the network.
        :param x: The input tensor.
        :return: The activated output of the network. 
        """
        x = x.float()
        for conv in self.convolutions:
            x = conv(x)
        x = self.flatten(x)
        for fc in self.fcs:
            x = fc(x)
        return x
    
def weights_init(m, init="Normal"):
    """
    Initialize weights by drawing from a Gaussian distribution or
    Xavier initializers. We can initialize our weights differently 
    depending on the layer type.
    """
    if isinstance(m, torch.nn.Conv2d):
        if init == "Normal":
            torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
            torch.nn.init.normal_(m.bias.data, mean=0.0, std=0.1)
        else:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.xavier_normal_(m.bias.data)
    if isinstance(m, torch.nn.Linear):
        if init == "Normal":
            torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
            torch.nn.init.normal_(m.bias.data, mean=0.0, std=0.1)
        else:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.xavier_normal_(m.bias.data)

            
def get_size_one_layer(size, padding, filter, stride):
    #h_out = (size + 2*padding[0] - (filter-1) - 1)/stride[0] + 1
    #w_out = (h + 2*padding[1] - (filter-1) - 1)/stride[1] + 1
    return (size + 2*padding - (filter-1) - 1)/stride[0] + 1

def get_size(filters, size_init):
    size = 0
    for filter in filters:
        size = get_size_one_layer(size_init, 0, filter, 1)
        size/2
    return size**2 


