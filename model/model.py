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
    def __init__(self, feature_map_sizes, filter_sizes, num_classes, img_size, activation=torch.nn.LeakyReLU()):
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
        self.fc = FullyConnectedLayer( int(int(img_size / 2**(len(filter_sizes)))**2 * out_channels), num_classes, 'dense_layer', activation=activation)

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
        for conv in self.convolutions:
            x = conv(x)
        x = self.flatten(x)
        x = self.fc(x)
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

            
