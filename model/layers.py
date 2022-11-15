import torch
import math


class FullyConnectedLayer(torch.nn.Module):
    """
    Simple fully connected layer with optional activation.
    """
    def __init__(self, input_dim, output_dim, layer_name, activation=None, bias=True):            
        """
        Constructor for our FullyConnected layer.
        :param input_dim: The input dimension of the layer.
        :param output_dim: The desired output size we want to map to.
        :param layer_name: A name for this layer.
        :param activation: Activation function used on the output of the dense layer.
        :param bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        super().__init__()
        self.hidden = 32
        self.fc1 = torch.nn.Linear(input_dim, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, 10)
        self.fc3 = torch.nn.Linear(10, output_dim)
        # self.fc = torch.nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.name = layer_name
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        """
        Compute the forward pass of the FullyConnected layer.
        :param x: The input tensor.
        :return: The output of this layer. 
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        if self.activation:
            x = self.activation(x)
        return x

class ConvolutionPoolLayer(torch.nn.Module):
    """
        Convolutional layer with optional MaxPooling and optional activation.
    """
    def __init__(self, in_channels, filter_size, out_channels, layer_name, activation, bias=True, use_pooling=True):
        """
        Constructor for ConvolutionPool layer.
        :param in_channels: Number of channels of the input.
        :param filter_size: Width and height of the square filter (scalar).
        :param out_channels: How many feature maps to produce with this layer.
        :param layer_name: A name for this layer.
        :param activation: Activation function used on the output of the layer.
        :param bias: If set to False, the layer will not learn an additive bias. Default: True.
        :param use_pooling: Use 2x2 max-pooling if True. Default: True.
        """   
        super().__init__()
        # Convolution parameters
        self.stride = (1, 1)
        self.filter_size = filter_size
        # Convolution operation - we do the padding manually in order to get tensorflow 'same' padding
        self.conv = torch.nn.Conv2d(in_channels, out_channels, self.filter_size, stride=self.stride, padding=0, bias=bias)
        # Use pooling to down-sample the image resolution?
        # This is 2x2 max-pooling, which means that we consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        self.max_pool = torch.nn.MaxPool2d(2, 2) if use_pooling else None
        # This adds some non-linearity to the formula and allows us to learn more complicated functions.
        self.activation = activation

        self.dropout = torch.nn.Dropout(0.25)
        
    def get_padding_amount(self, shape):
        """
        Computes the amount of padding so that the input size is equal to output size.
        PyTorch doesn't provide 'same' padding so we use the implementation from TensorFlow.
        """
        _, _, input_h, input_w = shape
        output_h = int(math.ceil(float(input_h) / float(self.stride[0])))
        output_w = int(math.ceil(float(input_w) / float(self.stride[1])))
         
        if input_h % self.stride[0] == 0:
            pad_along_height = max((self.filter_size - self.stride[0]), 0)
        else:
            pad_along_height = max(self.filter_size - (input_h % self.stride[0]), 0)
        if input_w % self.stride[1] == 0:
            pad_along_width = max((self.filter_size - self.stride[1]), 0)
        else:
            pad_along_width = max(self.filter_size - (input_w % self.stride[1]), 0)
            
        pad_top = pad_along_height // 2 # amount of zero padding on the top
        pad_bottom = pad_along_height - pad_top     # amount of zero padding on the bottom
        pad_left = pad_along_width // 2     # amount of zero padding on the left
        pad_right = pad_along_width - pad_left  # amount of zero padding on the right

        return pad_left, pad_right, pad_top, pad_bottom

    def forward(self, x):
        """
        Compute the forward pass of the ConvolutionPoolLayer layer.
        :param x: The input tensor.
        :return: The output of this layer. 
        """
        padding = self.get_padding_amount(x.shape)
        x = torch.nn.functional.pad(x, padding)  # [left, right, top, bot]
        x = self.conv(x)
        x = self.dropout(x)
        if self.max_pool:
            x = self.max_pool(x)
        if self.activation:
            x = self.activation(x)
        return x