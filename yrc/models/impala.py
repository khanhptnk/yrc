import math

import torch
import torch.nn as nn

from yrc.utils.model import xavier_uniform_init


class Impala(nn.Module):
    """
    IMPALA convolutional neural network for feature extraction from image observations.

    Parameters
    ----------
    input_size : tuple of int
        Shape of the input observation (C, H, W).
    scale : int, optional
        Scaling factor for the number of channels. Default is 1.

    Attributes
    ----------
    block1 : ImpalaBlock
        First convolutional block.
    block2 : ImpalaBlock
        Second convolutional block.
    block3 : ImpalaBlock
        Third convolutional block.
    fc : nn.Linear
        Fully connected layer after convolutional blocks.
    output_dim : int
        Output feature dimension (default 256).

    Examples
    --------
    >>> model = Impala((3, 64, 64))
    >>> x = torch.randn(8, 3, 64, 64)
    >>> out = model(x)
    >>> print(out.shape)
    """

    def __init__(self, input_size, scale=1):
        super(Impala, self).__init__()
        self.block1 = ImpalaBlock(in_channels=input_size[0], out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)

        fc_input_size = self._get_fc_input_size(input_size)

        self.fc = nn.Linear(in_features=fc_input_size, out_features=256)
        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def _get_fc_input_size(self, input_size):
        """
        Compute the input size for the fully connected layer after convolutions.

        Parameters
        ----------
        input_size : tuple of int
            Shape of the input observation (C, H, W).

        Returns
        -------
        int
            Flattened feature size after convolutional blocks.
        """
        test_in = torch.zeros((1,) + input_size)
        test_out = self.block3(self.block2(self.block1(test_in)))
        return math.prod(test_out.shape[1:])

    def forward(self, x):
        """
        Forward pass of the IMPALA model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, C, H, W).

        Returns
        -------
        torch.Tensor
            Output feature tensor of shape (batch_size, output_dim).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        if torch.isnan(x).any():
            print("ImpalaModel output shape:", x.shape)
            print("ImpalaModel output contains NaN:", torch.isnan(x).any())
        return x


class ImpalaBlock(nn.Module):
    """
    A convolutional block used in the IMPALA architecture, consisting of a convolution, max pooling, and two residual blocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.

    Attributes
    ----------
    conv : nn.Conv2d
        Convolutional layer.
    res1 : ResidualBlock
        First residual block.
    res2 : ResidualBlock
        Second residual block.

    Examples
    --------
    >>> block = ImpalaBlock(3, 16)
    >>> x = torch.randn(8, 3, 64, 64)
    >>> out = block(x)
    >>> print(out.shape)
    """

    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        """
        Forward pass of the ImpalaBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after convolution, pooling, and residual blocks.
        """
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and ReLU activations.

    Parameters
    ----------
    in_channels : int
        Number of input and output channels.

    Attributes
    ----------
    conv1 : nn.Conv2d
        First convolutional layer.
    conv2 : nn.Conv2d
        Second convolutional layer.

    Examples
    --------
    >>> block = ResidualBlock(16)
    >>> x = torch.randn(8, 16, 32, 32)
    >>> out = block(x)
    >>> print(out.shape)
    """

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        """
        Forward pass of the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after residual connection.
        """
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class Flatten(nn.Module):
    """
    Module to flatten a tensor except for the batch dimension.

    Examples
    --------
    >>> flatten = Flatten()
    >>> x = torch.randn(8, 16, 4, 4)
    >>> out = flatten(x)
    >>> print(out.shape)
    """

    def forward(self, x):
        """
        Flatten the input tensor except for the batch dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Flattened tensor of shape (batch_size, -1).
        """
        return torch.flatten(x, start_dim=1)
