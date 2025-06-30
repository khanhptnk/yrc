import math

import torch
import torch.nn as nn

from yrc.utils.model import xavier_uniform_init


class Impala(nn.Module):
    """
    IMPALA convolutional neural network for feature extraction from image observations.

    Examples
    --------
    >>> model = Impala((3, 64, 64))
    >>> x = torch.randn(8, 3, 64, 64)
    >>> out = model(x)
    >>> print(out.shape)
    """

    def __init__(self, input_size: tuple, scale: int = 1) -> None:
        """
        Initialize the IMPALA model.

        Parameters
        ----------
        input_size : tuple of int
            Shape of the input observation (C, H, W).
        scale : int, optional
            Scaling factor for the number of channels. Default is 1.

        Returns
        -------
        None

        Examples
        --------
        >>> model = Impala((3, 64, 64))
        """
        super(Impala, self).__init__()
        self.block1 = ImpalaBlock(in_channels=input_size[0], out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)

        fc_input_size = self._get_fc_input_size(input_size)

        self.fc = nn.Linear(in_features=fc_input_size, out_features=256)
        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def _get_fc_input_size(self, input_size: tuple) -> int:
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

        Examples
        --------
        >>> model = Impala((3, 64, 64))
        >>> size = model._get_fc_input_size((3, 64, 64))
        """
        test_in = torch.zeros((1,) + input_size)
        test_out = self.block3(self.block2(self.block1(test_in)))
        return math.prod(test_out.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        Examples
        --------
        >>> model = Impala((3, 64, 64))
        >>> x = torch.randn(8, 3, 64, 64)
        >>> out = model(x)
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

    Examples
    --------
    >>> block = ImpalaBlock(3, 16)
    >>> x = torch.randn(8, 3, 64, 64)
    >>> out = block(x)
    >>> print(out.shape)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the ImpalaBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns
        -------
        None

        Examples
        --------
        >>> block = ImpalaBlock(3, 16)
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        Examples
        --------
        >>> block = ImpalaBlock(3, 16)
        >>> x = torch.randn(8, 3, 64, 64)
        >>> out = block(x)
        """
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and ReLU activations.

    Examples
    --------
    >>> block = ResidualBlock(16)
    >>> x = torch.randn(8, 16, 32, 32)
    >>> out = block(x)
    >>> print(out.shape)
    """

    def __init__(self, in_channels: int) -> None:
        """
        Initialize the ResidualBlock.

        Parameters
        ----------
        in_channels : int
            Number of input and output channels.

        Returns
        -------
        None

        Examples
        --------
        >>> block = ResidualBlock(16)
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        Examples
        --------
        >>> block = ResidualBlock(16)
        >>> x = torch.randn(8, 16, 32, 32)
        >>> out = block(x)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        Examples
        --------
        >>> flatten = Flatten()
        >>> x = torch.randn(8, 16, 4, 4)
        >>> out = flatten(x)
        """
        return torch.flatten(x, start_dim=1)
