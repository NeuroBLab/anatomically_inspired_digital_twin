from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.init as init


class MLPBehavior(nn.Module):
    """
    An MLP that processes time-series behavioral data for network modulation.

    The module can optionally apply a causal 1D convolution over the time
    dimension to incorporate temporal context. The output is expanded with two
    singleton spatial dimensions to facilitate broadcasting with 4D or 5D
    feature maps from a convolutional core.
    """

    _ACTIVATIONS = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
    }

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = None,
        hidden_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        activation: str = "tanh",
        gamma: float = 0.0,
        temporal_conv: bool = False,
        temporal_kernel: int = 5,
    ):
        """
        Initializes the MLPBehavior modulator.

        Args:
            input_channels (int): The number of input behavioral features.
            output_channels (int): The number of output features.
            hidden_channels (int, optional): The number of units in hidden
                layers. Defaults to 4 * output_channels.
            hidden_layers (int, optional): The number of hidden layers.
                Defaults to 1.
            bias (bool, optional): If True, adds a bias term to linear layers.
                Defaults to True.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            activation (str, optional): The activation function to use.
                Must be one of 'tanh', 'relu', 'sigmoid'. Defaults to 'tanh'.
            gamma (float, optional): L1 regularization strength. Defaults to 0.0.
            temporal_conv (bool, optional): If True, applies a temporal
                convolution after the MLP. Defaults to False.
            temporal_kernel (int, optional): The kernel size for the temporal
                convolution. Defaults to 5.
        """
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Activation '{activation}' not supported. "
                f"Choose from {list(self._ACTIVATIONS.keys())}"
            )
        if hidden_channels is None:
            hidden_channels = 4 * output_channels

        self.gamma = gamma
        activation_fn = self._ACTIVATIONS[activation]

        layers = []
        in_features = input_channels
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_features, hidden_channels, bias=bias),
                    activation_fn,
                    nn.Dropout(p=dropout),
                ]
            )
            in_features = hidden_channels
        layers.extend([nn.Linear(in_features, output_channels, bias=bias), activation_fn])
        self.mlp = nn.Sequential(*layers)

        if temporal_conv:
            # Use padding for a causal convolution
            self.temporal_conv = nn.Conv1d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=temporal_kernel,
                padding=temporal_kernel - 1,
            )
        else:
            self.temporal_conv = None

        self.initialize_weights()

    def regularizer(self) -> torch.Tensor:
        """Computes the L1 regularization term for all module parameters."""
        return sum(p.abs().sum() for p in self.parameters()) * self.gamma

    def initialize_weights(self):
        """Initializes the weights of the module."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight, gain=5 / 3)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)
        if self.temporal_conv is not None:
            init.kaiming_uniform_(self.temporal_conv.weight)
            if self.temporal_conv.bias is not None:
                self.temporal_conv.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time).

        Returns:
            torch.Tensor: Output tensor of shape (batch, channels, time, 1, 1).
        """
        # Transpose to (batch, time, channels) for MLP
        x = x.transpose(1, 2).contiguous()
        x = self.mlp(x)
        # Transpose back to (batch, channels, time) for convolution
        x = x.transpose(1, 2).contiguous()

        if self.temporal_conv is not None:
            n_frames = x.shape[2]
            # Slice output to maintain temporal dimension (causal convolution)
            x = self.temporal_conv(x)[..., :n_frames]

        # Expand spatial dimensions for broadcasting with core features
        return x.unsqueeze(-1).unsqueeze(-1)


class StackedMLPBehavior(nn.Module):
    """
    A container for multiple MLPBehavior modules that produces a dictionary of
    outputs, enabling layer-specific modulation in a larger network.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: Union[int, List[int]],
        layers: List[int],
        **kwargs,
    ):
        """
        Initializes the StackedMLPBehavior module.

        Args:
            input_channels (int): The number of input behavioral features.
            output_channels (Union[int, List[int]]): The number of output
                features for each MLP. If an int, it is used for all MLPs.
            layers (List[int]): A list of identifiers (e.g., layer indices)
                to use as keys in the output dictionary.
            **kwargs: Additional arguments passed to each MLPBehavior instance.
        """
        super().__init__()
        if isinstance(output_channels, int):
            output_channels = [output_channels] * len(layers)

        if len(output_channels) != len(layers):
            raise ValueError("Length of output_channels must match length of layers.")

        self.layers = layers
        self.mlps = nn.ModuleList(
            [
                MLPBehavior(
                    input_channels=input_channels,
                    output_channels=out_ch,
                    **kwargs,
                )
                for out_ch in output_channels
            ]
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Performs a forward pass through all MLPs.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, time).

        Returns:
            dict: A dictionary mapping layer identifiers to output tensors.
        """
        return {
            layer_id: mlp(x) for layer_id, mlp in zip(self.layers, self.mlps)
        }

    def regularizer(self) -> torch.Tensor:
        """Computes the sum of regularization terms from all MLPs."""
        return sum(mlp.regularizer() for mlp in self.mlps)