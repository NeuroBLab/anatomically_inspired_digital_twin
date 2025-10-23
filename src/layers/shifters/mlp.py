import torch
from torch import nn
import torch.nn.init as init
from neuralpredictors.layers.shifters.base import Shifter

class MLP(Shifter):
    """
    A multi-layer perceptron that computes a 2D spatial shift.

    The network consists of one or more linear layers with Tanh activations.
    It is designed to take pupil center coordinates as input and predict a
    shift vector to adjust neuronal receptive fields.
    """

    def __init__(
        self, input_features: int = 2, hidden_channels: int = 10, shift_layers: int = 1, bias: bool = True
    ):
        """
        Initializes the MLP shifter.

        Args:
            input_features (int): Number of input features (e.g., 2 for x, y
                coordinates). Defaults to 2.
            hidden_channels (int): Number of units in the hidden layers.
            shift_layers (int): Total number of linear layers. A value of 1
                means no hidden layer.
            bias (bool): If True, adds a learnable bias to the linear layers.
        """
        super().__init__()
        layers = []
        in_features = input_features

        for _ in range(shift_layers - 1):
            layers.append(nn.Linear(in_features, hidden_channels, bias=bias))
            layers.append(nn.Tanh())
            in_features = hidden_channels

        layers.append(nn.Linear(in_features, 2, bias=bias))
        layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

        self.initialize_weights()

    def regularizer(self) -> torch.Tensor:
        """
        Computes the L1 regularization term for all shifter parameters.
        """
        return sum(p.abs().sum() for p in self.parameters())

    def initialize_weights(self):
        """
        Initializes the weights of linear layers with Xavier uniform
        initialization and sets biases to a small positive value.
        """
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight, gain=5 / 3)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

    def forward(self, pupil_center: torch.Tensor, trial_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the 2D shift based on pupil center coordinates.

        Args:
            pupil_center (torch.Tensor): A tensor of shape (batch_size, 2)
                containing pupil center coordinates.
            trial_idx (torch.Tensor, optional): A tensor of shape
                (batch_size, n) containing trial indices or other features
                to be concatenated with pupil_center. Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 2) representing the
                predicted spatial shift.
        """
        if trial_idx is not None:
            pupil_center = torch.cat((pupil_center, trial_idx), dim=1)

        expected_features = self.mlp[0].in_features
        if expected_features != pupil_center.shape[1]:
            raise ValueError(
                f"Input feature mismatch. Shifter expected {expected_features} "
                f"features, but got {pupil_center.shape[1]}."
            )
        return self.mlp(pupil_center)


class MLPShifter(nn.ModuleDict):
    """
    A ModuleDict container for managing multiple MLP shifters, one for each
    data key.
    """

    def __init__(self, args, data_keys, input_channels: int = 2, bias: bool = True, **kwargs):
        """
        Initializes the MLPShifter.

        Args:
            args: A configuration object containing shifter parameters like
                `gamma_shifter`, `hidden_channels_shifter`, and `shift_layers`.
            data_keys (list[str]): A list of keys, where each key corresponds
                to a unique dataset or experimental session.
            input_channels (int): The number of input features for each MLP.
            bias (bool): If True, enables bias in the MLP layers.
        """
        super().__init__()
        self.register_buffer("gamma_shifter", torch.tensor(args.gamma_shifter))
        for k in data_keys:
            self.add_module(
                k,
                MLP(
                    input_channels,
                    args.hidden_channels_shifter,
                    args.shift_layers,
                    bias,
                ),
            )

    def regularizer(self, data_key: str) -> torch.Tensor:
        """
        Computes the scaled L1 regularization term for a specific shifter.

        Args:
            data_key (str): The key identifying which shifter to regularize.

        Returns:
            torch.Tensor: The regularization value.
        """
        return self[data_key].regularizer() * self.gamma_shifter