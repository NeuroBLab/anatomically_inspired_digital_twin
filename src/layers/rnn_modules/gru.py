"""
This module provides a convolutional Gated Recurrent Unit (GRU) for processing
sequential data in a neural network.
"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class RNNCore:
    """
    A mixin class providing common initialization and representation for RNN modules.

    This implementation is adapted from the one used in Sinz et al., 2018.
    """

    @staticmethod
    def init_conv(m: nn.Module):
        """
        Initializes the weights of a convolutional layer using Xavier normal
        initialization and sets the bias to zero.

        Args:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    def __repr__(self) -> str:
        """
        Extends the default representation to include regularization parameters.
        """
        s = super().__repr__()
        regularizers = []
        for attr in dir(self):
            if "gamma" in attr and not attr.startswith("_"):
                regularizers.append(f"{attr}={getattr(self, attr)}")
        if regularizers:
            s += f" [{self.__class__.__name__} regularizers: {' | '.join(regularizers)}]"
        return s


class ConvGRUCell(RNNCore, nn.Module):
    """
    A single-step Convolutional Gated Recurrent Unit (GRU) cell.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        input_kernel_size: int,
        recurrent_kernel_size: int,
        groups: int = 1,
        gamma_rec: float = 0.0,
        pad_input: bool = True,
    ):
        """
        Initializes the ConvGRUCell.

        Args:
            input_channels (int): The number of channels in the input tensor.
            hidden_channels (int): The number of channels in the hidden state.
            input_kernel_size (int): The size of the kernel for input convolutions.
            recurrent_kernel_size (int): The size of the kernel for recurrent
                convolutions.
            groups (int, optional): The number of groups for grouped convolutions.
                Defaults to 1.
            gamma_rec (float, optional): The regularization strength for recurrent
                connections. Defaults to 0.0.
            pad_input (bool, optional): If True, applies padding to the input
                convolutions to preserve spatial dimensions. Defaults to True.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.groups = groups
        self.gamma_rec = gamma_rec

        input_padding = input_kernel_size // 2 if pad_input else 0
        recurrent_padding = recurrent_kernel_size // 2
        self._shrinkage = 0 if pad_input else input_kernel_size - 1

        # Reset gate convolutions
        self.reset_gate_input = nn.Conv2d(
            input_channels, hidden_channels, input_kernel_size, padding=input_padding, groups=self.groups
        )
        self.reset_gate_hidden = nn.Conv2d(
            hidden_channels, hidden_channels, recurrent_kernel_size, padding=recurrent_padding, groups=self.groups
        )

        # Update gate convolutions
        self.update_gate_input = nn.Conv2d(
            input_channels, hidden_channels, input_kernel_size, padding=input_padding, groups=self.groups
        )
        self.update_gate_hidden = nn.Conv2d(
            hidden_channels, hidden_channels, recurrent_kernel_size, padding=recurrent_padding, groups=self.groups
        )

        # Output gate convolutions
        self.out_gate_input = nn.Conv2d(
            input_channels, hidden_channels, input_kernel_size, padding=input_padding, groups=self.groups
        )
        self.out_gate_hidden = nn.Conv2d(
            hidden_channels, hidden_channels, recurrent_kernel_size, padding=recurrent_padding, groups=self.groups
        )

        self.apply(self.init_conv)

    def init_state(self, input_tensor: torch.Tensor) -> Parameter:
        """
        Initializes the hidden state to a tensor of zeros.

        Args:
            input_tensor (torch.Tensor): The input tensor to infer batch size
                and spatial dimensions.

        Returns:
            Parameter: The initialized hidden state.
        """
        batch_size, _, *spatial_size = input_tensor.data.size()
        state_size = [batch_size, self.hidden_channels] + [
            s - self._shrinkage for s in spatial_size
        ]
        state = torch.zeros(*state_size, device=input_tensor.device)
        return Parameter(state)

    def forward(self, input_tensor: torch.Tensor, prev_state: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a single forward step of the GRU cell.

        Args:
            input_tensor (torch.Tensor): The input for the current time step.
            prev_state (torch.Tensor, optional): The hidden state from the
                previous time step. If None, it is initialized to zeros.
                Defaults to None.

        Returns:
            torch.Tensor: The new hidden state.
        """
        if prev_state is None:
            prev_state = self.init_state(input_tensor)

        # Update gate
        update = torch.sigmoid(
            self.update_gate_input(input_tensor) + self.update_gate_hidden(prev_state)
        )

        # Reset gate
        reset = torch.sigmoid(
            self.reset_gate_input(input_tensor) + self.reset_gate_hidden(prev_state)
        )

        # Candidate hidden state
        out = torch.tanh(
            self.out_gate_input(input_tensor) + self.out_gate_hidden(prev_state * reset)
        )

        # Final hidden state
        new_state = prev_state * (1 - update) + out * update
        return new_state

    def regularizer(self) -> torch.Tensor:
        """
        Computes the regularization term for the recurrent connections.
        """
        return self.gamma_rec * self.bias_l1()

    def bias_l1(self) -> torch.Tensor:
        """
        Computes a custom L1 penalty on the biases and weights of the
        recurrent connections.
        """
        # This specific combination of terms is preserved from the original implementation.
        l1_term = (
            self.reset_gate_hidden.bias.abs().mean()
            + self.update_gate_hidden.weight.abs().mean()
            + self.out_gate_hidden.bias.abs().mean()
        )
        return l1_term / 3


class GRU_Module(nn.Module):
    """
    A wrapper for ConvGRUCell that processes a sequence of inputs.

    This module iterates over the time dimension of a 4D or 5D input tensor,
    applying the GRU cell at each step.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        input_kernel_size: int,
        recurrent_kernel_size: int,
        **kwargs,
    ):
        """
        Initializes the GRU_Module.

        Args:
            input_channels (int): The number of channels in the input tensor.
            hidden_channels (int): The number of channels in the hidden state.
            input_kernel_size (int): The size of the kernel for input convolutions.
            recurrent_kernel_size (int): The size of the kernel for recurrent
                convolutions.
            **kwargs: Additional arguments passed to ConvGRUCell.
        """
        super().__init__()
        self.gru = ConvGRUCell(
            input_channels,
            hidden_channels,
            input_kernel_size,
            recurrent_kernel_size,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass over a sequence.

        Args:
            x (torch.Tensor): A 4D (C, T, H, W) or 5D (N, C, T, H, W) tensor.

        Returns:
            torch.Tensor: The sequence of hidden states, with the same
                dimensions as the input.
        """
        if x.dim() not in [4, 5]:
            raise ValueError(
                f"Expected 4D or 5D input, but got tensor with shape {x.shape}"
            )

        is_batched = x.dim() == 5
        if not is_batched:
            x = x.unsqueeze(0)  # Add a batch dimension

        hidden = None
        states = []
        time_dim = 2
        for t in range(x.shape[time_dim]):
            hidden = self.gru(x[:, :, t, :, :], hidden)
            states.append(hidden)

        out = torch.stack(states, dim=time_dim)

        if not is_batched:
            out = out.squeeze(0)  # Remove the batch dimension
        return out