import torch
import torch.nn as nn

class Exp(nn.Module):
    """Applies the element-wise exponential function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor with the exponential function applied.
        """
        return torch.exp(x)


class Softplus(nn.Module):
    """
    Applies the Softplus function element-wise.

    Softplus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to be always positive.
    """

    def __init__(self, beta: int = 1, threshold: int = 20):
        """
        Args:
            beta (int): The beta value for the Softplus formulation.
            threshold (int): Values above this threshold are approximated as
                identity to improve numerical stability.
        """
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor with the Softplus function applied.
        """
        return nn.functional.softplus(x, self.beta, self.threshold)


class LearnableSoftplus(nn.Module):
    """
    Applies a Softplus function with a learnable beta parameter.
    """

    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta (float): The initial value for the learnable beta parameter.
        """
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor with the learnable Softplus
                function applied.
        """
        # This implementation is a numerically stable version of the softplus
        # function: log(1 + exp(x * beta)) / beta
        xb = x * self.beta
        return (torch.clamp(xb, 0) + torch.log1p(torch.exp(-torch.abs(xb)))) / self.beta


class PReLU(nn.Module):
    """
    Applies the Parametric Rectified Linear Unit function (PReLU).

    This is a learnable version of LeakyReLU, where the coefficient of the
    negative part is a learnable parameter.
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        """
        Args:
            num_parameters (int): Number of parameters to learn. Can be 1 for
                a shared parameter across all channels, or the number of
                channels for a channel-wise parameter.
            init (float): The initial value of the learnable parameter.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_parameters,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor with the PReLU function applied.
        """
        return nn.functional.prelu(x, self.weight)