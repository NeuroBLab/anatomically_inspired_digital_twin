import warnings

import numpy as np
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from neuralpredictors.layers.readouts.base import ConfigurationError, Readout

from src.layers.readouts.readout import register


@register("gaussian")
class FullGaussian2d(Readout):
    """
    A readout using a spatial transformer layer where neuron positions are
    sampled from a 2D Gaussian distribution. The mean and covariance of the
    Gaussian for each neuron are learned parameters.
    """

    def __init__(
        self,
        in_shape,
        outdims,
        brain_area,
        brain_area_to_layer=None,
        bias: bool = True,
        init_mu_range=0.1,
        init_sigma=1.0,
        batch_sample=True,
        align_corners=True,
        gauss_type="full",
        grid_mean_predictor=None,
        shared_features=None,
        shared_grid=None,
        source_grid=None,
        mean_activity=None,
        feature_reg_weight=None,
        dispersion_reg=0.0,
        gamma_readout=None,  # Deprecated
        num_samples=1,
        hidden_channels=None,
        regularize_per_layer=False,
        modulator_channels=0,
        **kwargs,
    ):
        """
        Initializes the FullGaussian2d readout.

        Args:
            in_shape (tuple): Shape of the input feature map (channels, width, height).
            outdims (int): Number of output neurons.
            brain_area (np.ndarray): Array indicating the brain area for each neuron.
            brain_area_to_layer (dict, optional): Mapping from brain areas to core
                layer indices. Defaults to None.
            bias (bool, optional): If True, adds a bias term. Defaults to True.
            init_mu_range (float, optional): Range for uniform initialization of
                the Gaussian means. Defaults to 0.1.
            init_sigma (float, optional): Value for initializing the Gaussian
                standard deviation. Defaults to 1.0.
            batch_sample (bool, optional): If True, samples a different position for
                each batch item. Defaults to True.
            align_corners (bool, optional): `align_corners` argument for
                `F.grid_sample`. Defaults to True.
            gauss_type (str, optional): Type of Gaussian covariance: 'full',
                'uncorrelated', or 'isotropic'. Defaults to "full".
            grid_mean_predictor (dict, optional): Configuration for a network that
                predicts grid means. Defaults to None.
            shared_features (dict, optional): Configuration for sharing feature
                vectors. Defaults to None.
            shared_grid (dict, optional): Configuration for sharing grid parameters.
                Defaults to None.
            source_grid (np.ndarray, optional): Source grid for the
                grid_mean_predictor. Defaults to None.
            mean_activity (torch.Tensor, optional): Mean activity for bias
                initialization. Defaults to None.
            feature_reg_weight (float, optional): L1 regularization weight for
                features. Defaults to None.
            dispersion_reg (float, optional): Regularization weight for the
                position network. Defaults to 0.0.
            gamma_readout (float, optional): Deprecated. Use feature_reg_weight.
            num_samples (int, optional): Number of spatial samples to draw per neuron.
                Defaults to 1.
            hidden_channels (list, optional): Number of channels per hidden layer,
                used for layer-wise regularization. Defaults to None.
            regularize_per_layer (bool, optional): If True, applies regularization
                per layer. Defaults to False.
            modulator_channels (int, optional): Number of channels from a modulator
                to concatenate. Defaults to 0.
        """
        super().__init__()
        if not 0.0 < init_mu_range <= 1.0:
            raise ValueError("init_mu_range must be in (0, 1]")
        if init_sigma <= 0.0:
            raise ValueError("init_sigma must be positive")

        self.in_shape = in_shape
        self.outdims = outdims
        self.brain_area = brain_area
        self.brain_area_to_layer = brain_area_to_layer
        self.modulator_channels = modulator_channels
        self.mean_activity = mean_activity
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(
            feature_reg_weight, gamma_readout, default=1.0
        )
        self.dispersion_reg = dispersion_reg
        self.gauss_type = gauss_type
        self.batch_sample = batch_sample
        self.num_samples = num_samples
        self.align_corners = align_corners
        self.grid_shape = (1, outdims, 1, 2)
        self.hidden_channels = hidden_channels
        self.regularize_per_layer = regularize_per_layer

        # Initialize grid parameters
        self._predicted_grid = False
        self._shared_grid = False
        self._original_grid = True
        if grid_mean_predictor is not None and shared_grid is not None:
            raise ConfigurationError(
                "grid_mean_predictor and shared_grid cannot both be set."
            )
        elif grid_mean_predictor is not None:
            self.init_grid_predictor(source_grid=source_grid, **grid_mean_predictor)
            self._original_grid = False
        elif shared_grid is not None:
            self.initialize_shared_grid(**(shared_grid or {}))
        else:
            self._mu = Parameter(torch.Tensor(*self.grid_shape))

        # Initialize covariance parameters based on gauss_type
        if gauss_type == "full":
            self.sigma_shape = (1, outdims, 2, 2)
        elif gauss_type == "uncorrelated":
            self.sigma_shape = (1, outdims, 1, 2)
        elif gauss_type == "isotropic":
            self.sigma_shape = (1, outdims, 1, 1)
        else:
            raise ValueError(f"Unknown gauss_type: {gauss_type}")
        self.sigma = Parameter(torch.Tensor(*self.sigma_shape))
        self.init_sigma = init_sigma

        self.initialize_features(**(shared_features or {}))

        if bias:
            self.bias = Parameter(torch.Tensor(outdims))
        else:
            self.register_parameter("bias", None)

        self.init_mu_range = init_mu_range
        self.initialize(mean_activity)

    @property
    def shared_features(self):
        """Returns the shared feature parameters."""
        return self._features

    @property
    def shared_grid(self):
        """Returns the shared grid mean parameters."""
        return self._mu

    @property
    def features(self):
        """Returns the feature weights for each neuron."""
        if self._shared_features:
            return self.scales * self._features[..., self.feature_sharing_index]
        return self._features

    @property
    def grid(self):
        """Returns the mean grid locations."""
        return self.sample_grid(batch_size=1, sample=False)

    @property
    def mu(self):
        """
        Returns the mean of the Gaussian grid for each neuron, applying any
        transformations or sharing logic.
        """
        if self._predicted_grid:
            return self.mu_transform(self.source_grid.squeeze()).view(*self.grid_shape)
        if self._shared_grid:
            if self._original_grid:
                return self._mu[:, self.grid_sharing_index, ...]
            return self.mu_transform(self._mu.squeeze())[
                self.grid_sharing_index
            ].view(*self.grid_shape)
        return self._mu

    def feature_l1(self, reduction="sum", average=None):
        """Computes the L1 regularization term for features."""
        if self.regularize_per_layer:
            return self.apply_reduction_per_layer(reduction=reduction, average=average)
        if self._original_features:
            return self.apply_reduction(
                self.features.abs(), reduction=reduction, average=average
            )
        return 0.0

    def apply_reduction_per_layer(self, reduction="sum", average=None):
        """Applies L1 regularization to features on a per-layer basis."""
        max_num_channels = max(self.hidden_channels)
        indices = [0] + np.cumsum(self.hidden_channels).tolist()
        reg = 0.0
        for i, n_channels in enumerate(self.hidden_channels):
            reg += (
                self.features[0, indices[i] : indices[i + 1], ...].abs().sum()
                * (n_channels / max_num_channels)
            )
        return reg

    def position_network_l1(self, reduction="sum", average=None):
        """Computes the L1 regularization for the position predictor network."""
        if self._predicted_grid:
            reg = self.apply_reduction(
                self.mu_transform[0].weight.abs(), reduction=reduction, average=average
            )
            reg += self.apply_reduction(
                self.mu_transform[2].weight.abs(), reduction=reduction, average=average
            )
            return reg
        return 0.0

    def regularizer(self, reduction="sum", average=None):
        """Computes the total regularization term for the readout."""
        feature_reg = self.feature_reg_weight * self.feature_l1(
            reduction=reduction, average=average
        )
        dispersion_reg = self.dispersion_reg * self.position_network_l1(
            reduction=reduction, average=average
        )
        return feature_reg + dispersion_reg

    def sample_grid(self, batch_size, sample=None):
        """
        Samples grid locations from the Gaussian distribution for each neuron.

        Args:
            batch_size (int): The batch size.
            sample (bool, optional): If True, samples from the distribution. If
                False, uses the mean. If None, samples during training and uses
                the mean during evaluation. Defaults to None.

        Returns:
            torch.Tensor: A tensor of grid locations.
        """
        self.mu.requires_grad_()
        with torch.no_grad():
            self.mu.clamp_(min=-1, max=1)

        grid_shape = (batch_size, self.outdims, self.num_samples, 2)
        if self.num_samples > 1:
            sample = True

        if sample is None:
            sample = self.training

        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(*grid_shape).zero_()

        if self.gauss_type != "full":
            grid = norm * self.sigma + self.mu
        else:
            grid = torch.einsum("ancd,bnid->bnic", self.sigma, norm) + self.mu

        return torch.clamp(grid, min=-1, max=1)

    def init_grid_predictor(
        self, source_grid, hidden_features=30, hidden_layers=1, final_tanh=True
    ):
        """Initializes the grid mean predictor network."""
        self._original_grid = False
        layers = [
            nn.Linear(
                source_grid.shape[1], hidden_features if hidden_layers > 0 else 2
            )
        ]
        for i in range(hidden_layers):
            layers.extend(
                [
                    nn.ELU(),
                    nn.Linear(
                        hidden_features,
                        hidden_features if i < hidden_layers - 1 else 2,
                    ),
                ]
            )
        if final_tanh:
            layers.append(nn.Tanh())
        self.mu_transform = nn.Sequential(*layers)

        source_grid_norm = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid_norm /= np.abs(source_grid_norm).max()
        self.register_buffer(
            "source_grid", torch.from_numpy(source_grid_norm.astype(np.float32))
        )
        self._predicted_grid = True
        self.init_weights(self.mu_transform)

    def init_weights(self, m):
        """Initializes weights of the grid predictor."""
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                if mod.out_features == 2:
                    init.xavier_uniform_(mod.weight, gain=5 / 3)
                else:
                    init.kaiming_uniform_(mod.weight, mode="fan_in")
                if mod.bias is not None:
                    mod.bias.data.fill_(0.01)

    def initialize(self, mean_activity=None):
        """Initializes all parameters of the readout."""
        if mean_activity is None:
            mean_activity = self.mean_activity
        if not self._predicted_grid or self._original_grid:
            self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gauss_type != "full":
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)

        self._features.data.fill_(1 / self.in_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def initialize_features(self, match_ids=None, shared_features=None):
        """Initializes feature parameters, handling sharing logic."""
        c, w, h = self.in_shape
        self._original_features = True
        if match_ids is not None:
            assert self.outdims == len(match_ids)
            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                expected_shape = (1, c, 1, n_match_ids)
                assert shared_features.shape == expected_shape, (
                    f"shared_features shape mismatch. Expected {expected_shape}, "
                    f"got {shared_features.shape}"
                )
                self._features = shared_features
                self._original_features = False
            else:
                self._features = Parameter(torch.Tensor(1, c, 1, n_match_ids))
            self.scales = Parameter(torch.Tensor(1, 1, 1, self.outdims))
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = Parameter(torch.Tensor(1, c, 1, self.outdims))
            self._shared_features = False

    def initialize_shared_grid(self, match_ids=None, shared_grid=None):
        """Initializes grid parameters for sharing."""
        if match_ids is None:
            raise ConfigurationError("match_ids must be provided for sharing grid.")
        assert self.outdims == len(
            match_ids
        ), "One match_id per output dimension is required."

        n_match_ids = len(np.unique(match_ids))
        if shared_grid is not None:
            expected_shape = (1, n_match_ids, 1, 2)
            assert shared_grid.shape == expected_shape, (
                f"shared_grid shape mismatch. Expected {expected_shape}, "
                f"got {shared_grid.shape}"
            )
            self._mu = shared_grid
            self._original_grid = False
            self.mu_transform = nn.Linear(2, 2)
            self.mu_transform.bias.data.fill_(0.0)
            self.mu_transform.weight.data = torch.eye(2)
        else:
            self._mu = Parameter(torch.Tensor(1, n_match_ids, 1, 2))

        _, sharing_idx = np.unique(match_ids, return_inverse=True)
        self.register_buffer("grid_sharing_index", torch.from_numpy(sharing_idx))
        self._shared_grid = True

    def forward(self, x, sample=None, shift=None, out_idx=None, **kwargs):
        """
        Performs a forward pass through the readout.

        Args:
            x (torch.Tensor): Input tensor from the core.
            sample (bool, optional): Overrides sampling behavior. Defaults to None.
            shift (torch.Tensor, optional): A tensor to shift the grid locations.
                Defaults to None.
            out_idx (np.ndarray, optional): Indices of neurons to compute output for.
                Defaults to None.

        Returns:
            torch.Tensor: The predicted neural activity.
        """
        N, c, w, h = x.shape
        c_in, w_in, h_in = self.in_shape
        feat = self.features.view(1, c_in, self.outdims)
        bias = self.bias
        outdims = self.outdims

        grid = self.sample_grid(batch_size=N, sample=sample)

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray) and out_idx.dtype == bool:
                out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x.contiguous(), grid.contiguous(), align_corners=self.align_corners)
        y = y.mean(-1)

        if self.brain_area_to_layer is not None:
            out = torch.zeros(N, outdims, device=y.device)
            for area, v in self.brain_area_to_layer.items():
                area_idxs = self.brain_area == area
                y_area_parts = [y[:, el[0] : el[1], area_idxs] for el in v]
                y_area_parts.append(y[:, -self.modulator_channels :, area_idxs])
                y_area = torch.cat(y_area_parts, dim=1)
                out[:, area_idxs] = (y_area * feat[:, :, area_idxs]).sum(1)
            y = out
        else:
            y = (y * feat).sum(1)

        if self.bias is not None:
            y = y + bias
        return y

    def __repr__(self):
        c, h, w = self.in_shape
        s = (
            f"{self.gauss_type} {self.__class__.__name__}"
            f"({c}x{h}x{w} -> {self.outdims}"
        )
        if self.bias is not None:
            s += " with bias"
        if self._shared_features:
            s += f", with {'original' if self._original_features else 'shared'} features"
        if self._predicted_grid:
            s += ", with predicted grid"
        if self._shared_grid:
            s += f", with {'original' if self._original_grid else 'shared'} grid"
        s += ")"
        return s


@register("twopartgaussian")
class TwoPartFullGaussian2D(FullGaussian2d):
    """
    A Gaussian readout for a two-part (hurdle) model.

    It computes a binary decision (whether a neuron is active) and the
    magnitude of the response if active.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the TwoPartFullGaussian2D readout."""
        super().__init__(*args, **kwargs)
        c, _, _ = self.in_shape
        self.zero_features = Parameter(torch.Tensor(1, c, 1, self.outdims))
        self.threshold = 0.5
        self._initialize_zero_features()

    def _initialize_zero_features(self):
        """Initializes the feature weights for the binary part of the model."""
        init.normal_(self.zero_features, mean=0, std=0.01)

    def forward(self, x, sample=None, shift=None, out_idx=None, **kwargs):
        """
        Performs a forward pass.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - prob_zero: Logits for the binary decision (active vs. inactive).
                - y: The predicted response magnitude (regression part).
                - response: The final response (y * binary_decision).
        """
        N, c, w, h = x.shape
        feat = self.features.view(1, c, self.outdims)
        grid = self._get_grid(N, sample, out_idx)
        if shift is not None:
            grid = grid + shift[:, None, None, :]

        cores_window = F.grid_sample(x, grid, align_corners=self.align_corners)
        y = (cores_window.squeeze(-1) * feat).sum(1).view(N, self.outdims)
        y = F.relu(y)

        zero_feat = self.zero_features.view(1, c, self.outdims)
        prob_zero = (cores_window.squeeze(-1) * zero_feat).sum(1).view(N, self.outdims)

        if self.bias is not None:
            y = y + self.bias
            prob_zero = prob_zero + self.bias

        decision = self.compute_binary_decision(prob_zero)
        response = decision * y

        return prob_zero, y, response

    def _get_grid(self, N, sample, out_idx):
        """Helper method to sample or get a fixed grid."""
        if self.batch_sample:
            grid = self.sample_grid(batch_size=N, sample=sample)
        else:
            grid = self.sample_grid(batch_size=1, sample=sample).expand(
                N, self.outdims, 1, 2
            )

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray) and out_idx.dtype == bool:
                out_idx = np.where(out_idx)[0]
            grid = grid[:, out_idx]
        return grid

    def compute_binary_decision(self, prob_zero):
        """
        Computes a binary decision based on logits and a threshold.
        """
        prob_zero_sigmoid = torch.sigmoid(prob_zero)
        return (prob_zero_sigmoid > self.threshold).float()