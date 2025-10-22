import numpy as np
import torch
from torch import nn
from neuralpredictors.layers.readouts.base import Readout

from .readout import register


@register("factorized")
class FullFactorized2d(Readout):
    """
    A factorized fully connected layer.

    The weights are a sum of outer products between a spatial filter and a
    feature vector for each neuron.
    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias: bool = True,
        normalize=True,
        init_noise=1e-3,
        constrain_pos=False,
        positive_weights=False,
        shared_features=None,
        mean_activity=None,
        feature_reg_weight=None,
        gamma_readout=None,  # Deprecated
        **kwargs,
    ):
        """
        Initializes the FullFactorized2d readout.

        Args:
            in_shape (tuple): The shape of the input tensor (channels, height, width).
            outdims (int): The number of output neurons.
            bias (bool, optional): If True, a bias term is added. Defaults to True.
            normalize (bool, optional): If True, the spatial weights are normalized.
                Defaults to True.
            init_noise (float, optional): Standard deviation for weight initialization.
                Defaults to 1e-3.
            constrain_pos (bool, optional): If True, spatial weights are constrained
                to be non-negative. Defaults to False.
            positive_weights (bool, optional): If True, feature weights are constrained
                to be non-negative. Defaults to False.
            shared_features (dict, optional): Configuration for sharing feature vectors
                among neurons. Defaults to None.
            mean_activity (torch.Tensor, optional): Mean activity of neurons for bias
                initialization. Defaults to None.
            feature_reg_weight (float, optional): Regularization weight for features.
                Defaults to None.
            gamma_readout (float, optional): Deprecated. Use feature_reg_weight.
        """
        super().__init__()

        h, w = in_shape[1:]
        self.in_shape = in_shape
        self.outdims = outdims
        self.positive_weights = positive_weights
        self.constrain_pos = constrain_pos
        self.init_noise = init_noise
        self.normalize = normalize
        self.mean_activity = mean_activity
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(
            feature_reg_weight, gamma_readout, default=1.0
        )

        self._original_features = True
        self.initialize_features(**(shared_features or {}))
        self.spatial = nn.Parameter(torch.Tensor(self.outdims, h, w))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(outdims))
        else:
            self.register_parameter("bias", None)

        self.initialize(mean_activity)

    @property
    def shared_features(self):
        """Returns the shared feature parameters."""
        return self.features

    @property
    def features(self):
        """Returns the feature weights for each neuron."""
        if self._shared_features:
            return self.scales * self._features[self.feature_sharing_index, ...]
        return self._features

    @property
    def weight(self):
        """
        Constructs the full weight tensor from spatial and feature components.
        """
        if self.positive_weights:
            self.features.data.clamp_min_(0)
        n = self.outdims
        c, h, w = self.in_shape
        return self.normalized_spatial.view(n, 1, w, h) * self.features.view(
            n, c, 1, 1
        )

    @property
    def normalized_spatial(self):
        """
        Returns the spatial weights, normalized if specified.
        """
        if self.normalize:
            norm = (
                self.spatial.pow(2)
                .sum(dim=(1, 2), keepdim=True)
                .sqrt()
                .expand_as(self.spatial)
                + 1e-6
            )
            weight = self.spatial / norm
        else:
            weight = self.spatial
        if self.constrain_pos:
            weight.data.clamp_min_(0)
        return weight

    def regularizer(self, reduction="sum", average=None):
        """Computes the L1 regularization term for the features."""
        return self.l1(reduction=reduction, average=average) * self.feature_reg_weight

    def l1(self, reduction="sum", average=None):
        """Computes the L1 norm of the weights."""
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)
        if reduction is None:
            raise ValueError("Reduction of None is not supported in this regularizer")

        n = self.outdims
        c, h, w = self.in_shape
        ret = (
            self.normalized_spatial.view(self.outdims, -1).abs().sum(dim=1, keepdim=True)
            * self.features.view(self.outdims, -1).abs().sum(dim=1)
        ).sum()
        if reduction == "mean":
            ret /= n * c * w * h
        return ret

    def initialize(self, mean_activity=None):
        """Initializes the parameters of the readout."""
        if mean_activity is None:
            mean_activity = self.mean_activity
        self.spatial.data.normal_(0, self.init_noise)
        self._features.data.normal_(0, self.init_noise)
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        Initializes feature parameters, handling shared features if specified.
        """
        c = self.in_shape[0]
        if match_ids is not None:
            assert self.outdims == len(
                match_ids
            ), "match_ids must have same length as outdims"

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    n_match_ids,
                    c,
                ), f"shared_features shape mismatch"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = nn.Parameter(torch.Tensor(n_match_ids, c))
            self.scales = nn.Parameter(torch.Tensor(self.outdims, 1))
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = nn.Parameter(torch.Tensor(self.outdims, c))
            self._shared_features = False

    def forward(self, x, shift=None, **kwargs):
        """Performs a forward pass."""
        if shift is not None:
            raise NotImplementedError("Shift is not implemented for this readout.")
        if self.positive_weights:
            self.features.data.clamp_min_(0)

        c, h, w = x.shape[1:]
        c_in, h_in, w_in = self.in_shape
        if (c_in, h_in, w_in) != (c, h, w):
            raise ValueError("Input shape does not match expected in_shape.")

        y = torch.einsum("ncwh,owh->nco", x, self.normalized_spatial)
        y = torch.einsum("nco,oc->no", y, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        c, h, w = self.in_shape
        s = (
            f"{self.__class__.__name__}({c} x {h} x {w} -> {self.outdims}"
            f"{' with bias' if self.bias is not None else ''}"
        )
        if self._shared_features:
            s += f", with {'original' if self._original_features else 'shared'} features"
        s += f", {'normalized' if self.normalize else 'unnormalized'})"
        return s