import itertools
from collections import OrderedDict

import torch
from neuralpredictors import regularizers
from neuralpredictors.layers.activations import AdaptiveELU
from neuralpredictors.layers.affine import Bias3DLayer, Scale3DLayer
from neuralpredictors.layers.cores.conv3d import Core3d
from neuralpredictors.regularizers import DepthLaplaceL21d
from neuralpredictors.utils import check_hyperparam_for_layers

from src.layers.cores.core import register


@register("conv3d")
class Basic3dCore(Core3d, torch.nn.Module):
    """
    A multi-layer 3D convolutional core.

    Attributes:
        features (torch.nn.Sequential): The sequential container of convolutional layers.
        ... and other attributes storing hyperparameters.
    """

    def __init__(
        self,
        args,
        stride=1,
        hidden_nonlinearities="elu",
        x_shift=0,
        y_shift=0,
        bias=True,
        padding=True,
        batch_norm_scale=True,
        momentum=0.5,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        final_nonlin=True,
        independent_bn_bias=True,
        spatial_dilation: int = 1,
        temporal_dilation: int = 1,
        hidden_spatial_dilation=1,
        hidden_temporal_dilation=1,
    ):
        """
        Initializes the Basic3dCore module.

        Args:
            args: An object containing model hyperparameters. Expected attributes include:
                layers (int): Number of convolutional layers.
                core_input_channels (int): Number of input channels.
                input_kernel (int or tuple): Kernel size for the first layer.
                hidden_channels (int or list): Number of channels in hidden layers.
                hidden_kernel (int or tuple): Kernel size for hidden layers.
                batch_norm (bool): Whether to use batch normalization.
                layer_norm (bool): Whether to use layer normalization.
                gamma_input_spatial (float): Regularization factor for spatial smoothing.
                gamma_input_temporal (float): Regularization factor for temporal smoothing.
                frames (int): Number of temporal frames to process.
                use_residuals (bool): Whether to use residual connections.
                input_dim (tuple, optional): Input dimensions for layer norm shape calculation.
                device (str): The device to run the model on ('cuda' or 'cpu').
            stride (int): The stride of the convolutions.
            hidden_nonlinearities (str): Type of nonlinearity to use (e.g., 'elu', 'relu').
            x_shift (float): x-shift for AdaptiveELU.
            y_shift (float): y-shift for AdaptiveELU.
            bias (bool): Whether to include a bias term in convolutions.
            padding (bool): Whether to pad convolutions to preserve spatial dimensions.
            batch_norm_scale (bool): If True, learns a scaling factor in BatchNorm.
            momentum (float): Momentum for batch normalization.
            laplace_padding (int, optional): Padding for the Laplace regularizer.
            input_regularizer (str): The type of regularizer for the input weights.
            final_nonlin (bool): Whether to apply a nonlinearity after the final layer.
            independent_bn_bias (bool): If True, BatchNorm learns its own bias.
            spatial_dilation (int): Dilation for the first spatial kernel.
            temporal_dilation (int): Dilation for the first temporal kernel.
            hidden_spatial_dilation (int or list): Dilation for hidden spatial kernels.
            hidden_temporal_dilation (int or list): Dilation for hidden temporal kernels.
        """
        super().__init__()

        # --- Initialize hyperparameters from args ---
        self.layers = args.layers
        self.input_channels = args.core_input_channels
        self.input_kernel = args.input_kernel
        self.hidden_channels = args.hidden_channels
        self.hidden_kernel = args.hidden_kernel
        self.batch_norm = args.batch_norm
        self.layer_norm = args.layer_norm
        self.gamma_input_spatial = args.gamma_input_spatial
        self.gamma_input_temporal = args.gamma_input_temporal
        self.num_frames = args.frames
        self.use_residuals = args.use_residuals
        self.idim = getattr(args, "input_dim", None)

        # --- Initialize other attributes ---
        self.stride = stride
        self.bias = bias
        self.padding = padding
        self.batch_norm_scale = batch_norm_scale
        self.independent_bn_bias = independent_bn_bias
        self.momentum = momentum
        self.spatial_dilation = spatial_dilation
        self.temporal_dilation = temporal_dilation
        self.final_nonlinearity = final_nonlin
        self.nonlinearities = {
            "elu": torch.nn.ELU,
            "softplus": torch.nn.Softplus,
            "relu": torch.nn.ReLU,
            "adaptive_elu": AdaptiveELU,
        }

        # --- Configure regularizers ---
        reg_config = (
            dict(padding=laplace_padding, kernel=args.input_kernel)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weight_regularizer = getattr(regularizers, input_regularizer)(
            **reg_config
        )
        self.temporal_regularizer = DepthLaplaceL21d()

        # --- Standardize hyperparameter formats ---
        self.hidden_channels = check_hyperparam_for_layers(
            self.hidden_channels, self.layers
        )
        self.output_channels = self.hidden_channels[-1]
        self.hidden_temporal_dilation = check_hyperparam_for_layers(
            hidden_temporal_dilation, self.layers
        )
        self.hidden_spatial_dilation = check_hyperparam_for_layers(
            hidden_spatial_dilation, self.layers
        )

        if isinstance(self.input_kernel, int):
            self.input_kernel = (self.input_kernel,) * 3
        if isinstance(self.hidden_kernel, int):
            self.hidden_kernel = (self.hidden_kernel,) * 3
        if isinstance(self.hidden_kernel, (tuple, list)):
            self.hidden_kernel = [self.hidden_kernel] * (self.layers - 1)
        if isinstance(self.stride, int):
            self.stride = [self.stride] * self.layers

        # --- Build network layers ---
        self.features = torch.nn.Sequential()
        self._build_layers(hidden_nonlinearities, x_shift, y_shift)

        # --- Initialize residual connections if specified ---
        if self.use_residuals:
            self.residual_blocks = torch.nn.ModuleList()
            for i in range(1, len(self.hidden_channels)):
                in_channels = self.hidden_channels[i - 1]
                out_channels = self.hidden_channels[i]
                self.residual_blocks.append(
                    torch.nn.Conv3d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0
                    )
                )

        self.initialize(cuda=(args.device == "cuda"))
        self.initialize_weights()

    def _build_layers(self, hidden_nonlinearities, x_shift, y_shift):
        """Constructs the convolutional layers of the core."""
        # Calculate shapes for LayerNorm if enabled
        normalized_shapes = self._get_normalized_shapes() if self.layer_norm else None

        # --- First Layer ---
        layer = OrderedDict()
        pad = (
            (
                self.input_kernel[0] - 1,
                self.input_kernel[1] // 2,
                self.input_kernel[2] // 2,
            )
            if self.padding
            else 0
        )
        layer["conv"] = torch.nn.Conv3d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels[0],
            kernel_size=self.input_kernel,
            stride=(1, self.stride[0], self.stride[0]),
            dilation=(
                self.temporal_dilation,
                self.spatial_dilation,
                self.spatial_dilation,
            ),
            bias=self.bias,
            padding=pad,
        )

        self.add_bn_layer(layer, self.hidden_channels[0])
        if self.layer_norm:
            self.add_ln_layer(layer, normalized_shapes[0])

        if self.layers > 1 or self.final_nonlinearity:
            if hidden_nonlinearities == "adaptive_elu":
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](
                    xshift=x_shift, yshift=y_shift
                )
            else:
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()

        self.features.add_module("layer0", torch.nn.Sequential(layer))

        # --- Hidden Layers ---
        for l in range(self.layers - 1):
            layer = OrderedDict()
            pad = (
                (
                    self.hidden_kernel[l][0] - 1,
                    self.hidden_kernel[l][1] // 2,
                    self.hidden_kernel[l][2] // 2,
                )
                if self.padding
                else 0
            )
            layer[f"conv_{l + 1}"] = torch.nn.Conv3d(
                self.hidden_channels[l],
                self.hidden_channels[l + 1],
                kernel_size=self.hidden_kernel[l],
                dilation=(
                    self.hidden_temporal_dilation[l],
                    self.hidden_spatial_dilation[l],
                    self.hidden_spatial_dilation[l],
                ),
                stride=(1, self.stride[l + 1], self.stride[l + 1]),
                padding=pad,
            )

            self.add_bn_layer(layer, self.hidden_channels[l + 1])
            if self.layer_norm:
                self.add_ln_layer(layer, normalized_shapes[l + 1])

            if self.final_nonlinearity or l < self.layers - 2:
                if hidden_nonlinearities == "adaptive_elu":
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](
                        x_shift=x_shift, y_shift=y_shift
                    )
                else:
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()

            self.features.add_module(f"layer{l + 1}", torch.nn.Sequential(layer))

    def _get_normalized_shapes(self):
        """Computes the feature map shapes for each layer for LayerNorm."""
        if not self.idim:
            raise ValueError("input_dim is required for layer normalization.")

        shapes = []
        # First layer
        shape = self.calc_activation_shape(
            self.idim,
            self.input_kernel,
            (self.temporal_dilation, self.spatial_dilation, self.spatial_dilation),
            (1, self.stride[0], self.stride[0]),
            (0, self.input_kernel[1] // 2, self.input_kernel[2] // 2)
            if self.padding
            else (0, 0, 0),
        )
        shapes.append((self.hidden_channels[0], *shape))

        # Hidden layers
        for l in range(self.layers - 1):
            shape = self.calc_activation_shape(
                shapes[l][1:],
                self.hidden_kernel[l],
                (
                    self.hidden_temporal_dilation[l],
                    self.hidden_spatial_dilation[l],
                    self.hidden_spatial_dilation[l],
                ),
                (1, self.stride[l + 1], self.stride[l + 1]),
                (0, self.hidden_kernel[l][1] // 2, self.hidden_kernel[l][2] // 2)
                if self.padding
                else (0, 0, 0),
            )
            shapes.append((self.hidden_channels[l + 1], *shape))
        return shapes

    def initialize_weights(self):
        """Initializes the weights of all convolutional layers."""
        for layer in self.features:
            if isinstance(layer, torch.nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, torch.nn.Conv3d):
                        torch.nn.init.kaiming_uniform_(sublayer.weight)
                        if sublayer.bias is not None:
                            sublayer.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Computes the forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch, channels, T, H, W).

        Returns:
            torch.Tensor: The output feature map.
        """
        # Process first layer and truncate temporal dimension
        x = self.features[0](x)[:, :, : self.num_frames, :, :]

        # Process subsequent layers
        for i, layer in enumerate(self.features[1:], 1):
            if self.use_residuals:
                residual = self.residual_blocks[i - 1](x)
                x = layer(x)[:, :, : self.num_frames, :, :] + residual
            else:
                x = layer(x)[:, :, : self.num_frames, :, :]
        return x

    def laplace_spatial(self):
        """
        Computes the spatial Laplace regularizer for the first convolutional layer.

        Returns:
            torch.Tensor: The spatial regularization term.
        """
        laplace = 0
        conv_weight = self.features[0].conv.weight
        for i in range(conv_weight.shape[2]):
            laplace += self._input_weight_regularizer(conv_weight[:, :, i, :, :])
        return laplace

    def laplace_temporal(self):
        """
        Computes the temporal Laplace regularizer for the first convolutional layer.

        Returns:
            torch.Tensor: The temporal regularization term.
        """
        laplace = 0
        conv_weight = self.features[0].conv.weight
        for w, h in itertools.product(
            range(conv_weight.shape[-2]), range(conv_weight.shape[-1])
        ):
            laplace += self.temporal_regularizer(conv_weight[:, :, :, w, h])
        return laplace

    def regularizer(self):
        """
        Computes the total regularization term for the input weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the spatial and
                temporal regularization terms.
        """
        return (
            self.gamma_input_spatial * self.laplace_spatial(),
            self.gamma_input_temporal * self.laplace_temporal(),
        )

    def add_bn_layer(self, layer, hidden_channels):
        """
        Adds a batch normalization layer to the layer dictionary.

        Args:
            layer (OrderedDict): The dictionary of layers for a Sequential block.
            hidden_channels (int): The number of channels for the BatchNorm layer.
        """
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = torch.nn.BatchNorm3d(
                    hidden_channels, momentum=self.momentum
                )
            else:
                affine = self.bias and self.batch_norm_scale
                layer["norm"] = torch.nn.BatchNorm3d(
                    hidden_channels, momentum=self.momentum, affine=affine
                )
                if self.bias and not self.batch_norm_scale:
                    layer["bias"] = Bias3DLayer(hidden_channels)
                elif self.batch_norm_scale:
                    layer["scale"] = Scale3DLayer(hidden_channels)

    def add_ln_layer(self, layer, normalized_shape):
        """
        Adds a layer normalization layer to the layer dictionary.

        Args:
            layer (OrderedDict): The dictionary of layers for a Sequential block.
            normalized_shape (tuple): The shape over which to normalize.
        """
        layer["norm"] = torch.nn.LayerNorm(normalized_shape)

    @property
    def out_channels(self):
        """Returns the number of output channels."""
        return self.hidden_channels[-1]

    @property
    def is_2d(self):
        """Returns False as this is a 3D core."""
        return False

    def get_kernels(self):
        """Returns a list of all kernel sizes used in the core."""
        return [self.input_kernel] + self.hidden_kernel

    @staticmethod
    def calc_activation_shape(dim, ksize, dilation, stride, padding) -> tuple:
        """
        Calculates the output shape of a 3D convolution.

        Args:
            dim (tuple): Input dimensions (T, H, W).
            ksize (tuple): Kernel size (kT, kH, kW).
            dilation (tuple): Dilation (dT, dH, dW).
            stride (tuple): Stride (sT, sH, sW).
            padding (tuple): Padding (pT, pH, pW).

        Returns:
            tuple: Output dimensions (T_out, H_out, W_out).
        """

        def shape_each_dim(i):
            odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
            return int(odim_i / stride[i]) + 1

        return shape_each_dim(0), shape_each_dim(1), shape_each_dim(2)


@register("factorizedconv3d")
class Factorized3dCore(Core3d, torch.nn.Module):
    """
    A multi-layer 3D convolutional core with factorized convolutions.

    Each 3D convolution is separated into a spatial convolution (2D) followed
    by a temporal convolution (1D).

    Attributes:
        features (torch.nn.Sequential): The sequential container of layers.
        ... and other attributes storing hyperparameters.
    """

    def __init__(
        self,
        args,
        final_nonlin=True,
        stride=1,
        x_shift=0.0,
        y_shift=0.0,
        hidden_nonlinearities="elu",
        bias=True,
        batch_norm_scale=True,
        independent_bn_bias=True,
        momentum=0.01,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        spatial_dilation=1,
        temporal_dilation=1,
        hidden_spatial_dilation=1,
        hidden_temporal_dilation=1,
    ):
        """
        Initializes the Factorized3dCore module.

        Args:
            args: An object containing model hyperparameters. See Basic3dCore for
                a list of common attributes. Specific attributes for this class include:
                spatial_input_kernel (int or tuple): Spatial kernel for the first layer.
                temporal_input_kernel (int): Temporal kernel for the first layer.
                spatial_hidden_kernel (int or tuple): Spatial kernel for hidden layers.
                temporal_hidden_kernel (int): Temporal kernel for hidden layers.
                dropout_rate (float): Dropout probability.
                pooling (bool): Whether to use max pooling.
                pooling_layers (list): Indices of layers to apply pooling after.
                behavior_integration_mode (str): How to integrate behavior ('mul' or 'sum').
                modulator_layers (list): Indices of layers to integrate behavior.
                concat_behavior (bool): If True, concatenates behavior at the output.
                stack (list or int): Layer indices to stack for the final output.
            ... (other arguments are similar to Basic3dCore).
        """
        super().__init__()

        # --- Initialize hyperparameters from args ---
        self.layers = args.layers
        self.input_channels = args.core_input_channels
        self.hidden_channels = args.hidden_channels
        self.spatial_input_kernel = args.spatial_input_kernel
        self.temporal_input_kernel = args.temporal_input_kernel
        self.spatial_hidden_kernel = args.spatial_hidden_kernel
        self.temporal_hidden_kernel = args.temporal_hidden_kernel
        self.batch_norm = args.batch_norm
        self.padding = args.padding
        self.gamma_input_spatial = args.gamma_input_spatial
        self.gamma_input_temporal = args.gamma_input_temporal
        self.num_frames = args.frames
        self.use_residuals = args.use_residuals
        self.pooling = args.pooling
        self.dropout_rate = args.dropout_rate
        self.behavior_integration_mode = args.behavior_integration_mode
        self.concat_behavior = getattr(args, "concat_behavior", False)
        self.intermediate_bn = getattr(args, "intermediate_bn", False)
        self.two_streams = getattr(args, "two_streams", False)

        # --- Initialize other attributes ---
        self.bias = bias
        self.stride = stride
        self.batch_norm_scale = batch_norm_scale
        self.independent_bn_bias = independent_bn_bias
        self.momentum = momentum
        self.spatial_dilation = spatial_dilation
        self.temporal_dilation = temporal_dilation
        self.nonlinearities = {
            "elu": torch.nn.ELU,
            "softplus": torch.nn.Softplus,
            "relu": torch.nn.ReLU,
            "adaptive_elu": AdaptiveELU,
        }

        # --- Configure regularizers ---
        reg_config = (
            dict(padding=laplace_padding, kernel=args.spatial_input_kernel)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weight_regularizer = getattr(regularizers, input_regularizer)(
            **reg_config
        )
        self.temporal_regularizer = DepthLaplaceL21d()

        # --- Standardize hyperparameter formats ---
        self.hidden_channels = check_hyperparam_for_layers(
            self.hidden_channels, self.layers
        )
        self.output_channels = self.hidden_channels[-1]
        self.hidden_temporal_dilation = check_hyperparam_for_layers(
            hidden_temporal_dilation, self.layers - 1
        )
        self.hidden_spatial_dilation = check_hyperparam_for_layers(
            hidden_spatial_dilation, self.layers - 1
        )

        if isinstance(self.spatial_input_kernel, int):
            self.spatial_input_kernel = (self.spatial_input_kernel,) * 2
        if isinstance(self.spatial_hidden_kernel, int):
            self.spatial_hidden_kernel = (self.spatial_hidden_kernel,) * 2
        if isinstance(self.spatial_hidden_kernel, (tuple, list)):
            self.spatial_hidden_kernel = [self.spatial_hidden_kernel] * (
                self.layers - 1
            )
            self.temporal_hidden_kernel = [self.temporal_hidden_kernel] * (
                self.layers - 1
            )
        if isinstance(self.stride, int):
            self.stride = [self.stride] * self.layers

        # --- Configure layer stacking and behavior modulation ---
        self.modulator_layers = getattr(args, "modulator_layers", list(range(self.layers)))
        if args.stack is None:
            self.stack = range(self.layers)
        else:
            self.stack = (
                [*range(self.layers)[args.stack :]]
                if isinstance(args.stack, int)
                else args.stack
            )

        # --- Build network layers ---
        self.features = torch.nn.Sequential()
        self._build_layers(final_nonlin, hidden_nonlinearities, x_shift, y_shift, args)

        # --- Initialize residual connections if specified ---
        if self.use_residuals:
            self.residual_blocks = torch.nn.ModuleList()
            for i in range(1, len(self.hidden_channels)):
                in_channels = self.hidden_channels[i - 1]
                out_channels = self.hidden_channels[i]
                self.residual_blocks.append(
                    torch.nn.Conv3d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0
                    )
                )

        self.initialize(cuda=(args.device == "cuda"))
        self.initialize_weights()

    def _build_layers(
        self, final_nonlin, hidden_nonlinearities, x_shift, y_shift, args
    ):
        """Constructs the factorized convolutional layers of the core."""
        # --- First Layer ---
        layer = OrderedDict()
        spat_pad = (
            (0, self.spatial_input_kernel[0] // 2, self.spatial_input_kernel[1] // 2)
            if self.padding
            else 0
        )
        layer["conv_spatial"] = torch.nn.Conv3d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels[0],
            kernel_size=(1,) + self.spatial_input_kernel,
            stride=(1, self.stride[0], self.stride[0]),
            bias=self.bias,
            dilation=(1, self.spatial_dilation, self.spatial_dilation),
            padding=spat_pad,
        )
        if self.intermediate_bn:
            self.add_bn_layer(
                layer, self.hidden_channels[0], name="intermediate_norm"
            )
        layer["conv_temporal"] = torch.nn.Conv3d(
            self.hidden_channels[0],
            self.hidden_channels[0],
            kernel_size=(self.temporal_input_kernel, 1, 1),
            bias=self.bias,
            dilation=(self.temporal_dilation, 1, 1),
            padding=(self.temporal_input_kernel - 1, 0, 0),
        )
        self.add_bn_layer(layer, self.hidden_channels[0], name="norm")

        if self.layers > 1 or final_nonlin:
            if hidden_nonlinearities == "adaptive_elu":
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](
                    xshift=x_shift, yshift=y_shift
                )
            else:
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()

        if self.pooling and 0 in args.pooling_layers:
            layer["pooling"] = torch.nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2)
            )
        self.add_dropout_layer(layer, self.dropout_rate)
        self.features.add_module("layer0", torch.nn.Sequential(layer))

        # --- Hidden Layers ---
        for l in range(self.layers - 1):
            layer = OrderedDict()
            spat_pad = (
                (
                    0,
                    self.spatial_hidden_kernel[l][0] // 2,
                    self.spatial_hidden_kernel[l][1] // 2,
                )
                if self.padding
                else 0
            )
            layer[f"conv_spatial_{l+1}"] = torch.nn.Conv3d(
                in_channels=self.hidden_channels[l],
                out_channels=self.hidden_channels[l + 1],
                kernel_size=(1,) + self.spatial_hidden_kernel[l],
                stride=(1, self.stride[l], self.stride[l]),
                bias=self.bias,
                dilation=(1, self.hidden_spatial_dilation[l], self.hidden_spatial_dilation[l]),
                padding=spat_pad,
            )
            if self.intermediate_bn:
                self.add_bn_layer(
                    layer, self.hidden_channels[l + 1], name="intermediate_norm"
                )
            layer[f"conv_temporal_{l+1}"] = torch.nn.Conv3d(
                self.hidden_channels[l + 1],
                self.hidden_channels[l + 1],
                kernel_size=(self.temporal_hidden_kernel[l], 1, 1),
                bias=self.bias,
                dilation=(self.hidden_temporal_dilation[l], 1, 1),
                padding=(self.temporal_hidden_kernel[l] - 1, 0, 0),
            )
            self.add_bn_layer(layer, self.hidden_channels[l + 1], name="norm")

            if final_nonlin or l < self.layers - 2:
                if hidden_nonlinearities == "adaptive_elu":
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](
                        x_shift=x_shift, y_shift=y_shift
                    )
                else:
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()

            if self.pooling and l + 1 in args.pooling_layers:
                layer[f"pooling_{l+1}"] = torch.nn.MaxPool3d(
                    kernel_size=(1, 2, 2), stride=(1, 2, 2)
                )
            self.add_dropout_layer(layer, self.dropout_rate)
            self.features.add_module(f"layer{l + 1}", torch.nn.Sequential(layer))

    def initialize_weights(self):
        """Initializes the weights of all convolutional layers."""
        for layer in self.features:
            if isinstance(layer, torch.nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, torch.nn.Conv3d):
                        torch.nn.init.kaiming_uniform_(sublayer.weight)
                        if sublayer.bias is not None:
                            sublayer.bias.data.fill_(0.01)

    def integrate_behavior(self, x, behavior, how="mul"):
        """
        Integrates behavior information into the feature map.

        Args:
            x (torch.Tensor): The input feature map.
            behavior (torch.Tensor): The behavior tensor.
            how (str): Integration method, 'mul' for multiplication or 'sum' for addition.

        Returns:
            torch.Tensor: The modulated feature map.
        """
        if x.shape[1] != behavior.shape[1]:
            raise ValueError(
                "Behavior tensor must have the same number of channels as the input."
            )
        if how == "mul":
            return x * behavior
        elif how == "sum":
            return x + behavior
        raise ValueError(f"Unknown behavior integration mode: {how}")

    def forward(self, x, behavior=None):
        """
        Computes the forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch, channels, T, H, W).
            behavior (list of torch.Tensor, optional): A list of behavior tensors,
                one for each layer.

        Returns:
            torch.Tensor: The output feature map.
        """
        num_frames = x.shape[2]
        if behavior is not None and not self.concat_behavior:
            if len(behavior) != self.layers:
                raise ValueError(
                    "Length of behavior list must equal the number of layers."
                )

        layer_outputs = []
        for i, layer in enumerate(self.features):
            if behavior is not None and not self.concat_behavior and i in self.modulator_layers:
                # Apply layer modules sequentially and integrate behavior after spatial conv
                for name, module in layer.named_children():
                    x = module(x)[:, :, :num_frames, :, :]
                    if "conv_spatial" in name:
                        x = self.integrate_behavior(
                            x, behavior[i], how=self.behavior_integration_mode
                        )
            else:
                x = layer(x)[:, :, :num_frames, :, :]

            if self.use_residuals and i > 0:
                residual = self.residual_blocks[i - 1](x)
                x = x + residual

            layer_outputs.append(x)
            if self.two_streams and i == self.layers - 3:
                x = layer_outputs[-2]  # Use output from the second to last layer

        # Stack outputs from specified layers
        stacked_output = torch.cat([layer_outputs[ind] for ind in self.stack], dim=1)

        if behavior is not None and self.concat_behavior:
            # Concatenate behavior to the final output
            b = behavior[self.modulator_layers[0]]
            b = b.expand(-1, -1, -1, stacked_output.shape[3], stacked_output.shape[4])
            stacked_output = torch.cat([stacked_output, b], dim=1)

        return stacked_output

    def laplace_spatial(self):
        """
        Computes the spatial Laplace regularizer for the first layer.

        Returns:
            torch.Tensor: The spatial regularization term.
        """
        weight = self.features[0].conv_spatial.weight[:, :, 0, :, :]
        return self._input_weight_regularizer(weight)

    def laplace_temporal(self):
        """
        Computes the temporal Laplace regularizer for the first layer.

        Returns:
            torch.Tensor: The temporal regularization term.
        """
        weight = self.features[0].conv_temporal.weight[:, :, :, 0, 0]
        return self.temporal_regularizer(weight)

    def regularizer(self):
        """
        Computes the total regularization term for the input weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the spatial and
                temporal regularization terms.
        """
        return (
            self.gamma_input_spatial * self.laplace_spatial(),
            self.gamma_input_temporal * self.laplace_temporal(),
        )

    def get_kernels(self):
        """Returns a list of all effective kernel sizes used in the core."""
        kernels = [(self.temporal_input_kernel,) + self.spatial_input_kernel]
        kernels.extend(
            (t_kern,) + s_kern
            for t_kern, s_kern in zip(
                self.temporal_hidden_kernel, self.spatial_hidden_kernel
            )
        )
        return kernels

    def add_bn_layer(self, layer, hidden_channels, name="norm"):
        """
        Adds a batch normalization layer to the layer dictionary.

        Args:
            layer (OrderedDict): The dictionary of layers for a Sequential block.
            hidden_channels (int): The number of channels for the BatchNorm layer.
            name (str): The key for the layer in the dictionary.
        """
        if self.batch_norm:
            if self.independent_bn_bias:
                layer[name] = torch.nn.BatchNorm3d(
                    hidden_channels, momentum=self.momentum
                )
            else:
                affine = self.bias and self.batch_norm_scale
                layer[name] = torch.nn.BatchNorm3d(
                    hidden_channels, momentum=self.momentum, affine=affine
                )
                if self.bias and not self.batch_norm_scale:
                    layer["bias"] = Bias3DLayer(hidden_channels)
                elif self.batch_norm_scale:
                    layer["scale"] = Scale3DLayer(hidden_channels)

    def add_dropout_layer(self, layer, dropout_rate=0.5):
        """
        Adds a dropout layer to the layer dictionary.

        Args:
            layer (OrderedDict): The dictionary of layers for a Sequential block.
            dropout_rate (float): The dropout probability.
        """
        if dropout_rate > 0:
            layer["dropout"] = torch.nn.Dropout3d(p=dropout_rate)

    @property
    def is_2d(self):
        """Returns False as this is a 3D core."""
        return False
