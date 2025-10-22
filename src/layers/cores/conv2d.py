import warnings
from collections import OrderedDict
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable

import torch
from torch import nn

from neuralpredictors import regularizers
from neuralpredictors.layers import activations
from neuralpredictors.layers.activations import AdaptiveELU
from neuralpredictors.layers.affine import Bias2DLayer, Scale2DLayer
from neuralpredictors.layers.attention import AttentionConv
from neuralpredictors.layers.conv import DepthSeparableConv2d
from neuralpredictors.layers.cores.base import Core

from src.layers.cores.core import register


class Core2d(Core):
    """
    A base class for 2D PyTorch cores. Provides initialization and device placement.
    """

    def initialize(self, cuda=False):
        """
        Initializes the weights of the core and moves it to the specified device.

        Args:
            cuda (bool): If True, moves the core to the GPU.
        """
        self.apply(self.init_conv)
        self.put_to_cuda(cuda=cuda)

    def put_to_cuda(self, cuda):
        """
        Moves the core to the GPU if specified.

        Args:
            cuda (bool): If True, moves the core to the GPU.
        """
        if cuda:
            self.cuda()

    @staticmethod
    def init_conv(m):
        """
        Initializes weights of a convolutional layer.

        Args:
            m (nn.Module): The module to initialize. If it's a Conv2d layer,
                           weights are initialized with Xavier normal, and
                           biases are set to zero.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)


@register("stackedconv2d")
class Stacked2dCore(Core2d, nn.Module):
    """
    A configurable core consisting of a stack of 2D convolutional layers.

    This core allows for flexible construction of various architectures,
    supporting features like skip connections, batch normalization, depth-separable
    convolutions, and various regularization schemes.
    """

    def __init__(
        self,
        args,
        skip=0,
        stride=1,
        input_stride=1,
        final_nonlinearity=True,
        elu_shift=(0, 0),
        bias=True,
        momentum=0.1,
        batch_norm_scale=True,
        final_batchnorm_scale=True,
        independent_bn_bias=True,
        hidden_dilation=1,
        laplace_padding=0,
        input_regularizer="LaplaceL2",
        use_avg_reg=True,
        depth_separable=False,
        attention_conv=False,
        linear=False,
        nonlinearity_type="AdaptiveELU",
        nonlinearity_config=None,
    ):
        """
        Initializes the Stacked2dCore.

        Args:
            args (object): An object containing model configuration. Expected attributes:
                core_input_channels (int): Number of input channels.
                hidden_channels (int or list): Number of hidden channels per layer.
                input_kernel (int): Kernel size of the first layer.
                hidden_kernel (int): Kernel size of subsequent layers.
                layers (int): Number of layers in the core.
                gamma_input (float): Regularization factor for input weights.
                gamma_hidden (float): Regularization factor for hidden weights (group sparsity).
                batch_norm (bool): Whether to use batch normalization.
                padding (bool): Whether to apply zero-padding to convolutions.
                stack (int or list, optional): Layers to stack for the output.
                    Defaults to all layers.
                device (str): The device to run the model on ('cuda' or 'cpu').
            skip (int): Adds a skip connection every `skip` layers.
            stride (int): Stride for hidden convolutional layers.
            input_stride (int): Stride for the first convolutional layer.
            final_nonlinearity (bool): If True, adds a nonlinearity to the final layer.
            elu_shift (tuple): A tuple (x_shift, y_shift) for the AdaptiveELU activation.
            bias (bool): If True, adds a bias to convolutional layers.
            momentum (float): Momentum for batch normalization.
            batch_norm_scale (bool): If True, batch norm layers will have a learnable scale parameter.
            final_batchnorm_scale (bool): If True, the final batch norm layer will have a learnable scale.
            independent_bn_bias (bool): If True, uses a simplified logic for batch norm
                that is independent of `bias` and `batch_norm_scale` flags.
            hidden_dilation (int): Dilation for hidden convolutional layers.
            laplace_padding (int): Padding for the Laplace regularizer.
            input_regularizer (str): The type of regularizer for the input layer.
            use_avg_reg (bool): If True, averages the regularization loss; otherwise, sums it.
            depth_separable (bool): If True, uses depth-separable convolutions for hidden layers.
            attention_conv (bool): If True, uses attention-based convolutions for hidden layers.
            linear (bool): If True, removes all nonlinearities.
            nonlinearity_type (str): The type of activation function to use.
            nonlinearity_config (dict, optional): Configuration for the activation function.
        """
        super().__init__()

        if depth_separable and attention_conv:
            raise ValueError(
                "depth_separable and attention_conv can not both be true"
            )

        if independent_bn_bias:
            if not bias or not batch_norm_scale or not final_batchnorm_scale:
                warnings.warn(
                    "The default of `independent_bn_bias=True` will ignore "
                    "the kwargs `bias`, `batch_norm_scale`, and "
                    "`final_batchnorm_scale` when initializing batchnorm. "
                    "Set `independent_bn_bias=False` to use these arguments."
                )

        self.batch_norm = args.batch_norm
        self.final_batchnorm_scale = final_batchnorm_scale
        self.bias = bias
        self.independent_bn_bias = independent_bn_bias
        self.batch_norm_scale = batch_norm_scale
        self.num_layers = args.layers
        self.gamma_input = args.gamma_input
        self.gamma_hidden = args.gamma_hidden
        self.input_channels = args.core_input_channels

        if isinstance(args.hidden_channels, Iterable) and skip > 1:
            raise NotImplementedError(
                "Passing a list of hidden channels and `skip > 1` is not supported."
            )
        self.hidden_channels = (
            args.hidden_channels
            if isinstance(args.hidden_channels, Iterable)
            else [args.hidden_channels] * self.num_layers
        )
        self.skip = skip
        self.stride = stride
        self.input_stride = input_stride
        self.use_avg_reg = use_avg_reg
        if use_avg_reg:
            warnings.warn(
                "The averaged value of regularizer will be used.", UserWarning
            )
        self.hidden_padding = args.padding
        self.input_kern = args.input_kernel
        self.hidden_kern = args.hidden_kernel
        self.laplace_padding = laplace_padding
        self.hidden_dilation = hidden_dilation
        self.final_nonlinearity = final_nonlinearity
        self.elu_xshift, self.elu_yshift = elu_shift
        self.momentum = momentum
        self.padding = args.padding
        self.linear = linear

        if args.stack is None:
            self.stack = range(self.num_layers)
        else:
            self.stack = (
                [*range(self.num_layers)[args.stack :]]
                if isinstance(args.stack, int)
                else args.stack
            )

        # Configure convolution type
        if depth_separable:
            self.conv_layer_name = "ds_conv"
            self.ConvLayer = DepthSeparableConv2d
            self.ignore_group_sparsity = True
        elif attention_conv:
            self.conv_layer_name = "attention_conv"
            self.ConvLayer = self.AttentionConvWrapper
            self.ignore_group_sparsity = True
        else:
            self.conv_layer_name = "conv"
            self.ConvLayer = nn.Conv2d
            self.ignore_group_sparsity = False

        if self.ignore_group_sparsity and self.gamma_hidden > 0:
            warnings.warn(
                "Group sparsity cannot be applied for the chosen conv type. "
                "gamma_hidden will be ignored."
            )

        # Configure regularizer
        reg_config = (
            dict(padding=laplace_padding, kernel=args.input_kernel)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[
            input_regularizer
        ](**reg_config)

        self.activation_fn = activations.__dict__[nonlinearity_type]
        self.activation_config = (
            nonlinearity_config if nonlinearity_config is not None else {}
        )

        self.set_batchnorm_type()
        self.features = nn.Sequential()
        self.add_first_layer()
        self.add_subsequent_layers()
        self.initialize(cuda=args.device == "cuda")

    def set_batchnorm_type(self):
        """Sets the classes for batchnorm and affine layers."""
        self.batchnorm_layer_cls = nn.BatchNorm2d
        self.bias_layer_cls = Bias2DLayer
        self.scale_layer_cls = Scale2DLayer

    def penultimate_layer_built(self):
        """Checks if the second to last layer has been built."""
        return len(self.features) == self.num_layers - 1

    def add_bn_layer(self, layer, hidden_channels):
        """
        Adds a batch normalization layer and optional affine layers to a layer dict.

        The behavior depends on `self.independent_bn_bias` and other flags.

        Args:
            layer (OrderedDict): The dictionary of layers to add to.
            hidden_channels (int): The number of channels for the batch norm layer.
        """
        if not self.batch_norm:
            return

        if self.independent_bn_bias:
            layer["norm"] = self.batchnorm_layer_cls(
                hidden_channels, momentum=self.momentum
            )
        else:
            # Complex logic for backward compatibility
            is_final_layer = self.penultimate_layer_built()
            use_affine = self.bias and self.batch_norm_scale
            if is_final_layer:
                use_affine &= self.final_batchnorm_scale

            layer["norm"] = self.batchnorm_layer_cls(
                hidden_channels, momentum=self.momentum, affine=use_affine
            )

            if self.bias and (
                not self.batch_norm_scale
                or (is_final_layer and not self.final_batchnorm_scale)
            ):
                layer["bias"] = self.bias_layer_cls(hidden_channels)
            elif self.batch_norm_scale and not (
                is_final_layer and not self.final_batchnorm_scale
            ):
                layer["scale"] = self.scale_layer_cls(hidden_channels)

    def add_activation(self, layer):
        """
        Adds a nonlinearity to a layer dict.

        Args:
            layer (OrderedDict): The dictionary of layers to add to.
        """
        if self.linear:
            return

        is_final_layer = self.penultimate_layer_built()
        if not is_final_layer or self.final_nonlinearity:
            if self.activation_fn == AdaptiveELU:
                layer["nonlin"] = AdaptiveELU(self.elu_xshift, self.elu_yshift)
            else:
                layer["nonlin"] = self.activation_fn(**self.activation_config)

    def add_first_layer(self):
        """Builds and adds the first layer of the core."""
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            self.input_channels,
            self.hidden_channels[0],
            self.input_kern,
            stride=self.input_stride,
            padding=self.input_kern // 2 if self.padding else 0,
            bias=self.bias and not self.batch_norm,
        )
        self.add_bn_layer(layer, self.hidden_channels[0])
        self.add_activation(layer)
        self.features.add_module("layer0", nn.Sequential(layer))

    def add_subsequent_layers(self):
        """Builds and adds all layers after the first one."""
        if not isinstance(self.hidden_kern, Iterable):
            self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

        for l in range(1, self.num_layers):
            layer = OrderedDict()
            if self.hidden_padding != 0:
                # Overwrite padding to preserve feature map size for dilated convs
                self.hidden_padding = (
                    (self.hidden_kern[l - 1] - 1) * self.hidden_dilation + 1
                ) // 2

            in_channels = (
                self.hidden_channels[l - 1]
                if not self.skip > 1
                else min(self.skip, l) * self.hidden_channels[0]
            )

            layer[self.conv_layer_name] = self.ConvLayer(
                in_channels=in_channels,
                out_channels=self.hidden_channels[l],
                kernel_size=self.hidden_kern[l - 1],
                stride=self.stride,
                padding=self.hidden_padding,
                dilation=self.hidden_dilation,
                bias=self.bias,
            )
            self.add_bn_layer(layer, self.hidden_channels[l])
            self.add_activation(layer)
            self.features.add_module(f"layer{l}", nn.Sequential(layer))

    class AttentionConvWrapper(AttentionConv):
        """
        A wrapper to make AttentionConv compatible with nn.Conv2d arguments.
        """

        def __init__(self, dilation=None, **kwargs):
            """
            Initializes the wrapper.

            Args:
                dilation: This argument is accepted and ignored for compatibility.
                **kwargs: Arguments passed to the parent AttentionConv class.
            """
            super().__init__(**kwargs)

    def forward(self, x):
        """
        Computes the forward pass through the core.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor, which is a concatenation of
                          feature maps from the layers specified in `self.stack`.
        """
        outputs = []
        for l, layer in enumerate(self.features):
            if l > 0 and self.skip > 1:
                # Concatenate outputs of previous `skip` layers as input
                skip_layers = outputs[-min(self.skip, l) :]
                x = torch.cat(skip_layers, dim=1)
            x = layer(x)
            outputs.append(x)

        return torch.cat([outputs[i] for i in self.stack], dim=1)

    def laplace(self):
        """
        Computes the Laplace regularization for the first layer filters.

        Returns:
            torch.Tensor: The regularization term.
        """
        return self._input_weights_regularizer(
            self.features[0].conv.weight, avg=self.use_avg_reg
        )

    def group_sparsity(self):
        """
        Computes the group sparsity regularization for hidden layer filters.

        Returns:
            torch.Tensor: The regularization term.
        """
        if self.ignore_group_sparsity:
            return 0.0

        reg_term = 0.0
        for feature_layer in self.features[1:]:
            conv_weight = feature_layer.conv.weight
            reg_term += (
                conv_weight.pow(2)
                .sum(3, keepdim=True)
                .sum(2, keepdim=True)
                .sqrt()
                .mean()
            )
        return reg_term / ((self.num_layers - 1) if self.num_layers > 1 else 1)

    def regularizer(self):
        """
        Computes the total regularization term for the core.

        Returns:
            torch.Tensor: The combined regularization loss.
        """
        return (
            self.group_sparsity() * self.gamma_hidden
            + self.gamma_input * self.laplace()
        )

    @property
    def out_channels(self):
        """
        The number of output channels, calculated as the number of layers
        multiplied by the number of channels in the last layer. Note that this
        may not be accurate if `self.stack` is customized or if layers have
        varying channel counts.
        """
        return len(self.features) * self.hidden_channels[-1]

    @property
    def is_2d(self):
        """
        Returns True, indicating that this is a 2D core.
        """
        return True
