"""
This module provides various encoder classes for neural network models,
including core, readout, shifter, and modulator components.
"""
import torch
import torch.nn as nn
from neuralpredictors.layers import activations
from torchvision.models import resnet50, ResNet50_Weights

from src.layers.activations import Exp, LearnableSoftplus
from src.layers.readouts import MultiReadoutBase, get_readout
from src.engine.utils import get_dims_for_loader_dict


class Encoder(nn.Module):
    """
    Wraps a core, readout, and optional shifter/modulator into a single model.

    The output is a positive value that can be interpreted as a firing rate,
    suitable for a Poisson loss function.
    """

    def __init__(
        self,
        core,
        readout,
        gru_module=None,
        shifter=None,
        modulator=None,
        elu_offset=0.0,
        nonlinearity_type="elu",
        nonlinearity_config=None,
        detach_core: bool = False,
    ):
        """
        Initializes the Encoder module.

        Args:
            core (nn.Module): The feature extraction core of the model.
            readout (nn.ModuleDict): A dictionary of readout modules, one for
                each data_key.
            gru_module (nn.Module, optional): A recurrent module (e.g., GRU)
                to be applied after the core. Defaults to None.
            shifter (nn.ModuleDict, optional): A shifter network to adjust
                readout based on pupil position. Defaults to None.
            modulator (nn.Module, optional): A modulator network to incorporate
                behavioral data. Defaults to None.
            elu_offset (float, optional): Offset for the ELU nonlinearity.
                Defaults to 0.0.
            nonlinearity_type (str, optional): The final nonlinearity function.
                Defaults to "elu".
            nonlinearity_config (dict, optional): Configuration for the
                nonlinearity function. Defaults to None.
            detach_core (bool, optional): If True, detaches the core from the
                computation graph, effectively freezing its weights.
                Defaults to False.
        """
        super().__init__()
        self.core = core
        self.gru_module = gru_module
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.offset = elu_offset
        self.detach_core = detach_core

        if nonlinearity_type == "elu":
            self.nonlinearity_fn = nn.ELU()
        elif nonlinearity_type == "identity":
            self.nonlinearity_fn = nn.Identity()
        elif nonlinearity_type == "exp":
            self.nonlinearity_fn = Exp()
        elif nonlinearity_type == "learnablesoftplus":
            self.nonlinearity_fn = LearnableSoftplus()
        else:
            nonlinearity_config = nonlinearity_config or {}
            self.nonlinearity_fn = activations.__dict__[nonlinearity_type](
                **nonlinearity_config
            )
        self.nonlinearity_type = nonlinearity_type
        self.twoD_core = core.is_2d

        if self.detach_core:
            for param in self.core.parameters():
                param.requires_grad = False

    def forward(
        self,
        inputs,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        **kwargs,
    ):
        """
        Performs a forward pass through the encoder.

        Args:
            inputs (torch.Tensor): The input tensor.
            data_key (str, optional): The key for the specific readout to use.
                Defaults to None.
            behavior (torch.Tensor, optional): Behavioral data for the
                modulator. Defaults to None.
            pupil_center (torch.Tensor, optional): Pupil center data for the
                shifter. Defaults to None.
            trial_idx (torch.Tensor, optional): Trial indices. Defaults to None.
            **kwargs: Additional arguments for the readout.

        Returns:
            torch.Tensor: The predicted neural activity.
        """
        batch_size, _, time_points, _, _ = inputs.shape

        # Reshape input for 2D cores by collapsing batch and time dimensions
        if self.twoD_core:
            inputs = torch.transpose(inputs, 1, 2)
            inputs = inputs.reshape((-1,) + inputs.size()[2:])

        if self.modulator:
            behavior = self.modulator(behavior)
        else:
            behavior = None

        x = self.core(inputs, behavior=behavior)

        if self.detach_core:
            x = x.detach()

        if self.gru_module:
            x = self.gru_module(x)
            if isinstance(x, list):
                x = x[-1]

        if not self.twoD_core:
            x = torch.transpose(x, 1, 2)
            time_points = x.shape[1]

        # Reshape for readout: collapse batch and time dimensions
        x = x.reshape((-1,) + x.size()[2:])

        shift = None
        if self.shifter:
            if pupil_center is None:
                pupil_center = kwargs["pupil_center"]
            pupil_center = pupil_center[:, :, -time_points:]
            pupil_center = torch.transpose(pupil_center, 1, 2)
            pupil_center = pupil_center.reshape((-1,) + pupil_center.size()[2:])
            shift = self.shifter[data_key](pupil_center, trial_idx)

        x = self.readout(x, data_key=data_key, shift=shift, **kwargs)

        if self.nonlinearity_type == "elu":
            x = self.nonlinearity_fn(x + self.offset) + 1
        else:
            x = self.nonlinearity_fn(x)

        # Reshape back to (batch, time, neurons)
        x = x.reshape((batch_size, time_points) + x.size()[1:])
        return x

    def forward_all(
        self,
        inputs,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        selected_sessions=None,
        **kwargs,
    ):
        """
        Performs a forward pass for all readouts.

        Args:
            inputs (torch.Tensor): The input tensor.
            behavior (torch.Tensor, optional): Behavioral data. Defaults to None.
            pupil_center (torch.Tensor, optional): Pupil center data.
                Defaults to None.
            trial_idx (torch.Tensor, optional): Trial indices. Defaults to None.
            selected_sessions (list, optional): A list of data_keys to compute
                predictions for. If None, computes for all. Defaults to None.
            **kwargs: Additional arguments for the readout.

        Returns:
            dict[str, torch.Tensor]: A dictionary mapping data_key to the
                predicted neural activity.
        """
        batch_size, _, time_points, _, _ = inputs.shape

        if self.twoD_core:
            inputs = torch.transpose(inputs, 1, 2)
            inputs = inputs.reshape((-1,) + inputs.size()[2:])

        if self.modulator:
            behavior = self.modulator(behavior)

        x = self.core(inputs, behavior=behavior)

        if self.detach_core:
            x = x.detach()

        if self.gru_module:
            x = self.gru_module(x)
            if isinstance(x, list):
                x = x[-1]

        if not self.twoD_core:
            x = torch.transpose(x, 1, 2)
            time_points = x.shape[1]

        shift = None
        if self.shifter:
            if pupil_center is None:
                pupil_center = kwargs["pupil_center"]
            pupil_center = pupil_center[:, :, -time_points:]
            pupil_center = torch.transpose(pupil_center, 1, 2)
            pupil_center = pupil_center.reshape((-1,) + pupil_center.size()[2:])

        x = x.reshape((-1,) + x.size()[2:])

        output = {}
        for key in self.readout.keys():
            if selected_sessions is not None and key not in selected_sessions:
                continue
            if self.shifter:
                shift = self.shifter[key](pupil_center, trial_idx)
            x_out = self.readout[key](x, shift=shift, **kwargs)

            if self.nonlinearity_type == "elu":
                x_out = self.nonlinearity_fn(x_out + self.offset) + 1
            else:
                x_out = self.nonlinearity_fn(x_out)

            x_out = x_out.reshape((batch_size, time_points) + x_out.size()[1:])
            output[key] = x_out

        return output

    def regularizer(self, data_key=None, reduction="sum", average=None):
        """
        Computes the regularization term for the model.

        Args:
            data_key (str, optional): The key for which to compute the
                regularizer. Defaults to None.
            reduction (str, optional): The reduction method ('sum', 'mean').
                Defaults to "sum".
            average (bool, optional): Deprecated. Use reduction='mean'.

        Returns:
            torch.Tensor: The regularization value.
        """
        reg = 0 if self.detach_core else self.core.regularizer()
        reg += self.readout.regularizer(
            data_key=data_key, reduction=reduction, average=average
        )
        if self.shifter:
            reg += self.shifter.regularizer(data_key=data_key)
        if self.modulator:
            reg += self.modulator.regularizer()
        return reg


class TwoPartEncoder(nn.Module):
    """
    An Encoder that produces three outputs for a hurdle model: logits for the
    binary part, and predictions for the regression and Poisson parts.
    """

    def __init__(
        self,
        core,
        readout,
        gru_module=None,
        shifter=None,
        modulator=None,
        elu_offset=0.0,
        nonlinearity_type="elu",
        nonlinearity_config=None,
        detach_core: bool = False,
    ):
        """
        Initializes the TwoPartEncoder module.

        Args:
            core (nn.Module): The feature extraction core of the model.
            readout (nn.ModuleDict): A dictionary of readout modules.
            gru_module (nn.Module, optional): A recurrent module. Defaults to None.
            shifter (nn.ModuleDict, optional): A shifter network. Defaults to None.
            modulator (nn.ModuleDict, optional): A modulator network.
                Defaults to None.
            elu_offset (float, optional): Offset for the ELU nonlinearity.
                Defaults to 0.0.
            nonlinearity_type (str, optional): The final nonlinearity function.
                Defaults to "elu".
            nonlinearity_config (dict, optional): Configuration for the
                nonlinearity. Defaults to None.
            detach_core (bool, optional): If True, freezes the core weights.
                Defaults to False.
        """
        super().__init__()
        self.core = core
        self.gru_module = gru_module
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.offset = elu_offset
        self.detach_core = detach_core

        if nonlinearity_type == "elu":
            self.nonlinearity_fn = nn.ELU()
        elif nonlinearity_type == "identity":
            self.nonlinearity_fn = nn.Identity()
        else:
            nonlinearity_config = nonlinearity_config or {}
            self.nonlinearity_fn = activations.__dict__[nonlinearity_type](
                **nonlinearity_config
            )
        self.nonlinearity_type = nonlinearity_type
        self.twoD_core = core.is_2d

        if self.detach_core:
            for param in self.core.parameters():
                param.requires_grad = False

    def forward(
        self,
        inputs,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        **kwargs,
    ):
        """
        Performs a forward pass.

        Args:
            inputs (torch.Tensor): The input tensor.
            data_key (str, optional): The key for the specific readout.
                Defaults to None.
            behavior (torch.Tensor, optional): Behavioral data. Defaults to None.
            pupil_center (torch.Tensor, optional): Pupil center data.
                Defaults to None.
            trial_idx (torch.Tensor, optional): Trial indices. Defaults to None.
            **kwargs: Additional arguments for the readout.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
                logits for the binary part, predictions for the regression part,
                and predictions for the final Poisson part.
        """
        batch_size, _, time_points, _, _ = inputs.shape

        if self.twoD_core:
            inputs = torch.transpose(inputs, 1, 2)
            inputs = inputs.reshape((-1,) + inputs.size()[2:])

        x = self.core(inputs)

        if self.detach_core:
            x = x.detach()

        if self.gru_module:
            x = self.gru_module(x)
            if isinstance(x, list):
                x = x[-1]

        if not self.twoD_core:
            x = torch.transpose(x, 1, 2)
            time_points = x.shape[1]

        shift = None
        if self.shifter:
            if pupil_center is None:
                pupil_center = kwargs["pupil_center"]
            pupil_center = pupil_center[:, :, -time_points:]
            pupil_center = torch.transpose(pupil_center, 1, 2)
            pupil_center = pupil_center.reshape((-1,) + pupil_center.size()[2:])
            shift = self.shifter[data_key](pupil_center, trial_idx)

        x = x.reshape((-1,) + x.size()[2:])

        logits_preds, regression_preds, poisson_preds = self.readout(
            x, data_key=data_key, shift=shift, **kwargs
        )

        if self.modulator:
            if behavior is None:
                behavior = kwargs["behavior"]
            regression_preds = self.modulator[data_key](
                regression_preds, behavior=behavior
            )

        if self.nonlinearity_type == "elu":
            regression_preds = self.nonlinearity_fn(regression_preds + self.offset) + 1
        else:
            regression_preds = self.nonlinearity_fn(regression_preds)

        logits_preds = logits_preds.reshape(
            (batch_size, time_points) + logits_preds.size()[1:]
        )
        regression_preds = regression_preds.reshape(
            (batch_size, time_points) + regression_preds.size()[1:]
        )
        poisson_preds = poisson_preds.reshape(
            (batch_size, time_points) + poisson_preds.size()[1:]
        )
        return logits_preds, regression_preds, poisson_preds

    def regularizer(self, data_key=None, reduction="sum", average=None):
        """
        Computes the regularization term for the model.

        Args:
            data_key (str, optional): The key for which to compute the
                regularizer. Defaults to None.
            reduction (str, optional): The reduction method ('sum', 'mean').
                Defaults to "sum".
            average (bool, optional): Deprecated. Use reduction='mean'.

        Returns:
            torch.Tensor: The regularization value.
        """
        reg = 0 if self.detach_core else self.core.regularizer()
        reg += self.readout.regularizer(
            data_key=data_key, reduction=reduction, average=average
        )
        if self.shifter:
            reg += self.shifter.regularizer(data_key=data_key)
        if self.modulator:
            reg += self.modulator.regularizer(data_key=data_key)
        return reg


class ResNet50Encoder(nn.Module):
    """
    An encoder using a pre-trained ResNet50 as the core.

    Activations from a specified intermediate layer are extracted using hooks
    and passed to a readout module.
    """

    def __init__(
        self,
        args,
        dataloaders: dict,
        layer_hook: str,
        nonlinearity_type="elu",
        elu_offset=0.0,
        nonlinearity_config=None,
        freeze_core=True,
    ):
        """
        Initializes the ResNet50Encoder.

        Args:
            args: A configuration object with model parameters.
            dataloaders (dict): Dataloaders for inferring shapes.
            layer_hook (str): The name of the ResNet50 layer to hook into for
                feature extraction.
            nonlinearity_type (str, optional): Final nonlinearity. Defaults to "elu".
            elu_offset (float, optional): Offset for the ELU. Defaults to 0.0.
            nonlinearity_config (dict, optional): Config for the nonlinearity.
                Defaults to None.
            freeze_core (bool, optional): If True, freezes the ResNet50 core.
                Defaults to True.
        """
        super().__init__()

        if "train" in dataloaders:
            dataloaders = dataloaders["train"]

        # Core
        self.core = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.layer_hook = layer_hook

        valid_layers = {name for name, _ in self.core.named_modules()}
        if self.layer_hook not in valid_layers:
            raise ValueError(
                f"Invalid layer name: {self.layer_hook}. "
                f"Valid layers are: {valid_layers}"
            )

        self.preprocess = ResNet50_Weights.DEFAULT.transforms(antialias=True)
        self.layer_hook_output = self.create_hook(self.core, self.layer_hook)

        # Readout
        dims_per_layer = self.output_dims_per_layer(self.core, self.layer_hook)
        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        input_shape_dict = {k: dims_per_layer for k in session_shape_dict.keys()}
        n_neurons_dict = {
            k: v["responses"][1] for k, v in session_shape_dict.items()
        }
        mean_activity_dict = {k: torch.zeros(v) for k, v in n_neurons_dict.items()}
        self.readout = MultiReadoutBase(
            args=args,
            base_readout=get_readout(args),
            in_shape_dict=input_shape_dict,
            n_neurons_dict=n_neurons_dict,
            mean_activity_dict=mean_activity_dict,
        )

        # Nonlinearity
        if nonlinearity_type == "elu":
            self.nonlinearity_fn = nn.ELU()
        elif nonlinearity_type == "identity":
            self.nonlinearity_fn = nn.Identity()
        elif nonlinearity_type == "exp":
            self.nonlinearity_fn = Exp()
        elif nonlinearity_type == "learnablesoftplus":
            self.nonlinearity_fn = LearnableSoftplus()
        else:
            nonlinearity_config = nonlinearity_config or {}
            self.nonlinearity_fn = activations.__dict__[nonlinearity_type](
                **nonlinearity_config
            )
        self.nonlinearity_type = nonlinearity_type
        self.offset = elu_offset

        self.freeze_core(freeze_core)

    def output_dims_per_layer(self, model, layer_to_hook):
        """
        Returns the output shape of the layer to hook.

        Args:
            model (torch.nn.Module): The model to inspect.
            layer_to_hook (str): Name of the layer to attach a hook to.

        Returns:
            torch.Size: The output shape of the hooked layer.
        """
        input_tensor = torch.randn(1, 3, 224, 224)
        temporary_hooks = self.create_hook(model, layer_to_hook)
        model(input_tensor)
        output_shape = temporary_hooks["outputs"][layer_to_hook].shape[1:]
        self.clear_hooks(temporary_hooks["handles"])
        return output_shape

    @staticmethod
    def create_hook(model, layer_to_hook):
        """
        Registers a forward hook on a specified layer of a model.

        Args:
            model (torch.nn.Module): The model to which the hook is applied.
            layer_to_hook (str): Name of the layer to attach the hook to.

        Returns:
            dict: A dictionary containing the hook's handle and output dict.
        """
        outputs = {}
        handles = []

        def hook_fn(module, input, output):
            outputs[layer_to_hook] = output

        layer = dict(model.named_modules())[layer_to_hook]
        handle = layer.register_forward_hook(hook_fn)
        handles.append(handle)

        return {"outputs": outputs, "handles": handles}

    @staticmethod
    def clear_hooks(handles):
        """
        Removes all hooks using their handles.

        Args:
            handles (list): List of hook handles to remove.
        """
        for handle in handles:
            handle.remove()

    def freeze_core(self, freeze=True):
        """Freezes or unfreezes the core parameters."""
        for param in self.core.parameters():
            param.requires_grad = not freeze

    def reshape_and_preprocess(self, inputs):
        """Reshapes and preprocesses the input tensor for ResNet."""
        batch_size, _, frames, _, _ = inputs.shape
        inputs = torch.transpose(inputs, 1, 2)
        inputs = inputs.reshape((-1,) + inputs.size()[2:])
        inputs = self.preprocess(inputs)
        return inputs, batch_size, frames

    def forward(self, inputs, data_key=None, **kwargs):
        """Performs a forward pass for a single data_key."""
        inputs, batch_size, frames = self.reshape_and_preprocess(inputs)
        self.core(inputs)
        layer_output = self.layer_hook_output["outputs"].get(self.layer_hook)
        if layer_output is None:
            raise ValueError(f"Output for layer {self.layer_hook} was not captured.")

        output = self.readout(layer_output, data_key=data_key)
        if self.nonlinearity_type == "elu":
            output = self.nonlinearity_fn(output + self.offset) + 1
        else:
            output = self.nonlinearity_fn(output)

        output = output.reshape((batch_size, frames) + output.size()[1:])
        self.layer_hook_output["outputs"].clear()
        return output

    def forward_all(self, inputs, to_cpu=False, **kwargs):
        """Performs a forward pass for all readouts."""
        inputs, batch_size, frames = self.reshape_and_preprocess(inputs)
        self.core(inputs)
        layer_output = self.layer_hook_output["outputs"].get(self.layer_hook)
        if layer_output is None:
            raise ValueError(f"Output for layer {self.layer_hook} was not captured.")

        output = {}
        for key in self.readout.keys():
            x_out = self.readout[key](layer_output, **kwargs)
            if self.nonlinearity_type == "elu":
                x_out = self.nonlinearity_fn(x_out + self.offset) + 1
            else:
                x_out = self.nonlinearity_fn(x_out)
            x_out = x_out.reshape((batch_size, frames) + x_out.size()[1:])
            if to_cpu:
                x_out = x_out.cpu()
            output[key] = x_out

        self.layer_hook_output["outputs"].clear()
        return output
