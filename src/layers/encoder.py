import torch
import torch.nn as nn
from neuralpredictors.layers import activations

from src.layers.activations import Exp, LearnableSoftplus

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