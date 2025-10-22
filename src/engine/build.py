from operator import itemgetter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from neuralpredictors.utils import get_module_output
from torch import nn
from torch.utils.data import DataLoader

from src.layers.cores import get_core
from src.layers.encoder import Encoder
from src.layers.modulators import StackedMLPBehavior
from src.layers.readouts import MultiReadoutBase, get_readout
from src.layers.rnn_modules import GRU_Module
from src.layers.shifters import MLPShifter
from src.engine.utils import (
    brain_area_layer_extractor,
    get_dims_for_loader_dict,
    prepare_grid,
)


def build_model(
    args: Any,
    dataloaders: Dict[str, DataLoader],
    layer: Optional[List[str]] = None,
    brain_area: Optional[List[str]] = None,
) -> nn.Module:
    """
    Builds an Encoder model by assembling a core, readout, and optional modules.

    The function configures each component based on the data shapes derived from
    the dataloaders and the parameters specified in the `args` object.

    Args:
        args: An object (e.g., from argparse) containing model configuration
              and hyperparameters.
        dataloaders: A dictionary of DataLoaders, mapping a dataset key
                     to a DataLoader instance.
        layer: An optional list of cortical layers to select neurons from.
        brain_area: An optional list of brain areas to select neurons from.

    Returns:
        An initialized PyTorch model (nn.Module) ready for training.
    """
    # If the dataloaders are nested (e.g., {'train': ..., 'val': ...}),
    # select the training set for model construction.
    if "train" in dataloaders:
        dataloaders = dataloaders["train"]

    # --- Determine neuron indices based on brain area and layer filters ---
    if args.layer is None:
        layer_idxs = {
            k: np.arange(len(v.dataset.neurons.cell_motor_coordinates))
            for k, v in dataloaders.items()
        }
    else:
        layer_list = [layer] if isinstance(layer, str) else layer
        layer_idxs = {
            k: np.where(np.isin(v.dataset.neurons.layer, layer_list))[0]
            for k, v in dataloaders.items()
        }

    if args.brain_area is None:
        brain_area_idxs = {
            k: np.arange(len(v.dataset.neurons.cell_motor_coordinates))
            for k, v in dataloaders.items()
        }
    else:
        area_list = [brain_area] if isinstance(brain_area, str) else brain_area
        brain_area_idxs = {
            k: np.where(np.isin(v.dataset.neurons.brain_area, area_list))[0]
            for k, v in dataloaders.items()
        }

    neurons_idxs = {
        k: np.intersect1d(layer_idxs[k], brain_area_idxs[k])
        for k in dataloaders
    }
    session_shape_dict = get_dims_for_loader_dict(dataloaders)

    # --- Build Core ---
    input_channels_per_key = [
        v["videos"][1] for v in session_shape_dict.values()
    ]
    if not np.all(np.diff(input_channels_per_key) == 0):
        raise ValueError("All sessions must have the same number of input channels.")
    args.core_input_channels = max(input_channels_per_key)

    if hasattr(args, "layer_norm") and args.layer_norm:
        args.input_dim = list(session_shape_dict.values())[0]["videos"][2:]
    core = get_core(args)(args)

    # --- Build Readout ---
    n_neurons_dict = {k: v["responses"][1] for k, v in session_shape_dict.items()}
    input_shapes_dict = {
        k: (v["videos"][0] * v["videos"][2], v["videos"][1]) + v["videos"][3:]
        if core.is_2d
        else v["videos"]
        for k, v in session_shape_dict.items()
    }

    if not core.is_2d:
        # For 3D cores, select (channels, height, width) from the output shape.
        subselect = itemgetter(0, 2, 3)
        in_shape_dict = {
            k: subselect(tuple(get_module_output(core, v)[1:]))
            for k, v in input_shapes_dict.items()
        }
    else:
        in_shape_dict = {
            k: get_module_output(core, v)[1:]
            for k, v in input_shapes_dict.items()
        }

    brain_area_to_layer = None
    if args.brain_area_to_layer is not None:
        brain_area_to_layer = brain_area_layer_extractor(
            args.brain_area_to_layer, args.hidden_channels
        )
        # Calculate total channels from specified layer ranges.
        n_channels = sum(
            el[-1] - el[0]
            for el in list(brain_area_to_layer.values())[0]
        )
        in_shape_dict = {
            k: (n_channels, v[1], v[2]) for k, v in in_shape_dict.items()
        }

    if args.modulator and getattr(args, "modulator_output_channels", None):
        extra_channels = args.modulator_output_channels[0]
        in_shape_dict = {
            k: (v[0] + extra_channels, v[1], v[2])
            for k, v in in_shape_dict.items()
        }

    if args.readout_bias:
        mean_activity_dict = {
            k: torch.from_numpy(
                np.take(
                    v.dataset.statistics.responses[
                        args.input_type
                    ].mean.mean(axis=1)
                    / v.dataset.statistics.responses[
                        args.input_type
                    ].std.mean(axis=1),
                    neurons_idxs[k],
                    axis=0,
                )
            )
            for k, v in dataloaders.items()
        }
    else:
        mean_activity_dict = {k: torch.zeros(v) for k, v in n_neurons_dict.items()}

    readout_params = {
        "in_shape_dict": in_shape_dict,
        "n_neurons_dict": n_neurons_dict,
        "brain_area_dict": {
            k: v.dataset.neurons.brain_area for k, v in dataloaders.items()
        },
        "brain_area_to_layer": brain_area_to_layer,
        "mean_activity_dict": mean_activity_dict,
    }

    if not args.disable_grid_predictor:
        grid_predictor, source_grids = prepare_grid(
            dataloaders,
            neurons_idxs,
            hidden_layers=args.grid_predictor_hidden_layers,
            input_dim=args.grid_predictor_dim,
            hidden_features=args.grid_predictor_hidden_features,
        )
        readout_params["grid_mean_predictor"] = grid_predictor
        readout_params["source_grids"] = source_grids

    readout = MultiReadoutBase(
        args=args, base_readout=get_readout(args), **readout_params
    )

    # --- Build Optional Modules ---
    shifter = None
    if args.shifter:
        data_keys = list(dataloaders.keys())
        shifter = MLPShifter(args, data_keys)

    modulator = None
    if getattr(args, "modulator", False):
        modulator_input_channels = [
            v["behavior"][1] for v in session_shape_dict.values()
        ]
        if not np.all(np.diff(modulator_input_channels) == 0):
            raise ValueError(
                "All sessions must have the same number of behavior channels."
            )
        modulator = StackedMLPBehavior(
            input_channels=max(modulator_input_channels),
            output_channels=getattr(args, "modulator_output_channels", None),
            layers=getattr(args, "modulator_layers", None),
            hidden_channels=getattr(args, "modulator_hidden_channels", None),
            hidden_layers=getattr(args, "modulator_hidden_layers", 1),
            bias=getattr(args, "modulator_bias", True),
            dropout=getattr(args, "modulator_dropout", 0.0),
            activation=getattr(args, "modulator_activation", "tanh"),
            gamma=getattr(args, "modulator_gamma", 0.0),
            temporal_conv=getattr(args, "modulator_temporal_conv", False),
            temporal_kernel=getattr(args, "modulator_temporal_kernel", 5),
        )

    gru_module = None
    if args.gru:
        rec_channels = args.rec_channels or core.output_channels
        gru_module = GRU_Module(
            input_channels=core.output_channels,
            rec_channels=rec_channels,
            input_kern=args.rec_input_kern,
            rec_kern=args.rec_kern,
            gamma_rec=args.gamma_rec,
            pad_input=args.rec_padding,
        )

    # --- Assemble Final Model ---
    model = Encoder(
        core=core,
        readout=readout,
        gru_module=gru_module,
        shifter=shifter,
        modulator=modulator,
        nonlinearity_type=args.nonlinearity,
        detach_core=args.detach_core,
    )

    model.to(args.device)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {num_params}")

    return model
