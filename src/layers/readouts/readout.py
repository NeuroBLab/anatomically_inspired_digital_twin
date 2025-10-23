import torch
from neuralpredictors.layers.readouts.base import ClonedReadout

_READOUTS = {}


def register(name):
    """A decorator to register a new readout class."""

    def add_to_dict(fn):
        _READOUTS[name] = fn
        return fn

    return add_to_dict


def get_readout(args):
    """
    Retrieves a readout class from the registry.

    Args:
        args: An object containing the readout name in `args.readout`.

    Returns:
        The registered readout class.
    """
    if args.readout not in _READOUTS:
        raise NotImplementedError(f"Readout '{args.readout}' is not implemented.")
    return _READOUTS[args.readout]


class MultiReadoutBase(torch.nn.ModuleDict):
    """
    A dictionary-like container for managing multiple readout modules, one for
    each dataset identified by a data_key.
    """

    _base_readout = None

    def __init__(
        self,
        args,
        in_shape_dict,
        n_neurons_dict,
        brain_area_dict,
        brain_area_to_layer=None,
        source_grids=None,
        base_readout=None,
        mean_activity_dict=None,
        clone_readout=False,
        **kwargs,
    ):
        """
        Initializes the MultiReadoutBase.

        Args:
            args: A configuration object with model parameters.
            in_shape_dict (dict): A dictionary mapping data_key to core output shape.
            n_neurons_dict (dict): A dictionary mapping data_key to the number of neurons.
            brain_area_dict (dict): A dictionary mapping data_key to brain area info.
            brain_area_to_layer (dict, optional): Mapping from brain areas to core
                layer indices. Defaults to None.
            source_grids (dict, optional): A dictionary of source grids for grid
                prediction. Defaults to None.
            base_readout (nn.Module, optional): The readout class to instantiate.
                Defaults to None.
            mean_activity_dict (dict, optional): A dictionary of mean neuron
                activities for bias initialization. Defaults to None.
            clone_readout (bool, optional): If True, clones the first readout for all
                subsequent data_keys. Defaults to False.
            **kwargs: Additional arguments passed to the readout constructor.
        """
        super().__init__()
        if self._base_readout is None:
            self._base_readout = base_readout
        if self._base_readout is None:
            raise ValueError("Attribute _base_readout must be set.")

        first_data_key = None
        for i, data_key in enumerate(n_neurons_dict):
            if i == 0:
                first_data_key = data_key

            mean_activity = (
                mean_activity_dict[data_key] if mean_activity_dict else None
            )
            if args.disable_grid_predictor:
                readout_kwargs = kwargs.copy()
            else:
                readout_kwargs = self.prepare_readout_kwargs(
                    i, source_grids, data_key, first_data_key, **kwargs
                )

            if i == 0 or not clone_readout:
                self.add_module(
                    data_key,
                    self._base_readout(
                        in_shape=in_shape_dict[data_key],
                        outdims=n_neurons_dict[data_key],
                        brain_area=brain_area_dict[data_key],
                        brain_area_to_layer=brain_area_to_layer,
                        mean_activity=mean_activity,
                        feature_reg_weight=args.readout_reg_weight,
                        dispersion_reg=args.dispersion_reg,
                        num_samples=args.num_samples,
                        hidden_channels=args.hidden_channels,
                        regularize_per_layer=args.regularize_per_layer,
                        modulator_channels=args.modulator_output_channels[0],
                        **readout_kwargs,
                    ),
                )
                original_readout = data_key
            else:
                self.add_module(data_key, ClonedReadout(self[original_readout]))

        self.initialize(mean_activity_dict)

    def prepare_readout_kwargs(
        self,
        i,
        source_grids,
        data_key,
        first_data_key,
        grid_mean_predictor=None,
        share_transform=False,
        share_grid=False,
        share_features=False,
        **kwargs,
    ):
        """Prepares keyword arguments for the readout constructor."""
        readout_kwargs = kwargs.copy()

        if grid_mean_predictor:
            readout_kwargs["source_grid"] = source_grids[data_key]
            readout_kwargs["grid_mean_predictor"] = grid_mean_predictor
            if share_transform:
                readout_kwargs["shared_transform"] = (
                    None if i == 0 else self[first_data_key].mu_transform
                )
        elif share_grid:
            readout_kwargs["shared_grid"] = {
                "match_ids": readout_kwargs["shared_match_ids"][data_key],
                "shared_grid": None if i == 0 else self[first_data_key].shared_grid,
            }

        if share_features:
            readout_kwargs["shared_features"] = {
                "match_ids": readout_kwargs["shared_match_ids"][data_key],
                "shared_features": (
                    None if i == 0 else self[first_data_key].shared_features
                ),
            }
        else:
            readout_kwargs["shared_features"] = None
        return readout_kwargs

    def forward(self, *args, data_key=None, **kwargs):
        """
        Performs a forward pass on the specified readout.
        If only one readout exists, it is used by default.
        """
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def initialize(self, mean_activity_dict=None):
        """Initializes all contained readout modules."""
        for data_key, readout in self.items():
            mean_activity = mean_activity_dict[data_key] if mean_activity_dict else None
            readout.initialize(mean_activity)

    def regularizer(self, data_key=None, reduction="sum", average=None):
        """
        Computes the regularization term for the specified readout.
        """
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key].regularizer(reduction=reduction, average=average)


class MultiReadoutSharedParametersBase(MultiReadoutBase):
    """
    Base class for MultiReadouts that share parameters between readouts.
    For more information on which parameters can be shared, refer to specific
    readout implementations like FullGaussian2d.
    """

    def prepare_readout_kwargs(
        self,
        i,
        data_key,
        first_data_key,
        grid_mean_predictor=None,
        grid_mean_predictor_type=None,
        share_transform=False,
        share_grid=False,
        share_features=False,
        **kwargs,
    ):
        """Prepares keyword arguments for shared-parameter readouts."""
        readout_kwargs = kwargs.copy()

        if grid_mean_predictor:
            if grid_mean_predictor_type == "cortex":
                readout_kwargs["source_grid"] = readout_kwargs["source_grids"][data_key]
                readout_kwargs["grid_mean_predictor"] = grid_mean_predictor
            else:
                raise KeyError(
                    f"Grid mean predictor '{grid_mean_predictor_type}' does not exist."
                )
            if share_transform:
                readout_kwargs["shared_transform"] = (
                    None if i == 0 else self[first_data_key].mu_transform
                )
        elif share_grid:
            readout_kwargs["shared_grid"] = {
                "match_ids": readout_kwargs["shared_match_ids"][data_key],
                "shared_grid": None if i == 0 else self[first_data_key].shared_grid,
            }

        if share_features:
            readout_kwargs["shared_features"] = {
                "match_ids": readout_kwargs["shared_match_ids"][data_key],
                "shared_features": (
                    None if i == 0 else self[first_data_key].shared_features
                ),
            }
        else:
            readout_kwargs["shared_features"] = None
        return readout_kwargs