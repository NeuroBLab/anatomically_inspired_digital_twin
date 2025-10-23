import numpy as np

def get_io_dims(data_loader):
    """Gets the input/output dimensions from a PyTorch DataLoader.

    The shape of each tensor yielded by the data loader is determined from a
    single batch. The batch dimension itself is included.

    Args:
        data_loader (torch.utils.data.DataLoader): A PyTorch DataLoader that
            yields batches as tuples, namedtuples, or dictionaries.

    Returns:
        dict or tuple: If the batch is a dict or namedtuple, a dictionary
        mapping keys to tensor shapes. Otherwise, a tuple of tensor shapes.
    """
    items = next(iter(data_loader))
    if hasattr(items, "_asdict"):  # Handle namedtuples
        items = items._asdict()

    if isinstance(items, dict):
        return {k: v.shape for k, v in items.items()}
    else:
        return tuple(v.shape for v in items)


def get_dims_for_loader_dict(dataloaders):
    """Applies get_io_dims to a dictionary of DataLoaders.

    Args:
        dataloaders (dict): A dictionary mapping keys (e.g., session IDs)
            to `torch.utils.data.DataLoader` instances.

    Returns:
        dict: A dictionary with the same keys as the input, where each value
            is the output of `get_io_dims` for the corresponding DataLoader.
    """
    return {k: get_io_dims(v) for k, v in dataloaders.items()}


def prepare_grid(
    dataloaders,
    neuron_indices,
    hidden_layers=2,
    hidden_features=20,
    final_tanh=True,
    input_dim=2,
):
    """Prepares configuration for a grid predictor based on neuron coordinates.

    This function generates a configuration dictionary for a grid-prediction
    network and extracts the source neuron coordinates from the provided
    datasets.

    Args:
        dataloaders (dict): A dictionary of PyTorch DataLoaders, keyed by
            session identifier.
        neuron_indices (dict): A dictionary mapping session identifiers to the
            indices of neurons to be used for that session.
        hidden_layers (int, optional): Number of hidden layers for the grid
            predictor. Defaults to 2.
        hidden_features (int, optional): Number of features in the hidden
            layers. Defaults to 20.
        final_tanh (bool, optional): Whether to apply a tanh activation to the
            final layer. Defaults to True.
        input_dim (int, optional): The number of coordinate dimensions to use
            from the dataset (e.g., 2 for x, y). Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - dict: The configuration dictionary for the grid mean predictor.
            - dict: A dictionary mapping session identifiers to the neuron
              coordinates (`source_grids`).
    """
    grid_mean_predictor_config = {
        "hidden_layers": hidden_layers,
        "hidden_features": hidden_features,
        "final_tanh": final_tanh,
    }
    source_grids = {
        key: loader.dataset.neurons.cell_motor_coordinates[
            neuron_indices[key], :input_dim
        ]
        for key, loader in dataloaders.items()
    }
    return grid_mean_predictor_config, source_grids


def brain_area_layer_extractor(area_to_layer_map, hidden_channels):
    """Maps brain areas to their corresponding channel ranges in a feature map.

    Given a mapping from brain areas to layer indices and a list of channel
    counts per layer, this function computes the start and end channel index
    for each brain area.

    Args:
        area_to_layer_map (dict): A dictionary where keys are brain area names
            and values are lists of layer indices associated with that area.
        hidden_channels (list[int]): A list of integers representing the
            number of channels in each consecutive layer.

    Returns:
        dict: A dictionary mapping brain area names to a list of [start, end]
            channel index pairs.
    """
    # Create boundaries for channel indices from channel counts
    channel_boundaries = np.r_[0, np.cumsum(hidden_channels)]

    area_to_channel_ranges = {
        area: [
            [channel_boundaries[i], channel_boundaries[i + 1]]
            for i in layer_indices
        ]
        for area, layer_indices in area_to_layer_map.items()
    }

    # Verify that each brain area is assigned the same total number of channels
    channels_per_area = {
        k: np.sum(np.diff(v, axis=1)) for k, v in area_to_channel_ranges.items()
    }
    if not np.all(np.diff(list(channels_per_area.values())) == 0):
        raise ValueError(
            f"Brain area channel assignments are not contiguous: {channels_per_area}"
        )

    # Normalize all channel ranges to be zero-based
    min_channel_index = min(
        [v[0][0] for v in area_to_channel_ranges.values()]
    )
    normalized_ranges = {
        area: [[start - min_channel_index, end - min_channel_index] for start, end in ranges]
        for area, ranges in area_to_channel_ranges.items()
    }

    return normalized_ranges