import numpy as np
import torch
from tqdm import tqdm

from src import metrics as m


class NpRunningStats:
    """
    Computes running statistics (mean, variance, std) for NumPy arrays.

    This class accumulates statistics incrementally from chunks of data, avoiding
    the need to load the entire dataset into memory.
    """

    def __init__(self):
        """Initializes the statistics."""
        self.n = 0
        self.sum = 0
        self.sum_squared = 0

    def clear(self):
        """Resets all accumulated statistics."""
        self.n = 0
        self.sum = 0
        self.sum_squared = 0

    def push(self, x, axis=None):
        """
        Updates statistics with a new batch of data.

        Args:
            x (np.ndarray): The input data array.
            axis (int, optional): The axis along which to compute statistics.
                If None, statistics are computed over the flattened array.
                Otherwise, statistics are computed for each slice along the
                given axis, and all other dimensions are flattened.
        """
        if axis is not None:
            # Move the target axis to the front and flatten all other dimensions.
            x = np.moveaxis(x, axis, 0)
            x = x.reshape(x.shape[0], -1)
            n_elements = x.shape[-1]
        else:
            x = x.ravel()
            n_elements = x.size

        self.n += n_elements
        self.sum += x.sum(axis=-1)
        self.sum_squared += (x**2).sum(axis=-1)

    def mean(self):
        """Returns the current mean."""
        return self.sum / self.n if self.n > 0 else 0.0

    def variance(self):
        """Returns the current variance."""
        if self.n == 0:
            return 0.0
        return (self.sum_squared / self.n) - (self.mean() ** 2)

    def standard_deviation(self):
        """Returns the current standard deviation."""
        return np.sqrt(self.variance())


class TorchRunningStats:
    """
    Computes running statistics (mean, variance, std) for PyTorch tensors.

    This class accumulates statistics incrementally from chunks of data, avoiding
    the need to load the entire dataset into memory.
    """

    def __init__(self):
        """Initializes the statistics."""
        self.clear()

    def clear(self):
        """Resets all accumulated statistics."""
        self.n = 0
        self.sum = 0
        self.sum_squared = 0

    def push(self, x, axes=None):
        """
        Updates statistics with a new batch of data.

        Args:
            x (torch.Tensor): The input data tensor.
            axes (list[int], optional): A list of axes to preserve. Statistics
                are computed by aggregating over all other dimensions. If None,
                statistics are computed over the flattened tensor.
        """
        if axes is not None:
            if isinstance(axes, int):
                axes = [axes]
            # Create a target shape for the summed statistics.
            output_shape = torch.ones(x.ndim, dtype=torch.int)
            for axis in axes:
                output_shape[axis] = x.shape[axis]

            # Move preserved axes to the front and flatten the rest.
            permute_axes = list(axes) + [
                i for i in range(x.ndim) if i not in axes
            ]
            x_permuted = x.permute(permute_axes).contiguous()
            new_shape = x_permuted.shape[: len(axes)] + (-1,)
            x_reshaped = x_permuted.view(new_shape)
            n_elements = x_reshaped.size(-1)
        else:
            output_shape = torch.ones(x.ndim, dtype=torch.int)
            x_reshaped = x.view(-1)
            n_elements = x_reshaped.numel()

        self.n += n_elements
        self.sum += x_reshaped.sum(dim=-1).view(torch.Size(output_shape)[1:])
        self.sum_squared += (x_reshaped**2).sum(dim=-1).view(
            torch.Size(output_shape)[1:]
        )

    def mean(self):
        """Returns the current mean as a NumPy array."""
        return (self.sum / self.n).numpy() if self.n > 0 else 0.0

    def variance(self):
        """Returns the current variance as a NumPy array."""
        if self.n == 0:
            return 0.0
        mean_val = self.sum / self.n
        var = (self.sum_squared / self.n) - (mean_val**2)
        return var.numpy()

    def standard_deviation(self):
        """Returns the current standard deviation as a NumPy array."""
        return np.sqrt(self.variance())


def _convert_dimension_to_axes(data_key, dimensions):
    """
    Converts a list of dimension names to a list of axis indices.

    Args:
        data_key (str): The type of data (e.g., 'responses', 'videos').
        dimensions (list[str]): A list of dimension names to convert.

    Returns:
        list[int] or None: A list of corresponding axis indices, or None if
        the conversion results in an empty list.
    """
    conversion_dict = {
        "responses": {"batch": 0, "neurons": 1, "frames": 2},
        "videos": {"batch": 0, "height": 1, "width": 2, "frames": 3},
        "behavior": {"batch": 0, "channels": 1, "frames": 2},
        "pupil_center": {"batch": 0, "coordinates": 1, "frames": 2},
    }
    if not dimensions:
        return None

    key_map = conversion_dict.get(data_key, {})
    axes = [key_map[dim] for dim in dimensions if dim in key_map]
    return axes if axes else None


def compute_stats(dataloader, dimensions=None):
    """
    Computes mean and standard deviation for dataset streams.

    This function iterates through the 'train' split of the provided dataloader
    dictionary. It calculates session-specific statistics for 'responses',
    'behavior', and 'pupil_center', while aggregating 'videos' statistics
    across all sessions.

    Args:
        dataloader (dict): A dictionary containing dataloader objects, expected
            to have a 'train' key which is another dictionary mapping session
            keys to DataLoader instances.
        dimensions (list[str], optional): A list of dimension names along which
            to compute statistics (e.g., ['neurons']). If None, statistics are
            computed over flattened data. Defaults to None.

    Returns:
        dict: A nested dictionary containing the mean and standard deviation
              for each data stream, organized by session.
    """
    if dimensions is None:
        dimensions = []

    videos_stats = TorchRunningStats()
    videos_axes = _convert_dimension_to_axes("videos", dimensions)
    responses_axes = _convert_dimension_to_axes("responses", dimensions)
    behavior_axes = _convert_dimension_to_axes("behavior", dimensions)
    pupil_center_axes = _convert_dimension_to_axes("pupil_center", dimensions)

    output = {}
    for key, loader in dataloader["train"].items():
        response_stats = TorchRunningStats()
        behavior_stats = TorchRunningStats()
        pupil_center_stats = TorchRunningStats()
        session_stats = {}

        for data in tqdm(loader, desc=f"Computing stats for {key}"):
            batch = data._asdict() if not isinstance(data, dict) else data
            response_stats.push(batch["responses"], axes=responses_axes)
            videos_stats.push(batch["videos"], axes=videos_axes)
            if "behavior" in batch:
                behavior_stats.push(batch["behavior"], axes=behavior_axes)
            if "pupil_center" in batch:
                pupil_center_stats.push(
                    batch["pupil_center"], axes=pupil_center_axes
                )

        session_stats["responses"] = {
            "mean": response_stats.mean(),
            "std": response_stats.standard_deviation(),
        }
        if behavior_stats.n > 0:
            session_stats["behavior"] = {
                "mean": behavior_stats.mean(),
                "std": behavior_stats.standard_deviation(),
            }
        if pupil_center_stats.n > 0:
            session_stats["pupil_center"] = {
                "mean": pupil_center_stats.mean(),
                "std": pupil_center_stats.standard_deviation(),
            }
        output[key] = session_stats

    output["videos"] = {
        "mean": videos_stats.mean(),
        "std": videos_stats.standard_deviation(),
    }
    return output


def extract_neurons_subset(dataloader, metric_name):
    """
    Selects a subset of neurons based on a metric's median value.

    For each session in the 'val' split of the dataloader, this function
    computes a specified metric for all neurons and returns the indices of
    neurons with a metric value at or above the 50th percentile.

    Args:
        dataloader (dict): A dictionary containing dataloader objects, expected
            to have a 'val' key which is another dictionary mapping session
            keys to DataLoader instances.
        metric_name (str): The name of the metric function to use from the
            `src.metrics` module.

    Returns:
        dict: A dictionary mapping each session key to an array of indices
              for the selected neurons.
    """
    metric_fn = getattr(m, metric_name)
    output = {}
    for key, loader in dataloader["val"].items():
        responses = []
        for data in tqdm(loader, desc=f"Extracting neurons for {key}"):
            batch = data._asdict()
            responses.append(batch["responses"].cpu().numpy())

        num_frames = batch["responses"].shape[2]
        all_responses = np.concatenate(responses, axis=0)

        # The following reshape assumes a fixed trial structure.
        # These dimensions (e.g., 6 trials, 10 repeats) may need to be
        # adjusted for different experimental designs.
        # Shape changes from (trials*repeats, neurons, frames)
        # to (trials, repeats, neurons, frames).
        all_responses = all_responses.reshape(6, 10, -1, num_frames)

        session_metric = metric_fn(all_responses)
        median_threshold = np.quantile(session_metric, 0.5)
        selected_indices = np.where(session_metric >= median_threshold)[0]
        output[key] = selected_indices

    return output
