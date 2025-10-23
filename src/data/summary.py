import json
import pickle
import numpy as np

class Summary:
    """Manages the collection and serialization of experiment information.

    This class provides a structured way to gather details about the model,
    dataset, training parameters, and results, and save them to a file.

    Attributes:
        summary_info (dict): A dictionary holding all collected information.
    """

    def __init__(self, model, dataset, input_type):
        """Initializes the Summary object.

        Args:
            model (str): The name or identifier of the model.
            dataset (str): The name of the dataset being used.
            input_type (str): The type of input data (e.g., 'natural').
        """
        self.model = model
        self.dataset = dataset
        self.input_type = input_type
        self.summary_info = {}
        self._initialize_info()

    def _initialize_info(self):
        """Sets up the basic structure of the summary dictionary."""
        self.summary_info["model"] = {}
        self.summary_info["dataset"] = {}
        self.summary_info["training"] = {}
        self.summary_info["results"] = {}

    def add_dataloader_info(self, dataloader):
        """Extracts and stores information from dataloaders.

        This method iterates through the 'train' dataloader views, calculates
        the number of trials and neurons, and extracts metadata like neuron IDs,
        layers, and brain areas. It performs special filtering for the
        'microns30' dataset based on the specified input type.

        Args:
            dataloader (dict): A dictionary of dataloaders, expected to have
                a 'train' key which maps to a dictionary of data views.
        """
        self.summary_info["model"] = {}  # Reset model-specific info
        train_data_views = dataloader["train"]
        total_trials = 0
        total_neurons = 0

        # For the 'microns30' dataset, map input types to specific trial types.
        microns_type_map = {
            "natural": ["Clip"],
            "parametric": ["Monet2", "Tippy"],
            "ori": ["Monet2"],
            "clips": ["Clip", "Monet2", "Tippy"],
            "spontaneous": ["black"],
        }

        for data_key, data_view in train_data_views.items():
            trial_info = data_view.dataset.trial_info
            neuron_info = data_view.dataset.neurons

            # Start with all indices corresponding to the 'train' tier.
            valid_indices = np.where(trial_info.tiers == "train")[0]

            layer = None
            brain_area = None
            if self.dataset == "microns30":
                trial_types = microns_type_map.get(self.input_type)
                if trial_types:
                    type_indices = np.where(
                        np.isin(trial_info.type, trial_types)
                    )[0]
                    valid_indices = np.intersect1d(
                        valid_indices, type_indices
                    )
                layer = neuron_info.layer
                brain_area = neuron_info.brain_area

            self.summary_info["model"][data_key] = {
                "num_trials": valid_indices.size,
                "unit_ids": neuron_info.unit_ids,
                "layer": layer,
                "brain_area": brain_area,
                "cell_motor_coordinates": neuron_info.cell_motor_coordinates,
            }
            total_trials += valid_indices.size
            total_neurons += neuron_info.unit_ids.size

        self.summary_info["model"]["num_trials"] = total_trials
        self.summary_info["model"]["num_neurons"] = total_neurons

    def add_readout_info(self, model):
        """Extracts and stores information from the model's readout layers.

        Args:
            model: The model object, which is expected to have a `readout`
                attribute that is a dictionary of readout layers.
        """
        for data_key, readout_layer in model.readout.items():
            mu = readout_layer.mu.cpu().detach().numpy().squeeze()
            feature_weights = (
                readout_layer.features.data.detach().cpu().numpy().squeeze()
            )
            self.summary_info["model"][data_key].update(
                {
                    "spatial_mask": mu,
                    "feature_weights": feature_weights,
                }
            )
            # Check for an optional predicted source grid and add it if present.
            if getattr(readout_layer, "_predicted_grid", False):
                source_grid = readout_layer.source_grid.cpu().detach().numpy()
                self.summary_info["model"][data_key].update(
                    {"source_grid": source_grid}
                )

    def add_training_info(self, args):
        """Stores training configuration arguments.

        Args:
            args: An object (e.g., from argparse) whose attributes represent
                the training configuration.
        """
        self.summary_info["training"] = vars(args)

    def add_dataset_info(
        self, neurons_idxs=None, input_stats=None, directions=None,
        singular_values=None
    ):
        """Stores dataset-specific information.

        Args:
            neurons_idxs (any, optional): Indices of neurons used.
            input_stats (any, optional): Statistics of the input data.
            directions (any, optional): Directional data, e.g., from PCA.
            singular_values (any, optional): Singular values, e.g., from SVD.
        """
        if neurons_idxs is not None:
            self.summary_info["dataset"]["neurons_idxs"] = neurons_idxs
        if input_stats is not None:
            self.summary_info["dataset"]["statistics"] = input_stats
        if directions is not None:
            self.summary_info["dataset"]["directions"] = directions
        if singular_values is not None:
            self.summary_info["dataset"]["singular_values"] = singular_values

    def add_model_data(self, model):
        """Calculates and stores model-wide data like parameter counts.

        Args:
            model: The model object.
        """
        model_info = self.summary_info["model"]
        num_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        model_info["num_params"] = num_params

        if "num_trials" in model_info and num_params > 0:
            num_trials = model_info["num_trials"]
            # Assuming 300 frames per trial for this ratio calculation.
            frames_params_ratio = (num_trials * 300) / num_params
            model_info["frames_params_ratio"] = frames_params_ratio

    def add_results(self, results, level):
        """Adds evaluation results to the summary under a specified level.

        Args:
            results (dict): A dictionary of results, where keys are data keys
                (e.g., neuron groups) and values are dictionaries of metrics.
            level (str): The evaluation level, e.g., 'train', 'validation',
                or 'test'.
        """
        level_results = self.summary_info["results"].setdefault(level, {})
        for data_key, metrics in results.items():
            data_key_results = level_results.setdefault(data_key, {})
            data_key_results.update(metrics)

    def _convert_to_str(self, data_dict):
        """Recursively converts all values in a dictionary to strings.

        Note: This is a lossy conversion and should be used with caution.

        Args:
            data_dict (dict): The dictionary to convert.
        """
        for key, value in data_dict.items():
            if isinstance(value, dict):
                self._convert_to_str(value)
            else:
                data_dict[key] = str(value)

    def _numpy_to_list(self, data):
        """Recursively converts NumPy arrays in a nested structure to lists.

        Args:
            data (any): The data structure to process (e.g., dict, list).

        Returns:
            any: The processed data structure with NumPy arrays converted.
        """
        if isinstance(data, dict):
            return {k: self._numpy_to_list(v) for k, v in data.items()}
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, list):
            return [self._numpy_to_list(i) for i in data]
        return data

    def save_summary(self, output_path, force_string=False):
        """Saves the summary dictionary to a JSON file.

        Args:
            output_path (str): The path to the output JSON file.
            force_string (bool): If True, converts all values to strings
                before saving.
        """
        if force_string:
            self._convert_to_str(self.summary_info)

        serializable_summary = self._numpy_to_list(self.summary_info)

        with open(output_path, "w") as f:
            json.dump(serializable_summary, f, indent=4)

    def save_to_pickle(self, output_path):
        """Saves the summary dictionary to a pickle file.

        Args:
            output_path (str): The path to the output pickle file.
        """
        with open(output_path, "wb") as f:
            pickle.dump(self.summary_info, f)