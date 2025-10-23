import json
import os
import random
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from scripts.misc.train_args import get_parser


def set_random_seed(seed: int, deterministic: bool = False):
    """Sets random seeds for major libraries to ensure reproducibility.

    Args:
        seed: The integer value to use as the seed.
        deterministic: If True, configures PyTorch to use deterministic
            algorithms, which may have a performance cost.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if deterministic:
            # This is needed for full determinism with some PyTorch operations.
            torch.use_deterministic_algorithms(True)
            # This environment variable is needed for deterministic convolution.
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def create_data_paths(
    dataset: str, session_ids: Optional[Tuple[str, ...]]
) -> List[str]:
    """Constructs a list of full directory paths for dataset sessions.

    Args:
        dataset: The name of the dataset directory, expected inside 'data/'.
        session_ids: A tuple of session IDs. If None, all subdirectories
            in the dataset folder are used as sessions.

    Returns:
        A list of full paths to each session directory.
    """
    data_folder = os.path.join('data', dataset)
    if session_ids is None:
        session_ids = tuple(os.listdir(data_folder))
    # The trailing empty string ensures the path ends with a directory separator.
    return [os.path.join(data_folder, session, '') for session in session_ids]


def create_paths_dict(
    dataset: str, session_ids: Optional[Tuple[str, ...]]
) -> Dict[str, List[str]]:
    """Constructs a dictionary mapping a dataset name to its session paths.

    Args:
        dataset: The name of the dataset directory, expected inside 'data/'.
        session_ids: A tuple of session IDs. If None, all subdirectories
            in the dataset folder are used as sessions.

    Returns:
        A dictionary with the dataset name as the key and a list of
        session paths as the value.
    """
    paths = create_data_paths(dataset, session_ids)
    return {dataset: paths}


class JsonConfig:
    """A configuration class that loads settings from a JSON file.

    Attributes of this class are dynamically created from the key-value
    pairs in the JSON file, making it behave like an argparse.Namespace.
    """

    def __init__(self, path: str):
        """Initializes the config object by reading a JSON file.

        Args:
            path: The path to the JSON configuration file.

        Raises:
            FileNotFoundError: If the specified path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'Configuration file not found: {path}')
        with open(path) as json_file:
            kwargs = json.load(json_file)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, args: Union[Namespace, Dict[str, Any]]):
        """Updates attributes from another configuration object or dictionary.

        Args:
            args: An argparse.Namespace or a dictionary containing new
                or updated key-value pairs.
        """
        new_args = vars(args) if isinstance(args, Namespace) else args
        for key, value in new_args.items():
            setattr(self, key, value)


def initialize_model_args(args: Namespace) -> JsonConfig:
    """Initializes and merges configuration from multiple sources.

    This function establishes a clear hierarchy for configuration settings:
    1.  Command-line arguments (or those passed in the `args` object).
    2.  Settings from a saved 'args.json' file in the model's output directory.
    3.  Default values defined in the argument parser from `get_parser()`.

    Args:
        args: An argparse.Namespace object, typically from parsing
            command-line arguments. It must contain `output_dir` and
            `model_name`.

    Returns:
        A JsonConfig object containing the final, merged configuration.
    """
    output_dir = os.path.join(args.output_dir, args.model_name)
    args.output_dir = output_dir
    json_path = os.path.join(output_dir, 'args.json')

    # Priority 3: Load base arguments from the JSON file.
    config = JsonConfig(json_path)

    # Priority 1: Override with any arguments provided at runtime (e.g., CLI).
    config.update(args)

    # Priority 2: Incorporate any missing default arguments from the main parser.
    # This ensures that new arguments added to the parser are included even
    # when loading an older configuration file.
    parser = get_parser()
    parser.set_defaults(**vars(config))
    # Parsing an empty list populates the namespace with all current defaults.
    defaults_ns = parser.parse_args([])

    for key, value in vars(defaults_ns).items():
        if not hasattr(config, key):
            setattr(config, key, value)

    # Force a specific setting after all merges.
    config.init_gaussian = False

    return config