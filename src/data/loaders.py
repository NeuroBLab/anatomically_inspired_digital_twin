import copy
import os
import random
import typing as t

import numpy as np
import torch
from neuralpredictors.data.datasets import MovieFileTreeDataset
from neuralpredictors.data.samplers import SubsetSequentialSampler
from neuralpredictors.data.transforms import (
    AddBehaviorAsChannels,
    AddPupilCenterAsChannels,
    ChangeChannelsOrder,
    CutVideos,
    ExpandChannels,
    Subsample,
    Subsequence,
    ToTensor,
)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.data.transforms import (
    AddFrames,
    AddPupilCenterToBehavior,
    ApplyAffine,
    GrayScreenFrames,
    NeuroNormalizer,
    RandomNoise,
    RemoveSpontActivity,
    RepeatChannels,
    ResizeInputs,
    ZeroBehavior,
)


def get_loader(
    args: t.Any,
    dataset: t.Union[str, t.List[str]],
    batch_size: int,
    paths_dict: t.Dict[str, str],
    input_type: str = "all",
    statistics: bool = False,
    input_stats: t.Optional[t.Dict] = None,
    neurons_subset: t.Optional[t.Dict] = None,
) -> t.Dict[str, t.Dict[str, DataLoader]]:
    """
    Factory function to get a specific data loader for a given dataset.

    Args:
        args: An object containing configuration parameters (e.g., from argparse).
        dataset: The name of the dataset or a list containing the name.
        batch_size: The number of samples per batch.
        paths_dict: A dictionary mapping dataset names to their file paths.
        input_type: Specifies the type of input data to use (e.g., 'all', 'clips').
        statistics: If True, returns a loader for computing statistics.
        input_stats: Precomputed statistics for normalization.
        neurons_subset: A dictionary mapping dataset names to neuron indices to subset.

    Returns:
        A dictionary of PyTorch DataLoaders, partitioned into 'train', 'val', 'test'.
    """
    dataset_name = dataset[0] if isinstance(dataset, list) else dataset

    loader_map = {
        "microns": MicronsLoader,
        "microns30": Microns30Loader,
        "objects": ObjectsVideoLoader,
        "objects_scaled": ObjectsVideoLoader,
        "objects_original": ObjectsVideoLoader,
        "white_noise": WhiteNoiseLoader,
        "towards": TowardsLoader,
        "object_rotations": ObjectRotationsVideoLoader,
    }
    loader_class = loader_map.get(dataset_name)
    if not loader_class:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = loader_class(batch_size, args, input_type, input_stats, neurons_subset)
    return loader.load(paths_dict[dataset_name], statistics)


class LoaderBase:
    """
    Base class for creating data loaders.

    Handles common tasks like setting random seeds, creating data tiers,
    and initializing PyTorch DataLoader instances.

    Attributes:
        batch_size (int): The batch size for the data loaders.
        args (t.Any): Configuration object.
        input_type (str): Type of input data.
        input_stats (dict, optional): Precomputed statistics for normalization.
        neurons_subset (dict, optional): Neuron indices for subsetting.
        data_keys (list): List of data keys to load (e.g., 'videos', 'responses').
        generator (torch.Generator): Random number generator for reproducibility.
    """

    def __init__(
        self,
        batch_size: int,
        args: t.Any,
        input_type: str = "all",
        input_stats: t.Optional[t.Dict] = None,
        neurons_subset: t.Optional[t.Dict] = None,
    ):
        """Initializes the LoaderBase."""
        data_keys = ["videos", "responses"]
        if getattr(args, "include_behavior", False):
            data_keys.append("behavior")
        if getattr(args, "include_pupil_centers", False):
            data_keys.append("pupil_center")

        self.data_keys = data_keys
        self.batch_size = batch_size
        self.args = args
        self.input_type = input_type
        self.input_stats = input_stats
        self.neurons_subset = neurons_subset
        self.generator = torch.Generator().manual_seed(args.seed)

        # Augmentation parameters
        self.alpha_affine = getattr(self.args, "alpha_affine", 0.5)
        self.rotation_max = getattr(self.args, "rotation_max", 20)
        self.scale_max = getattr(self.args, "scale_max", 0.2)
        self.translation_max = getattr(self.args, "translation_max", 0.1)
        self.alpha_noise = getattr(self.args, "alpha_noise", 0.5)
        self.noise_level = getattr(self.args, "noise_level", 10)

        self._set_seed(args.seed)

    def _set_seed(self, seed: int):
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def statistics_transforms(self, dataset: MovieFileTreeDataset):
        """
        Configures transforms for computing dataset statistics.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented in child class")

    def resolve_transforms(self, dataset: MovieFileTreeDataset, tier: str):
        """
        Configures transforms for a specific data tier (train, val, test).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented in child class")

    def create_tiers(
        self,
        dataloaders: t.Dict,
        dataset: MovieFileTreeDataset,
        path: str,
        statistics: bool = False,
    ):
        """
        Splits the dataset into train, val, and test tiers and creates loaders.

        Args:
            dataloaders: Dictionary to store the created DataLoaders.
            dataset: The MovieFileTreeDataset instance.
            path: The path to the dataset, used to derive its name.
            statistics: If True, applies statistics transforms.
        """
        tier_array = dataset.trial_info.tiers
        for tier in ["train", "val", "test"]:
            subset_idx = np.where(tier_array == tier)[0]
            if len(subset_idx) == 0:
                continue

            if tier == "train":
                sampler = SubsetRandomSampler(subset_idx, generator=self.generator)
            else:
                sampler = SubsetSequentialSampler(subset_idx)

            dataset_name = path.split("/")[-2]
            if statistics:
                self.statistics_transforms(dataset)
            else:
                self.resolve_transforms(dataset, tier)
            dataloaders[tier][dataset_name] = self._get_dataloader(dataset, sampler)

    def load(
        self, paths: t.List[str], statistics: bool = False
    ) -> t.Dict[str, t.Dict[str, DataLoader]]:
        """
        Loads datasets from paths and creates dataloaders.

        Args:
            paths: A list of file paths to the datasets.
            statistics: If True, configures loaders for statistics computation.

        Returns:
            A dictionary of DataLoaders for each tier and dataset.
        """
        dataloaders = {"train": {}, "val": {}, "test": {}}
        for path in paths:
            dataset = MovieFileTreeDataset(
                path, *self.data_keys, use_cache=getattr(self.args, "use_cache", True)
            )
            self.create_tiers(dataloaders, dataset, path, statistics)
        return dataloaders

    def _get_dataloader(
        self,
        dataset: MovieFileTreeDataset,
        sampler: t.Any,
        batch_size: t.Optional[int] = None,
    ) -> DataLoader:
        """
        Creates a PyTorch DataLoader instance.

        Args:
            dataset: The dataset to load.
            sampler: The sampler for drawing samples.
            batch_size: The batch size. Defaults to self.batch_size.

        Returns:
            A configured DataLoader.
        """

        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size if batch_size is None else batch_size,
            worker_init_fn=worker_init_fn,
            generator=self.generator,
        )

    def summarize(self, dataloaders: t.Dict):
        """Prints a summary of the loaded data."""
        total_videos, total_frames, total_minutes = 0, 0, 0

        for tier, datasets in dataloaders.items():
            for loader in datasets.values():
                fps = loader.dataset.trial_info.target_fps[0]
                n_frames = loader.dataset.statistics.videos["all"].mean.shape[-1]
                n_videos = len(loader) * self.batch_size

                total_videos += n_videos
                total_frames += n_videos * n_frames
                total_minutes += n_videos * n_frames / fps / 60

        print(f"Total number of videos: {total_videos}")
        print(f"Total number of frames: {total_frames}")
        print(f"Total number of minutes: {round(total_minutes, 2)}")


class MicronsLoader(LoaderBase):
    """Data loader for the MICrONS dataset."""

    def __init__(
        self,
        batch_size: int,
        args: t.Any,
        input_type: str = "all",
        input_stats: t.Optional[t.Dict] = None,
        neurons_subset: t.Optional[t.Dict] = None,
    ):
        super().__init__(batch_size, args, input_type, input_stats, neurons_subset)
        self.cutvideos = self.input_type in ["all", "clips"]

    def statistics_transforms(self, dataset: MovieFileTreeDataset):
        """Configures transforms for computing MICrONS statistics."""
        dataset_name = dataset.dirname.split("/")[-2]
        neurons_idxs = self._get_neuron_indices(dataset, dataset_name)

        transforms = []
        if self.cutvideos:
            transforms.append(
                CutVideos(
                    max_frame=getattr(self.args, "max_frame", 300),
                    frame_axis={data_key: -1 for data_key in self.data_keys},
                    target_groups=self.data_keys,
                )
            )
        if getattr(self.args, "removeSpont", False):
            transforms.append(
                RemoveSpontActivity(dataset, getattr(self.args, "n_components", 128))
            )
        transforms.append(Subsample(neurons_idxs, target_index=0))
        dataset.transforms = transforms

    def resolve_transforms(self, dataset: MovieFileTreeDataset, tier: str):
        """Configures transforms for the MICrONS dataset."""
        dataset_name = dataset.dirname.split("/")[-2]
        neurons_idxs = self._get_neuron_indices(dataset, dataset_name)

        transforms = []
        if self.cutvideos:
            transforms.append(
                CutVideos(
                    max_frame=getattr(self.args, "max_frame", 300),
                    frame_axis={data_key: -1 for data_key in self.data_keys},
                    target_groups=self.data_keys,
                )
            )

        # Training-only augmentations
        if tier == "train":
            if getattr(self.args, "random_noise", False):
                transforms.append(
                    RandomNoise(
                        noise_level=self.alpha_noise * self.noise_level,
                        in_name="videos",
                    )
                )
            if getattr(self.args, "augment_data", False):
                transforms.append(
                    ApplyAffine(
                        degrees=(
                            -self.alpha_affine * self.rotation_max,
                            self.alpha_affine * self.rotation_max,
                        ),
                        translate=(
                            self.alpha_affine * self.translation_max,
                            self.alpha_affine * self.translation_max,
                        ),
                        scale=(
                            1 - self.alpha_affine * self.scale_max,
                            1 + self.alpha_affine * self.scale_max,
                        ),
                        in_name="videos",
                        frame_axis=-1,
                        per_frames=getattr(self.args, "augment_per_frame", False),
                    )
                )

        # Normalization and data shaping
        exclude_keys = getattr(self.args, "exclude", [])
        if getattr(self.args, "resnet", False):
            exclude_keys.append("videos")
        transforms.append(
            NeuroNormalizer(
                dataset,
                stats_source=self.input_type,
                exclude=exclude_keys,
                input_stats=self.input_stats,
                in_name="videos",
            )
        )
        transforms.append(Subsample(neurons_idxs, target_index=0))
        transforms.append(ChangeChannelsOrder((2, 0, 1), in_name="videos"))
        transforms.append(ChangeChannelsOrder((1, 0), in_name="responses"))

        # Subsequence sampling for managing video length
        frames = getattr(self.args, "frames", 60)
        if tier == "train":
            transforms.append(
                Subsequence(
                    frames=frames,
                    channel_first=(),
                    offset=getattr(self.args, "offset", -1),
                )
            )
        elif frames < 300:
            transforms.append(
                Subsequence(
                    frames=frames,
                    channel_first=(),
                    offset=getattr(self.args, "val_offset", 50),
                )
            )

        # Final tensor formatting
        transforms.append(ChangeChannelsOrder((1, 0), in_name="responses"))
        transforms.append(ExpandChannels("videos"))
        if getattr(self.args, "resnet", False):
            transforms.append(RepeatChannels(3, "videos", channel_axis=0))

        if getattr(self.args, "include_behavior", False):
            transforms.append(ChangeChannelsOrder((1, 0), in_name="behavior"))
            transforms.append(AddBehaviorAsChannels("videos"))
        if getattr(self.args, "include_pupil_centers", False):
            transforms.append(ChangeChannelsOrder((1, 0), in_name="pupil_center"))
        if getattr(self.args, "include_pupil_centers_as_channels", False):
            transforms.append(AddPupilCenterAsChannels("videos"))

        transforms.append(ToTensor(getattr(self.args, "device", "cpu") == "cuda"))
        dataset.transforms = transforms

    def create_tiers(
        self,
        dataloaders: t.Dict,
        dataset_base: MovieFileTreeDataset,
        path: str,
        statistics: bool = False,
    ):
        """Custom tier creation for MICrONS to handle specific trial types."""
        tier_array = dataset_base.trial_info.tiers
        val_idx = np.isin(tier_array, ["val", "oracle"])
        mask = np.where(
            np.logical_or(val_idx, self._get_clip_indices(dataset_base, tier_array))
        )[0]

        for tier in ["train", "val", "test"]:
            dataset = copy.deepcopy(dataset_base)
            tier_set = {tier}
            if tier == "val":
                tier_set.add("oracle")

            subset_idx = np.where(np.isin(tier_array, list(tier_set)))[0]
            subset_idx = np.intersect1d(subset_idx, mask)
            if len(subset_idx) == 0:
                continue

            # Subsample validation set to reduce evaluation time
            if tier == "val":
                subset_idx = np.concatenate([subset_idx[i::6] for i in range(6)])

            if tier == "train" and self.input_type != "ori":
                sampler = SubsetRandomSampler(subset_idx, generator=self.generator)
            else:
                sampler = SubsetSequentialSampler(subset_idx)

            dataset_name = path.split("/")[-2]
            if statistics:
                self.statistics_transforms(dataset)
            else:
                self.resolve_transforms(dataset, tier)

            batch_size = self.batch_size
            if getattr(self.args, "resnet", False) and tier == "val":
                batch_size = 1
            dataloaders[tier][dataset_name] = self._get_dataloader(
                dataset, sampler, batch_size=batch_size
            )

    def _get_neuron_indices(
        self, dataset: MovieFileTreeDataset, dataset_name: str
    ) -> np.ndarray:
        """Gets neuron indices based on layer, area, and subset arguments."""
        layer_indices = self._get_layer_indices(dataset)
        area_indices = self._get_area_indices(dataset)
        neurons_idxs = np.intersect1d(layer_indices, area_indices)
        if self.neurons_subset is not None:
            neurons_idxs = np.intersect1d(
                neurons_idxs, self.neurons_subset[dataset_name]
            )
        return neurons_idxs

    def _get_layer_indices(self, dataset: MovieFileTreeDataset) -> np.ndarray:
        """Returns neuron indices corresponding to the specified cortical layer."""
        layer = getattr(self.args, "layer", None)
        if layer is None:
            return np.arange(len(dataset.neurons.cell_motor_coordinates))
        if isinstance(layer, str):
            return np.where(dataset.neurons.layer == layer)[0]
        return np.where(np.isin(dataset.neurons.layer, layer))[0]

    def _get_area_indices(self, dataset: MovieFileTreeDataset) -> np.ndarray:
        """Returns neuron indices corresponding to the specified brain area."""
        brain_area = getattr(self.args, "brain_area", None)
        if brain_area is None:
            return np.arange(len(dataset.neurons.cell_motor_coordinates))
        if isinstance(brain_area, str):
            return np.where(dataset.neurons.brain_area == brain_area)[0]
        return np.where(np.isin(dataset.neurons.brain_area, brain_area))[0]

    def _get_clip_indices(
        self, dataset: MovieFileTreeDataset, tier_array: np.ndarray
    ) -> np.ndarray:
        """Returns trial indices based on the specified input type."""
        type_map = {
            "all": ["Clip", "Monet2", "Trippy", "black"],
            "clips": ["Clip", "Monet2", "Trippy"],
            "natural": ["Clip"],
            "parametric": ["Monet2", "Trippy"],
            "ori": ["Monet2"],
            "spontaneous": ["black"],
        }
        if self.input_type in type_map:
            return np.isin(dataset.trial_info.type, type_map[self.input_type])
        return np.ones(len(tier_array), dtype=bool)


class Microns30Loader(MicronsLoader):
    """Data loader for the MICrONS dataset with 30Hz responses."""

    def resolve_transforms(self, dataset: MovieFileTreeDataset, tier: str):
        """Configures transforms for the 30Hz MICrONS dataset."""
        dataset_name = dataset.dirname.split("/")[-2]
        neurons_idxs = self._get_neuron_indices(dataset, dataset_name)

        transforms = []
        if self.cutvideos:
            transforms.append(
                CutVideos(
                    max_frame=getattr(self.args, "max_frame", 300),
                    frame_axis={data_key: -1 for data_key in self.data_keys},
                    target_groups=self.data_keys,
                )
            )

        # Training-only augmentations
        if tier == "train":
            if getattr(self.args, "random_noise", False):
                transforms.append(
                    RandomNoise(
                        noise_level=self.alpha_noise * self.noise_level,
                        in_name="videos",
                    )
                )
            if getattr(self.args, "augment_data", False):
                transforms.append(
                    ApplyAffine(
                        degrees=(
                            -self.alpha_affine * self.rotation_max,
                            self.alpha_affine * self.rotation_max,
                        ),
                        translate=(
                            self.alpha_affine * self.translation_max,
                            self.alpha_affine * self.translation_max,
                        ),
                        scale=(
                            1 - self.alpha_affine * self.scale_max,
                            1 + self.alpha_affine * self.scale_max,
                        ),
                        in_name="videos",
                        frame_axis=-1,
                        per_frames=getattr(self.args, "augment_per_frame", False),
                    )
                )

        if getattr(self.args, "removeSpont", False):
            transforms.append(
                RemoveSpontActivity(dataset, getattr(self.args, "n_components", 128))
            )

        transforms.append(Subsample(neurons_idxs, target_index=0))

        # Normalization and data shaping
        exclude_keys = getattr(self.args, "exclude", [])
        if getattr(self.args, "resnet", False):
            exclude_keys.append("videos")
        transforms.append(
            NeuroNormalizer(
                dataset,
                stats_source=self.input_type,
                exclude=exclude_keys,
                input_stats=self.input_stats,
                in_name="videos",
            )
        )
        transforms.append(ChangeChannelsOrder((2, 0, 1), in_name="videos"))
        transforms.append(ChangeChannelsOrder((1, 0), in_name="responses"))

        # Subsequence sampling
        frames = getattr(self.args, "frames", 60)
        if tier == "train":
            transforms.append(
                Subsequence(
                    frames=frames,
                    channel_first=(),
                    offset=getattr(self.args, "offset", -1),
                )
            )
        elif frames < 300:
            transforms.append(
                Subsequence(
                    frames=frames,
                    channel_first=(),
                    offset=getattr(self.args, "val_offset", 50),
                )
            )

        # Final tensor formatting
        transforms.append(ChangeChannelsOrder((1, 0), in_name="responses"))
        transforms.append(ExpandChannels("videos"))
        if getattr(self.args, "resnet", False):
            transforms.append(RepeatChannels(3, "videos", channel_axis=0))

        # Handle behavioral and pupil data
        use_modulator = getattr(self.args, "modulator", False)
        if getattr(self.args, "include_behavior", False):
            transforms.append(ChangeChannelsOrder((1, 0), in_name="behavior"))
            if not use_modulator:
                transforms.append(AddBehaviorAsChannels("videos"))
        if getattr(self.args, "include_pupil_centers", False):
            transforms.append(ChangeChannelsOrder((1, 0), in_name="pupil_center"))
        if getattr(self.args, "include_pupil_centers_as_channels", False):
            if not use_modulator:
                transforms.append(AddPupilCenterAsChannels("videos"))

        if use_modulator:
            assert getattr(
                self.args, "include_behavior", False
            ), "Behavior must be included for modulator"
            if getattr(self.args, "include_pupil_centers_as_channels", False):
                transforms.append(
                    AddPupilCenterToBehavior(
                        behavior_name="behavior", pupil_center_name="pupil_center"
                    )
                )
        if getattr(self.args, "no_behavior", False):
            transforms.append(
                ZeroBehavior(
                    behavior_name="behavior", pupil_center_name="pupil_center"
                )
            )

        transforms.append(ToTensor(getattr(self.args, "device", "cpu") == "cuda"))
        dataset.transforms = transforms


class WhiteNoiseLoader(LoaderBase):
    """Data loader for white noise stimulus datasets."""

    def resolve_transforms(self, dataset: MovieFileTreeDataset, tier: str):
        """Configures transforms for white noise datasets."""
        transforms = [AddFrames("videos", getattr(self.args, "frames", 70))]

        # Augmentations
        if getattr(self.args, "random_noise", False):
            transforms.append(
                RandomNoise(
                    noise_level=self.alpha_noise * self.noise_level,
                    in_name="videos",
                    frame_axis=-1,
                    per_frames=getattr(self.args, "augment_per_frame", False),
                )
            )
        if getattr(self.args, "augment_data", False):
            transforms.append(
                ApplyAffine(
                    degrees=(
                        -self.alpha_affine * self.rotation_max,
                        self.alpha_affine * self.rotation_max,
                    ),
                    translate=(
                        self.alpha_affine * self.translation_max,
                        self.alpha_affine * self.translation_max,
                    ),
                    scale=(
                        1 - self.alpha_affine * self.scale_max,
                        1 + self.alpha_affine * self.scale_max,
                    ),
                    in_name="videos",
                    frame_axis=-1,
                    per_frames=getattr(self.args, "augment_per_frame", False),
                )
            )

        # Add gray screen periods
        if getattr(self.args, "gray_screen_frames", 0) > 0:
            transforms.append(
                GrayScreenFrames(
                    dataset, "videos", getattr(self.args, "gray_screen_frames", 0)
                )
            )
        if getattr(self.args, "ending_gray_screen_frames", 0) > 0:
            transforms.append(
                GrayScreenFrames(
                    dataset,
                    "videos",
                    getattr(self.args, "ending_gray_screen_frames", 0),
                    start=False,
                )
            )

        # Normalization and shaping
        if not getattr(self.args, "resnet", False):
            transforms.append(
                NeuroNormalizer(
                    dataset,
                    stats_source=self.input_type,
                    exclude=["responses", "pupil_center", "behavior"],
                    input_stats=self.input_stats,
                    in_name="videos",
                )
            )
        transforms.append(ChangeChannelsOrder((2, 0, 1), in_name="videos"))
        transforms.append(ExpandChannels("videos"))
        if getattr(self.args, "resnet", False):
            transforms.append(RepeatChannels(3, "videos", channel_axis=0))

        # Handle behavioral and pupil data
        use_modulator = getattr(self.args, "modulator", False)
        if getattr(self.args, "include_behavior", False) and not use_modulator:
            transforms.append(AddBehaviorAsChannels("videos"))
        if getattr(self.args, "include_pupil_centers_as_channels", False) and not use_modulator:
            transforms.append(AddPupilCenterAsChannels("videos"))

        if use_modulator:
            assert getattr(
                self.args, "include_behavior", False
            ), "Behavior must be included for modulator"
            if getattr(self.args, "include_pupil_centers_as_channels", False):
                transforms.append(
                    AddPupilCenterToBehavior(
                        behavior_name="behavior", pupil_center_name="pupil_center"
                    )
                )
            transforms.append(
                AddFrames("behavior", getattr(self.args, "frames", 300), axis=-1)
            )
        if getattr(self.args, "include_pupil_centers", False):
            transforms.append(
                AddFrames("pupil_center", getattr(self.args, "frames", 70), axis=-1)
            )

        transforms.append(ToTensor(getattr(self.args, "device", "cpu") == "cuda"))
        dataset.transforms = transforms


class ObjectsVideoLoader(LoaderBase):
    """Data loader for object video datasets."""

    def resolve_transforms(self, dataset: MovieFileTreeDataset, tier: str):
        """Configures transforms for object video datasets."""
        transforms = [ResizeInputs(output_shape=(36, 64), in_name="videos")]

        # Normalization and shaping
        if not getattr(self.args, "resnet", False):
            transforms.append(
                NeuroNormalizer(
                    dataset,
                    stats_source=self.input_type,
                    exclude=["responses", "pupil_center", "behavior"],
                    input_stats=self.input_stats,
                    in_name="videos",
                )
            )
        transforms.append(ChangeChannelsOrder((2, 0, 1), in_name="videos"))
        transforms.append(ExpandChannels("videos"))
        if getattr(self.args, "resnet", False):
            transforms.append(RepeatChannels(3, "videos", channel_axis=0))

        # Handle behavioral and pupil data
        use_modulator = getattr(self.args, "modulator", False)
        if getattr(self.args, "include_behavior", False):
            transforms.append(ChangeChannelsOrder((1, 0), in_name="behavior"))
            if not use_modulator:
                transforms.append(AddBehaviorAsChannels("videos"))
        if getattr(self.args, "include_pupil_centers", False):
            transforms.append(ChangeChannelsOrder((1, 0), in_name="pupil_center"))
        if getattr(self.args, "include_pupil_centers_as_channels", False) and not use_modulator:
            transforms.append(AddPupilCenterAsChannels("videos"))

        if use_modulator:
            assert getattr(
                self.args, "include_behavior", False
            ), "Behavior must be included for modulator"
            if getattr(self.args, "include_pupil_centers_as_channels", False):
                transforms.append(
                    AddPupilCenterToBehavior(
                        behavior_name="behavior", pupil_center_name="pupil_center"
                    )
                )
            transforms.append(
                AddFrames("behavior", getattr(self.args, "frames", 300), axis=-1)
            )

        transforms.append(ToTensor(getattr(self.args, "device", "cpu") == "cuda"))
        dataset.transforms = transforms

class TowardsLoaderBase(LoaderBase):
    """Base data loader for the 'Towards' dataset."""
    def __init__(self, batch_size, args, input_type='all', input_stats=None, neurons_subset=None):
        super().__init__(batch_size, args, input_type, input_stats, neurons_subset)
        self.input_type = input_type

    def resolve_transforms(self, dataset, tier):
        """Configures transforms for the 'Towards' dataset."""
        dataset.transforms = []
        transforms = []
        if self.input_type == 'images':
            transforms.append(
                AddFrames("videos", getattr(self.args, 'frames', 70))
            )
        if getattr(self.args, 'random_noise', False):
            transforms.append(
                RandomNoise(
                    noise_level=self.alpha_noise*self.noise_level,
                    in_name='videos',
                    frame_axis=-1,
                    per_frames=getattr(self.args, 'augment_per_frame', False)
                )
            )
        if getattr(self.args, 'augment_data', False):
            transforms.append(
                ApplyAffine(
                    degrees=(-self.alpha_affine*self.rotation_max, self.alpha_affine*self.rotation_max), 
                    translate=(self.alpha_affine*self.translation_max, self.alpha_affine*self.translation_max), 
                    scale=(1-self.alpha_affine*self.scale_max, 1+self.alpha_affine*self.scale_max),
                    in_name='videos',
                    frame_axis=-1,
                    per_frames=getattr(self.args, 'augment_per_frame', False)
                )
            )
        if getattr(self.args, 'gray_screen_frames', 0) > 0:
            transforms.append(
                GrayScreenFrames(
                    dataset, 
                    "videos", 
                    getattr(self.args, 'gray_screen_frames', 0)
                )
            )
        if getattr(self.args, 'ending_gray_screen_frames', 0) > 0:
            transforms.append(
                GrayScreenFrames(
                    dataset, 
                    "videos", 
                    getattr(self.args, 'ending_gray_screen_frames', 0), 
                    start=False
                )
            )
        transforms.append(
            ChangeChannelsOrder((2, 0, 1), in_name="videos")
        )
        dataset.transforms.extend(transforms)

class TowardsLoader(Microns30Loader):
    """Data loader for the 'Towards' dataset, combining features from others."""

    def __init__(
        self,
        batch_size: int,
        args: t.Any,
        input_type: str = "all",
        input_stats: t.Optional[t.Dict] = None,
        neurons_subset: t.Optional[t.Dict] = None,
    ):
        super().__init__(batch_size, args, input_type, input_stats, neurons_subset)
        self.cutvideos = self.input_type in ["all", "clips"]

    def resolve_transforms(self, dataset: MovieFileTreeDataset, tier: str):
        """Configures transforms for the 'Towards' dataset."""
        dataset_name = dataset.dirname.split("/")[-2]
        neurons_idxs = self._get_neuron_indices(dataset, dataset_name)

        transforms = []
        if self.cutvideos:
            transforms.append(
                CutVideos(
                    max_frame=getattr(self.args, "max_frame", 300),
                    frame_axis={data_key: -1 for data_key in self.data_keys},
                    target_groups=self.data_keys,
                )
            )

        # Training-only augmentations
        if tier == "train":
            if getattr(self.args, "random_noise", False):
                transforms.append(
                    RandomNoise(
                        noise_level=self.alpha_noise * self.noise_level,
                        in_name="videos",
                    )
                )
            if getattr(self.args, "augment_data", False):
                transforms.append(
                    ApplyAffine(
                        degrees=(
                            -self.alpha_affine * self.rotation_max,
                            self.alpha_affine * self.rotation_max,
                        ),
                        translate=(
                            self.alpha_affine * self.translation_max,
                            self.alpha_affine * self.translation_max,
                        ),
                        scale=(
                            1 - self.alpha_affine * self.scale_max,
                            1 + self.alpha_affine * self.scale_max,
                        ),
                        in_name="videos",
                        frame_axis=-1,
                        per_frames=getattr(self.args, "augment_per_frame", False),
                    )
                )

        transforms.append(Subsample(neurons_idxs, target_index=0))
        if getattr(self.args, "normalize", []):
            transforms.append(
                NeuroNormalizer(
                    dataset,
                    stats_source=self.input_type,
                    exclude=getattr(self.args, "exclude", ["videos"]),
                    input_stats=self.input_stats,
                    in_name="videos",
                )
            )

        # Final tensor formatting
        transforms.append(ExpandChannels("videos"))
        transforms.append(ChangeChannelsOrder((3, 1, 2, 0), in_name="videos"))
        transforms.append(ChangeChannelsOrder((1, 0), in_name="responses"))
        if getattr(self.args, "include_behavior", False):
            transforms.append(ChangeChannelsOrder((1, 0), in_name="behavior"))
        if getattr(self.args, "include_pupil_centers", False):
            transforms.append(ChangeChannelsOrder((1, 0), in_name="pupil_center"))

        dataset.transforms = transforms


class ObjectRotationsVideoLoader(LoaderBase):
    """Data loader for object rotation video datasets."""

    def resolve_transforms(self, dataset: MovieFileTreeDataset, tier: str):
        """Configures transforms for object rotation datasets."""
        transforms = [
            ResizeInputs(output_shape=(36, 64), in_name="videos"),
            AddFrames("videos", getattr(self.args, "frames", 65)),
        ]

        # Augmentations
        if getattr(self.args, "random_noise", False):
            transforms.append(
                RandomNoise(
                    noise_level=self.alpha_noise * self.noise_level,
                    in_name="videos",
                    frame_axis=-1,
                    per_frames=getattr(self.args, "augment_per_frame", False),
                )
            )
        if getattr(self.args, "augment_data", False):
            transforms.append(
                ApplyAffine(
                    degrees=(
                        -self.alpha_affine * self.rotation_max,
                        self.alpha_affine * self.rotation_max,
                    ),
                    translate=(
                        self.alpha_affine * self.translation_max,
                        self.alpha_affine * self.translation_max,
                    ),
                    scale=(
                        1 - self.alpha_affine * self.scale_max,
                        1 + self.alpha_affine * self.scale_max,
                    ),
                    in_name="videos",
                    frame_axis=-1,
                    per_frames=getattr(self.args, "augment_per_frame", False),
                )
            )

        # Add gray screen periods
        if getattr(self.args, "gray_screen_frames", 50) > 0:
            transforms.append(
                GrayScreenFrames(
                    dataset, "videos", getattr(self.args, "gray_screen_frames", 0)
                )
            )
        if getattr(self.args, "ending_gray_screen_frames", 0) > 0:
            transforms.append(
                GrayScreenFrames(
                    dataset,
                    "videos",
                    getattr(self.args, "ending_gray_screen_frames", 0),
                    start=False,
                )
            )

        # Normalization and shaping
        if not getattr(self.args, "resnet", False):
            transforms.append(
                NeuroNormalizer(
                    dataset,
                    stats_source=self.input_type,
                    exclude=["responses", "pupil_center", "behavior"],
                    input_stats=self.input_stats,
                    in_name="videos",
                )
            )
        transforms.append(ChangeChannelsOrder((2, 0, 1), in_name="videos"))
        transforms.append(ExpandChannels("videos"))
        if getattr(self.args, "resnet", False):
            transforms.append(RepeatChannels(3, "videos", channel_axis=0))

        # Handle behavioral and pupil data
        use_modulator = getattr(self.args, "modulator", False)
        if getattr(self.args, "include_behavior", False) and not use_modulator:
            transforms.append(AddBehaviorAsChannels("videos"))
        if getattr(self.args, "include_pupil_centers_as_channels", False) and not use_modulator:
            transforms.append(AddPupilCenterAsChannels("videos"))

        if use_modulator:
            assert getattr(
                self.args, "include_behavior", False
            ), "Behavior must be included for modulator"
            if getattr(self.args, "include_pupil_centers_as_channels", False):
                transforms.append(
                    AddPupilCenterToBehavior(
                        behavior_name="behavior", pupil_center_name="pupil_center"
                    )
                )
            transforms.append(
                AddFrames("behavior", getattr(self.args, "frames", 300), axis=-1)
            )
        if getattr(self.args, "include_pupil_centers", False):
            transforms.append(
                AddFrames("pupil_center", getattr(self.args, "frames", 70), axis=-1)
            )

        transforms.append(ToTensor(getattr(self.args, "device", "cpu") == "cuda"))
        dataset.transforms = transforms
