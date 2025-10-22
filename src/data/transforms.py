import numpy as np
import torch
from neuralpredictors.data.transforms import (Invertible, MovieTransform,
                                              StaticTransform)
from scipy.interpolate import interp1d
from skimage.transform import resize
from torchvision.transforms import RandomAffine
from torchvision.transforms.functional import to_pil_image


class MissingValuesHandler:
    """A collection of static methods for handling NaN values in arrays."""

    @staticmethod
    def interpolate_linearly(x: np.ndarray,
                             axis: int = -1,
                             fill_value="extrapolate") -> np.ndarray:
        """
        Linearly interpolates over NaN values in a 1D slice of an array.

        Args:
            x: Input array with potential NaN values.
            axis: The axis along which to interpolate.
            fill_value: Value to use for out-of-range interpolation.

        Returns:
            The array with NaN values interpolated.
        """
        not_nan = np.logical_not(np.isnan(x))
        indices = np.arange(x.shape[axis])
        interp = interp1d(indices[not_nan],
                          x[not_nan],
                          axis=axis,
                          fill_value=fill_value)
        return interp(indices).astype(np.float32)

    @staticmethod
    def has_n_consecutive_nans(x: np.ndarray, n: int) -> bool:
        """
        Checks if an array contains at least n consecutive NaN values.

        Args:
            x: Input array.
            n: The number of consecutive NaNs to check for.

        Returns:
            True if n consecutive NaNs are found, False otherwise.
        """
        consecutive_counts = np.convolve(np.isnan(x),
                                         np.ones(n, dtype=np.int8),
                                         mode="valid")
        return np.any(consecutive_counts >= n)

    def num_consecutive_nans(self, x: np.ndarray, n: int = 1) -> int:
        """
        Recursively finds the maximum number of consecutive NaNs in an array.

        Args:
            x: Input array.
            n: Current count of consecutive NaNs (for recursion).

        Returns:
            The maximum number of consecutive NaNs.
        """
        if self.has_n_consecutive_nans(x, n):
            return self.num_consecutive_nans(x, n + 1)
        return n - 1

    @staticmethod
    def nan_ratio(x: np.ndarray) -> float:
        """
        Calculates the ratio of NaN values in an array.

        Args:
            x: Input array.

        Returns:
            The fraction of elements that are NaN.
        """
        return np.sum(np.isnan(x)) / x.size

    def consecutive_nans_per_trial(self, trials: list) -> list:
        """
        Calculates the max consecutive NaNs for each trial in a list.

        Args:
            trials: A list of trial arrays.

        Returns:
            A list containing the max number of consecutive NaNs for each trial.
        """
        output = []
        for trial in trials:
            nans = np.apply_along_axis(lambda x: self.num_consecutive_nans(x),
                                       -1, trial)
            output.append(nans.max())
        return output


class Upsampler:
    """Upsamples temporal data to a target number of frames."""

    def __init__(self, target_frames_clip: int,
                 target_frames_parametric: int) -> None:
        """
        Initializes the Upsampler with target frame counts.

        Args:
            target_frames_clip: Target number of frames for 'Clip' type trials.
            target_frames_parametric: Target number of frames for other trials.
        """
        self.target_frames_clip = target_frames_clip
        self.target_frames_parametric = target_frames_parametric

    def upsample_trials(self, data: list, trial_types: list) -> list:
        """
        Upsamples a list of trials to a target number of frames via interpolation.

        Args:
            data: List of arrays, where each array is a trial.
            trial_types: List of strings indicating the type of each trial.

        Returns:
            A list of upsampled trial arrays.
        """
        assert len(data) == len(
            trial_types), "Data and trial_types lists must have the same length"
        data_interpolated = []

        for trial, clip_type in zip(data, trial_types):
            n_frames = trial.shape[1]
            target_frames = (self.target_frames_clip
                             if clip_type in ["Clip", "black"] else
                             self.target_frames_parametric)

            f_interpol = interp1d(np.arange(0, n_frames),
                                  trial,
                                  axis=1,
                                  fill_value="extrapolate")
            interpolated = f_interpol(
                np.linspace(0, n_frames - 1, target_frames))
            data_interpolated.append(interpolated.astype(np.float32))

        return data_interpolated


class ResizeInputs(StaticTransform, Invertible, MovieTransform):
    """
    Resizes images or video frames using skimage.transform.resize.

    Applies the transformation to the data field specified by `in_name`.
    """

    def __init__(
        self,
        output_shape,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
        clip=True,
        in_name="images",
    ):
        self.output_shape = output_shape
        self.mode = mode
        self.anti_aliasing = anti_aliasing
        self.preserve_range = preserve_range
        self.clip = clip
        self.in_name = in_name

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals[self.in_name]
        key_vals[self.in_name] = resize(
            img,
            output_shape=self.output_shape,
            mode=self.mode,
            anti_aliasing=self.anti_aliasing,
            clip=self.clip,
            preserve_range=self.preserve_range,
        )
        return x.__class__(**key_vals)


class PerImageNormalizer(StaticTransform, Invertible, MovieTransform):
    """
    Normalizes each image or video frame by its own mean and standard deviation.

    Applies the transformation to the data field specified by `in_name`.
    """

    def __init__(self, in_name="videos", eps: float = 1e-6):
        self.in_name = in_name
        self.eps = eps

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals[self.in_name]
        if not np.all(img == 0.0):
            mean = img.mean(axis=(0, 1), keepdims=True)
            std = img.std(axis=(0, 1), keepdims=True)
            key_vals[self.in_name] = (img - mean) / (std + self.eps)
        return x.__class__(**key_vals)


class NeuroNormalizer(MovieTransform, StaticTransform, Invertible):
    """
    Normalizes a data sample containing neural, behavioral, and eye data.

    This transform applies specific normalizations to different data fields
    based on pre-computed statistics from a dataset object.
    - 'inputs': Centered on movie mean, scaled by training std.
    - 'behavior': Scaled by std.
    - 'eye_position': Z-scored.
    - 'responses': Scaled by per-neuron std.
    """

    def __init__(
        self,
        data,
        stats_source="all",
        exclude=None,
        input_stats=None,
        subtract_behavior_mean=True,
        in_name=None,
        out_name=None,
        eye_name=None,
    ):
        self.session_key = data.dirname.split("/")[-2]
        self.exclude = exclude or []

        in_name = in_name or ("images"
                              if "images" in data.statistics else "inputs")
        out_name = out_name or ("responses" if "responses" in data.statistics
                                else "targets")
        eye_name = eye_name or ("pupil_center"
                                if "pupil_center" in data.data_keys else
                                "eye_position")

        if input_stats is not None:
            self._load_stats_from_dict(input_stats, in_name, out_name,
                                       eye_name, subtract_behavior_mean)
        else:
            self._load_stats_from_dataset(data, stats_source, in_name,
                                          out_name, eye_name,
                                          subtract_behavior_mean)

        self._initialize_transforms(in_name, out_name, eye_name, data.data_keys)

    def _load_stats_from_dict(self, input_stats, in_name, out_name, eye_name,
                              subtract_behavior_mean):
        """Helper to load statistics from a provided dictionary."""
        if in_name not in self.exclude:
            self._inputs_mean = input_stats[in_name]["mean"]
            self._inputs_std = input_stats[in_name]["std"]
        else:
            self._inputs_mean, self._inputs_std = 0, 1

        if out_name not in self.exclude:
            self._response_mean = input_stats[self.session_key][out_name]["mean"]
            self._response_std = input_stats[self.session_key][out_name]["std"]
            self._calculate_response_precision()
        else:
            self._response_mean, self._response_std = 0, 1

        if eye_name not in self.exclude and eye_name in input_stats[
                self.session_key]:
            self._eye_mean = input_stats[self.session_key][eye_name]["mean"]
            self._eye_std = input_stats[self.session_key][eye_name]["std"]
        else:
            self._eye_mean, self._eye_std = 0, 1

        if "behavior" not in self.exclude and "behavior" in input_stats[
                self.session_key]:
            self.behavior_mean = (0 if not subtract_behavior_mean else
                                  input_stats[self.session_key]["behavior"]["mean"])
            self._behavior_precision = 1 / (
                input_stats[self.session_key]["behavior"]["std"] + 1e-8)
        else:
            self.behavior_mean, self._behavior_precision = 0, 1

    def _load_stats_from_dataset(self, data, stats_source, in_name, out_name,
                                 eye_name, subtract_behavior_mean):
        """Helper to load statistics from the dataset object."""
        self._inputs_mean = data.statistics[in_name][stats_source]["mean"][()]
        self._inputs_std = data.statistics[in_name][stats_source]["std"][()]

        self._response_std = np.array(
            data.statistics[out_name][stats_source]["std"])
        self._response_mean = np.array(
            data.statistics[out_name][stats_source]["mean"])
        self._calculate_response_precision()

        if eye_name in data.data_keys:
            self._eye_mean = data.statistics[eye_name][stats_source]["mean"][()]
            self._eye_std = data.statistics[eye_name][stats_source]["std"][()]

        if "behavior" in data.data_keys:
            s = np.array(data.statistics["behavior"][stats_source]["std"])
            self.behavior_mean = (0 if not subtract_behavior_mean else np.array(
                data.statistics["behavior"][stats_source]["mean"]))
            self._behavior_precision = 1 / (s + 1e-8)

    def _calculate_response_precision(self):
        """Calculates response precision, avoiding division by small stds."""
        threshold = 0.01 * np.nanmean(self._response_std)
        idx = self._response_std > threshold
        self._response_precision = np.ones_like(self._response_std) / (
            threshold + 1e-8)
        self._response_precision[idx] = 1 / (self._response_std[idx] + 1e-8)

    def _initialize_transforms(self, in_name, out_name, eye_name, data_keys):
        """Initializes forward and inverse transformation functions."""
        transforms, itransforms = {}, {}

        transforms[in_name] = lambda x: (x - self._inputs_mean) / (
            self._inputs_std + 1e-8)
        itransforms[in_name] = lambda x: x * self._inputs_std + self._inputs_mean

        transforms[out_name] = lambda x: np.maximum(x * self._response_precision,
                                                    0)
        itransforms[out_name] = lambda x: x / self._response_precision

        if eye_name in data_keys:
            transforms[eye_name] = lambda x: (x - self._eye_mean) / (
                self._eye_std + 1e-8)
            itransforms[eye_name] = lambda x: x * self._eye_std + self._eye_mean

        if "behavior" in data_keys:
            transforms["behavior"] = lambda x: (
                x - self.behavior_mean) * self._behavior_precision
            itransforms["behavior"] = (
                lambda x: x / self._behavior_precision + self.behavior_mean)

        self._transforms = transforms
        self._itransforms = itransforms

    def __call__(self, x):
        return x.__class__(**{
            k: (self._transforms.get(k, lambda v: v)(v)
                if k not in self.exclude else v)
            for k, v in zip(x._fields, x)
        })

    def inv(self, x):
        return x.__class__(**{
            k: (self._itransforms.get(k, lambda v: v)(v)
                if k not in self.exclude else v)
            for k, v in zip(x._fields, x)
        })

    def __repr__(self):
        excluded_str = (f"(not {', '.join(self.exclude)})"
                        if self.exclude else "")
        return super().__repr__() + excluded_str


class ExpandSelectedChannels(MovieTransform, StaticTransform, Invertible):
    """
    Ensures a video tensor has a channel dimension.

    If the input tensor (e.g., a grayscale video) lacks a channel dimension,
    this transform adds a singleton dimension to conform to a standard shape
    (e.g., T x C x H x W).
    """

    def __init__(self, key: str):
        if key not in ["videos", "images"]:
            raise ValueError("The provided key must be either 'videos' or 'images'")
        self.key = key

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals[self.key]

        if len(img.shape) != 4 and self.key == "videos":
            # Add channel dimension for videos: (T, H, W) -> (T, 1, H, W)
            img = np.expand_dims(img, axis=1)
        key_vals[self.key] = img
        return x.__class__(**key_vals)


class AddFrames(MovieTransform, StaticTransform):
    """Duplicates the single frame of an image to create a video of n_frames."""

    def __init__(self, key: str, n_frames: int, axis: int = -1):
        self.key = key
        self.n_frames = n_frames
        self.axis = axis

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals[self.key].squeeze()
        key_vals[self.key] = np.stack([img for _ in range(self.n_frames)],
                                      axis=self.axis)
        return x.__class__(**key_vals)


class GrayScreenFrames(MovieTransform, StaticTransform):
    """Replaces start or end frames of a video with gray or random noise."""

    def __init__(self,
                 key: str,
                 n_frames: int,
                 frame_type: str = "gray",
                 start: bool = True):
        if key not in ["videos", "images"]:
            raise ValueError("The provided key must be either 'videos' or 'images'")
        self.key = key
        self.n_frames = n_frames
        self.frame_type = frame_type
        self.start = start

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals[self.key]

        if self.frame_type == "gray":
            intensity = np.ones_like(img[..., :self.n_frames]) * 128
        elif self.frame_type == "random":
            intensity = np.random.randn(*img[..., :self.n_frames].shape)
        else:
            raise ValueError("frame_type must be 'gray' or 'random'")

        if self.start:
            img[..., :self.n_frames] = intensity
        else:
            img[..., -self.n_frames:] = intensity

        key_vals[self.key] = img
        return x.__class__(**key_vals)


class PerturbBehavior(MovieTransform):
    """Perturbs a specific channel of a data field."""

    def __init__(self, key: str, channel: int, p_type: str = "static", value=0):
        self.key = key
        self.channel = channel
        self.p_type = p_type
        self.value = value

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals[self.key]

        if self.p_type == "static":
            img[self.channel, ...] = self.value
        elif self.p_type == "random":
            frames = img.shape[1]
            rand_values = np.random.randn(frames)
            rand_values = np.expand_dims(rand_values, axis=(1, 2))
            shaper = np.ones(img.shape[1:])
            img[self.channel, ...] = rand_values * shaper
        else:
            raise ValueError("p_type must be 'static' or 'random'")

        key_vals[self.key] = img
        return x.__class__(**key_vals)


class RemoveSpontActivity(MovieTransform, StaticTransform):
    """
    Removes spontaneous activity from neural responses via PCA projection.

    Projects responses onto a subspace of spontaneous activity (pre-computed PCs)
    and subtracts this projection.
    """

    def __init__(self, data, n_components: int):
        self.n_components = n_components
        self.components = data.statistics["responses"]["all"]["spontPCs"][()]
        self.mean_spont = data.statistics["responses"]["spontaneous"]["mean_post"][()]
        self.std_spont = data.statistics["responses"]["spontaneous"]["std_post"][()]

    def _transform_responses(self, responses):
        """
        Applies the spontaneous activity removal.

        Args:
            responses (np.ndarray): Neurons x Frames array.

        Returns:
            np.ndarray: Responses with spontaneous activity removed.
        """
        us_pont = self.components[:self.n_components, :]
        normalized_responses = (responses - self.mean_spont) / self.std_spont
        proj = np.dot(np.dot(normalized_responses.T, us_pont.T), us_pont)
        removed_spont = normalized_responses - proj.T
        return removed_spont - removed_spont.mean(axis=1, keepdims=True)

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        if "responses" in key_vals:
            key_vals["responses"] = self._transform_responses(
                key_vals["responses"])
        return x.__class__(**key_vals)


class ApplyAffine(MovieTransform):
    """
    Applies random affine transformations to video frames.

    Uses torchvision.transforms.RandomAffine. Can apply the same transform
    to all frames or a different one to each frame.
    """

    def __init__(self,
                 degrees=0,
                 translate=None,
                 scale=None,
                 shear=None,
                 in_name="videos",
                 frame_axis=-1,
                 per_frame=False):
        self.affine_transform = RandomAffine(degrees=degrees,
                                             translate=translate,
                                             scale=scale,
                                             shear=shear)
        self.in_name = in_name
        self.frame_axis = frame_axis
        self.per_frame = per_frame

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        video = key_vals[self.in_name]
        video = np.moveaxis(video, self.frame_axis, 0)

        if self.per_frame:
            transformed_frames = []
            for frame in video:
                pil_frame = to_pil_image(frame)
                transformed_frame = self.affine_transform(pil_frame)
                transformed_frames.append(np.array(transformed_frame))
            transformed_video = np.stack(transformed_frames, axis=0)
        else:
            video_tensor = torch.tensor(video, dtype=torch.float32)
            transformed_tensor = self.affine_transform(video_tensor)
            transformed_video = transformed_tensor.numpy()

        transformed_video = np.moveaxis(transformed_video, 0, self.frame_axis)
        key_vals[self.in_name] = transformed_video
        return x.__class__(**key_vals)


class RandomNoise(MovieTransform):
    """Adds Gaussian noise to video frames."""

    def __init__(self,
                 noise_level=1.0,
                 in_name="videos",
                 frame_axis=-1,
                 per_frame=False,
                 deterministic=False):
        self.noise_level = noise_level
        self.in_name = in_name
        self.frame_axis = frame_axis
        self.per_frame = per_frame
        self.deterministic = deterministic

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        video = key_vals[self.in_name]

        if self.deterministic:
            np.random.seed(0)

        if self.per_frame:
            noise = np.random.normal(0, self.noise_level, video.shape)
            transformed_video = video + noise
        else:
            video = np.moveaxis(video, self.frame_axis, 0)
            _, h, w = video.shape
            noise = np.random.normal(0, self.noise_level, (1, h, w))
            transformed_video = video + noise
            transformed_video = np.moveaxis(transformed_video, 0,
                                            self.frame_axis)

        key_vals[self.in_name] = np.clip(transformed_video, 0, 255)
        return x.__class__(**key_vals)


class RepeatChannels(MovieTransform):
    """Repeats the channels of a video or image n times."""

    def __init__(self, n_channels=3, in_name="videos", channel_axis=-1):
        self.n_channels = n_channels
        self.in_name = in_name
        self.channel_axis = channel_axis

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        video = key_vals[self.in_name]
        key_vals[self.in_name] = np.repeat(video,
                                           self.n_channels,
                                           axis=self.channel_axis)
        return x.__class__(**key_vals)


class Pad(MovieTransform):
    """Pads video frames along the height and width dimensions."""

    def __init__(self, pad_size: int, height_axis: int):
        self.pad_size = pad_size
        self.height_axis = height_axis

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        video = key_vals["videos"]
        padding = [(0, 0)] * video.ndim
        padding[self.height_axis] = (self.pad_size, self.pad_size)
        key_vals["videos"] = np.pad(video,
                                    padding,
                                    mode="constant",
                                    constant_values=0)
        return x.__class__(**key_vals)


class AddPupilCenterToBehavior(MovieTransform):
    """Concatenates pupil center data to the behavior data."""

    def __init__(self,
                 behavior_name="behavior",
                 pupil_center_name="pupil_center",
                 axis=0):
        self.behavior_name = behavior_name
        self.pupil_center_name = pupil_center_name
        self.axis = axis

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        behavior = key_vals[self.behavior_name]
        pupil_center = key_vals[self.pupil_center_name]
        key_vals[self.behavior_name] = np.concatenate(
            [behavior, pupil_center], axis=self.axis)
        return x.__class__(**key_vals)


class ZeroBehavior(MovieTransform):
    """Sets behavior and pupil center data to zero."""

    def __init__(self,
                 behavior_name="behavior",
                 pupil_center_name="pupil_center"):
        self.behavior_name = behavior_name
        self.pupil_center_name = pupil_center_name

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        key_vals[self.behavior_name] = np.zeros_like(key_vals[self.behavior_name])
        key_vals[self.pupil_center_name] = np.zeros_like(
            key_vals[self.pupil_center_name])
        return x.__class__(**key_vals)