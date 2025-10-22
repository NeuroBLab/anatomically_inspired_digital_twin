import numpy as np
import torch

# ##############################################################################
# NumPy Implementations (CPU)
# ##############################################################################


def pearson_correlation(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """Computes Pearson correlation between two NumPy arrays.

    Args:
        y1 (np.ndarray): The first array.
        y2 (np.ndarray): The second array, must have the same shape as y1.
        axis (int or tuple[int]): The axis or axes along which to compute.
        eps (float): A small value to add to the standard deviation for
            numerical stability.
        **kwargs: Additional keyword arguments passed to the final `np.mean`.

    Returns:
        np.ndarray: The Pearson correlation coefficient.
    """
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (
        y1.std(axis=axis, keepdims=True, ddof=0) + eps
    )
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (
        y2.std(axis=axis, keepdims=True, ddof=0) + eps
    )
    return (y1 * y2).mean(axis=axis, **kwargs)


def oracle_explainable_variance(response, eps=1e-9):
    """Computes the oracle explainable variance per neuron.

    This metric estimates the fraction of variance in a neural response that is
    explainable, given the intrinsic noise across repeated trials.

    Args:
        response (np.ndarray): An array of neural responses with shape
            (trials, repeats, neurons, frames).
        eps (float): A small value for numerical stability.

    Returns:
        np.ndarray: The explainable variance for each neuron.
    """
    total_var = np.var(response, axis=(0, 1, 3), ddof=1)
    noise_var = np.mean(np.var(response, axis=1, ddof=1), axis=(0, 2))
    return (total_var - noise_var) / (total_var + eps)


def fraction_variance_explained(response, prediction):
    """Computes the fraction of variance explained (FVE).

    Args:
        response (np.ndarray): The ground-truth neural responses, with shape
            (trials, repeats, neurons, frames).
        prediction (np.ndarray): The model's predicted responses, with the
            same shape as `response`.

    Returns:
        np.ndarray: The fraction of variance explained for each neuron.
    """
    mse = np.mean((response - prediction) ** 2, axis=(0, 1, 3))
    total_var = np.var(response, axis=(0, 1, 3), ddof=1)
    return 1 - mse / total_var


def fraction_explainable_variance(response):
    """Computes the fraction of explainable variance (FEV).

    Args:
        response (np.ndarray): The ground-truth neural responses, with shape
            (trials, repeats, neurons, frames).

    Returns:
        np.ndarray: The fraction of explainable variance for each neuron.
    """
    total_var = np.var(response, axis=(0, 1, 3), ddof=1)
    noise_var = np.mean(np.var(response, axis=1, ddof=1), axis=(0, 2))
    return (total_var - noise_var) / total_var


def fraction_explainable_variance_explained(response, prediction):
    """Computes the fraction of explainable variance explained (FEVe).

    Args:
        response (np.ndarray): The ground-truth neural responses, with shape
            (trials, repeats, neurons, frames).
        prediction (np.ndarray): The model's predicted responses, with the
            same shape as `response`.

    Returns:
        np.ndarray: The FEVe for each neuron.
    """
    mse = np.mean((response - prediction) ** 2, axis=(0, 1, 3))
    total_var = np.var(response, axis=(0, 1, 3), ddof=1)
    noise_var = np.mean(np.var(response, axis=1, ddof=1), axis=(0, 2))
    return 1 - (mse - noise_var) / (total_var - noise_var)


def oracle_correlation(repeated_responses, eps=1e-8, axis=(0, 2)):
    """Computes the oracle correlation (CCmax).

    This is the theoretical maximum correlation achievable by any model, given
    the noise in the data.

    Args:
        repeated_responses (np.ndarray): An array of neural responses with
            shape (trials, repeats, neurons, frames).
        eps (float): A small value for numerical stability.
        axis (tuple[int]): The axes along which to compute the variance.

    Returns:
        np.ndarray: The oracle correlation for each neuron.
    """
    num_repeats = repeated_responses.shape[1]
    # The axes for variance calculation depend on the structure of `axis`.
    var_axis = (0, 3) if len(axis) == 2 else 3
    noise_var_axis = 0 if len(axis) == 2 else 1

    signal_var = np.var(repeated_responses.mean(axis=1), axis=axis, ddof=1)
    total_var_per_repeat = np.var(repeated_responses, axis=var_axis, ddof=1)
    noise_var = np.mean(total_var_per_repeat, axis=noise_var_axis)

    return np.sqrt(
        (num_repeats * signal_var - noise_var)
        / ((num_repeats - 1) * signal_var + eps)
    )


def normalized_correlation(response, prediction, eps=1e-8, axis=(0, 2)):
    """Computes the correlation normalized by the oracle correlation.

    Args:
        response (np.ndarray): The ground-truth neural responses, with shape
            (trials, repeats, neurons, frames).
        prediction (np.ndarray): The model's predicted responses, with the
            same shape as `response`.
        eps (float): A small value for numerical stability.
        axis (tuple[int]): The axes along which to compute the correlation.

    Returns:
        np.ndarray: The normalized correlation for each neuron.
    """
    response_mean = response.mean(1)
    prediction_mean = prediction.mean(1)
    cc_abs = pearson_correlation(response_mean, prediction_mean, axis=axis, eps=eps)
    cc_max = oracle_correlation(response, eps=eps, axis=axis)
    return cc_abs / cc_max


def mean_squared_error(response, prediction, normalize=True):
    """Computes the mean squared error (MSE).

    Args:
        response (np.ndarray): The ground-truth array.
        prediction (np.ndarray): The predicted array.
        normalize (bool): If True, returns the mean; otherwise, the sum.

    Returns:
        float: The mean or sum of the squared error.
    """
    loss = (response - prediction) ** 2
    return loss.mean() if normalize else loss.sum()


def cross_correlation_mse(output, target, axis=0, normalize=True):
    """Computes MSE between the cross-correlation matrices of two signals.

    Args:
        output (np.ndarray): The predicted signal array, with shape
            (neurons, time_bins).
        target (np.ndarray): The ground-truth signal array, with the same
            shape as `output`.
        axis (int): The axis representing the observation dimension (e.g., time).
        normalize (bool): If True, returns the mean; otherwise, the sum.

    Returns:
        float: The MSE between the two correlation matrices.
    """
    target_copy = target - np.mean(target, axis=axis, keepdims=True)
    target_cov = np.matmul(target_copy, target_copy.T) / (target.shape[axis] - 1)
    target_std = np.sqrt(np.diag(target_cov)).reshape(-1, 1)
    target_corr = target_cov / np.matmul(target_std, target_std.T)

    output_copy = output - np.mean(output, axis=axis, keepdims=True)
    output_cov = np.matmul(output_copy, output_copy.T) / (output.shape[axis] - 1)
    output_std = np.sqrt(np.diag(output_cov)).reshape(-1, 1)
    output_corr = output_cov / np.matmul(output_std, output_std.T)

    loss = (output_corr - target_corr) ** 2
    return loss.mean() if normalize else loss.sum()


def poisson_loss(output, target, normalize=True):
    """Computes the Poisson loss.

    Args:
        output (np.ndarray): The predicted rate (lambda) from the model.
        target (np.ndarray): The ground-truth event counts.
        normalize (bool): If True, returns the mean; otherwise, the sum.

    Returns:
        float: The Poisson loss.
    """
    loss = output - target * np.log(output + 1e-9)
    return loss.mean() if normalize else loss.sum()


# ##############################################################################
# PyTorch Implementations (GPU)
# ##############################################################################


def pearson_correlation_gpu(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """Computes Pearson correlation between two PyTorch tensors.

    Args:
        y1 (torch.Tensor): The first tensor.
        y2 (torch.Tensor): The second tensor, must have the same shape as y1.
        axis (int or tuple[int]): The axis or axes along which to compute.
        eps (float): A small value to add to the standard deviation for
            numerical stability.
        **kwargs: Additional keyword arguments passed to the final `torch.mean`.

    Returns:
        torch.Tensor: The Pearson correlation coefficient.
    """
    with torch.no_grad():
        y1 = (y1 - y1.mean(dim=axis, keepdim=True)) / (
            y1.std(dim=axis, keepdim=True, unbiased=False) + eps
        )
        y2 = (y2 - y2.mean(dim=axis, keepdim=True)) / (
            y2.std(dim=axis, keepdim=True, unbiased=False) + eps
        )
        return (y1 * y2).mean(dim=axis, **kwargs)


def oracle_explainable_variance_gpu(response, eps=1e-9):
    """Computes the oracle explainable variance per neuron on GPU.

    Args:
        response (torch.Tensor): A tensor of neural responses with shape
            (trials, repeats, neurons, frames).
        eps (float): A small value for numerical stability.

    Returns:
        torch.Tensor: The explainable variance for each neuron.
    """
    with torch.no_grad():
        total_var = torch.var(response, dim=(0, 1, 3), unbiased=True)
        noise_var = torch.mean(torch.var(response, dim=1, unbiased=True), dim=(0, 2))
        return (total_var - noise_var) / (total_var + eps)


def fraction_variance_explained_gpu(response, prediction):
    """Computes the fraction of variance explained (FVE) on GPU.

    Args:
        response (torch.Tensor): The ground-truth neural responses, with shape
            (trials, repeats, neurons, frames).
        prediction (torch.Tensor): The model's predicted responses, with the
            same shape as `response`.

    Returns:
        torch.Tensor: The fraction of variance explained for each neuron.
    """
    with torch.no_grad():
        mse = torch.mean((response - prediction) ** 2, dim=(0, 1, 3))
        total_var = torch.var(response, dim=(0, 1, 3), unbiased=True)
        return 1 - mse / total_var


def fraction_explainable_variance_gpu(response):
    """Computes the fraction of explainable variance (FEV) on GPU.

    Args:
        response (torch.Tensor): The ground-truth neural responses, with shape
            (trials, repeats, neurons, frames).

    Returns:
        torch.Tensor: The fraction of explainable variance for each neuron.
    """
    with torch.no_grad():
        total_var = torch.var(response, dim=(0, 1, 3), unbiased=True)
        noise_var = torch.mean(torch.var(response, dim=1, unbiased=True), dim=(0, 2))
        return (total_var - noise_var) / total_var


def fraction_explainable_variance_explained_gpu(response, prediction):
    """Computes the fraction of explainable variance explained (FEVe) on GPU.

    Args:
        response (torch.Tensor): The ground-truth neural responses, with shape
            (trials, repeats, neurons, frames).
        prediction (torch.Tensor): The model's predicted responses, with the
            same shape as `response`.

    Returns:
        torch.Tensor: The FEVe for each neuron.
    """
    with torch.no_grad():
        mse = torch.mean((response - prediction) ** 2, dim=(0, 1, 3))
        total_var = torch.var(response, dim=(0, 1, 3), unbiased=True)
        noise_var = torch.mean(torch.var(response, dim=1, unbiased=True), dim=(0, 2))
        return 1 - (mse - noise_var) / (total_var - noise_var)


def oracle_correlation_gpu(repeated_responses, eps=1e-8, axis=(0, 2)):
    """Computes the oracle correlation (CCmax) on GPU.

    Args:
        repeated_responses (torch.Tensor): A tensor of neural responses with
            shape (trials, repeats, neurons, frames).
        eps (float): A small value for numerical stability.
        axis (tuple[int]): The axes along which to compute the variance.

    Returns:
        torch.Tensor: The oracle correlation for each neuron.
    """
    with torch.no_grad():
        num_repeats = repeated_responses.shape[1]
        var_axis = (0, 3) if len(axis) == 2 else 3
        noise_var_axis = 0 if len(axis) == 2 else 1

        signal_var = torch.var(
            repeated_responses.mean(dim=1), dim=axis, unbiased=True
        )
        total_var_per_repeat = torch.var(
            repeated_responses, dim=var_axis, unbiased=True
        )
        noise_var = torch.mean(total_var_per_repeat, dim=noise_var_axis)

        return torch.sqrt(
            (num_repeats * signal_var - noise_var)
            / ((num_repeats - 1) * signal_var + eps)
        )


def normalized_correlation_gpu(response, prediction, eps=1e-8, axis=(0, 2)):
    """Computes the normalized correlation on GPU.

    Args:
        response (torch.Tensor): The ground-truth neural responses, with shape
            (trials, repeats, neurons, frames).
        prediction (torch.Tensor): The model's predicted responses, with the
            same shape as `response`.
        eps (float): A small value for numerical stability.
        axis (tuple[int]): The axes along which to compute the correlation.

    Returns:
        torch.Tensor: The normalized correlation for each neuron.
    """
    with torch.no_grad():
        response_mean = response.mean(dim=1)
        prediction_mean = prediction.mean(dim=1)
        cc_abs = pearson_correlation_gpu(
            response_mean, prediction_mean, axis=axis, eps=eps
        )
        cc_max = oracle_correlation_gpu(response, eps=eps, axis=axis)
        return cc_abs / cc_max


def mean_squared_error_gpu(response, prediction, normalize=True):
    """Computes the mean squared error (MSE) on GPU.

    Args:
        response (torch.Tensor): The ground-truth tensor.
        prediction (torch.Tensor): The predicted tensor.
        normalize (bool): If True, returns the mean; otherwise, the sum.

    Returns:
        torch.Tensor: The mean or sum of the squared error.
    """
    with torch.no_grad():
        loss = (response - prediction) ** 2
        return loss.mean() if normalize else loss.sum()


def cross_correlation_mse_gpu(output, target, axis=0, normalize=True):
    """Computes MSE between cross-correlation matrices on GPU.

    Args:
        output (torch.Tensor): The predicted signal tensor, with shape
            (neurons, time_bins).
        target (torch.Tensor): The ground-truth signal tensor, with the same
            shape as `output`.
        axis (int): The axis representing the observation dimension (e.g., time).
        normalize (bool): If True, returns the mean; otherwise, the sum.

    Returns:
        torch.Tensor: The MSE between the two correlation matrices.
    """
    with torch.no_grad():
        target = target - torch.mean(target, dim=axis, keepdim=True)
        target_cov = torch.matmul(target, target.T) / (target.shape[axis] - 1)
        target_std = torch.sqrt(torch.diag(target_cov)).reshape(-1, 1)
        target_corr = target_cov / torch.matmul(target_std, target_std.T)

        output = output - torch.mean(output, dim=axis, keepdim=True)
        output_cov = torch.matmul(output, output.T) / (output.shape[axis] - 1)
        output_std = torch.sqrt(torch.diag(output_cov)).reshape(-1, 1)
        output_corr = output_cov / torch.matmul(output_std, output_std.T)

        loss = (output_corr - target_corr) ** 2
        return loss.mean() if normalize else loss.sum()


def poisson_loss_gpu(output, target, normalize=True):
    """Computes the Poisson loss on GPU.

    Args:
        output (torch.Tensor): The predicted rate (lambda) from the model.
        target (torch.Tensor): The ground-truth event counts.
        normalize (bool): If True, returns the mean; otherwise, the sum.

    Returns:
        torch.Tensor: The Poisson loss.
    """
    with torch.no_grad():
        loss = output - target * torch.log(output + 1e-9)
        return loss.mean() if normalize else loss.sum()



# NumPy (CPU) aliases
corr = pearson_correlation
CCmax = oracle_correlation
CCnorm = normalized_correlation
mse = mean_squared_error
fev = fraction_explainable_variance
fev_e = fraction_explainable_variance_explained
fv_e = fraction_variance_explained
explainable_var = oracle_explainable_variance

# PyTorch (GPU) aliases
corr_gpu = pearson_correlation_gpu
CCmax_gpu = oracle_correlation_gpu
CCnorm_gpu = normalized_correlation_gpu
mse_gpu = mean_squared_error_gpu
fev_gpu = fraction_explainable_variance_gpu
fev_e_gpu = fraction_explainable_variance_explained_gpu
fv_e_gpu = fraction_variance_explained_gpu
explainable_var_gpu = oracle_explainable_variance_gpu
poissonloss_gpu = poisson_loss_gpu
