import argparse
import os
import pickle

import numpy as np
import torch
from fnn import microns
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from tqdm import tqdm

from src.data.loaders import TowardsLoaderBase
from scripts.utils import create_paths_dict


def get_model_output(model, stimuli, ids, device='cuda'):
    """
    Computes model predictions for a batch of video stimuli.

    Args:
        model: The neural network model to evaluate.
        stimuli (torch.Tensor): A batch of video stimuli with shape
            (B, F, H, W), where B is batch size, F is frames, H is height,
            and W is width.
        ids (torch.Tensor): A tensor of neuron IDs for which to get outputs.
        device (str): The device to run the computation on (e.g., 'cuda').

    Returns:
        torch.Tensor: The model's responses with shape (B, F, U), where U is
            the number of neurons.
    """
    model.reset()
    batch_size, n_frames, _, _ = stimuli.shape
    n_units = ids.shape[0]
    output = torch.empty(
        batch_size, n_frames, n_units, device=device, dtype=torch.float
    )

    # Model expects specific empty tensors for perspective and modulation.
    perspective = torch.zeros((batch_size, 2), device=device, dtype=torch.float)
    modulation = torch.zeros((batch_size, 2), device=device, dtype=torch.float)

    # Process frame by frame, as required by the model's recurrent state.
    stimuli = stimuli.permute(1, 0, 2, 3).contiguous() / 255.0
    for t in range(1, n_frames):
        frame_batch = stimuli[t].unsqueeze(1)
        model_output = model(
            stimulus=frame_batch,
            perspective=perspective,
            modulation=modulation
        )
        output[:, t].copy_(model_output)
    return output


def compute_arfs(dataloaders, tier, skip=0, selected_sessions=None):
    """
    Computes artificial receptive fields (aRFs) using reverse correlation.

    Iterates through the dataloader, gets model predictions for white noise
    stimuli, and calculates the aRF for each neuron by averaging the stimuli
    weighted by the neuron's response.

    Args:
        dataloaders (dict): A dictionary of data loaders.
        tier (str): The key for the desired dataloader (e.g., 'test').
        skip (int): The number of initial frames to skip in each video to
            avoid transient model responses.
        selected_sessions (list[str]): A list of session keys to process.

    Returns:
        dict: A dictionary where keys are session identifiers and values are
            NumPy arrays of computed aRFs with shape (U, H, W).
    """
    output = {}
    for data_key in selected_sessions:
        session, scan = int(data_key[0]), int(data_key[-1])
        session_model, ids = microns.scan(session, scan)
        session_model.eval()
        device = next(session_model.parameters()).device

        # Initialize a tensor to accumulate the response-weighted stimuli.
        summed_response_frames = None

        with torch.inference_mode(), torch.cuda.amp.autocast():
            for loader in dataloaders[tier].values():
                for data in tqdm(loader, desc=f"Session {data_key}"):
                    batch_kwargs = data._asdict()
                    stimuli = batch_kwargs['videos'].to(device, torch.uint8)
                    stimuli = stimuli.float()

                    pred = get_model_output(
                        session_model, stimuli=stimuli, ids=ids, device=device
                    )

                    # Time-average the response, skipping initial frames.
                    pred_mean = pred[:, skip:, :].mean(dim=1)  # Shape: [B, U]
                    last_frame = stimuli[:, -1, ...]           # Shape: [B, H, W]

                    # Weight the last frame by the mean response (reverse correlation).
                    # Broadcasting: [B, U] -> [B, U, 1, 1]
                    pred_img = pred_mean[:, :, None, None] * last_frame[:, None, :, :]

                    if summed_response_frames is None:
                        summed_response_frames = torch.zeros_like(
                            pred_img.squeeze(0)
                        )

                    # Accumulate over all batches.
                    summed_response_frames += pred_img.sum(dim=0)

        output[data_key] = summed_response_frames.detach().cpu().numpy()
    return output


def gaussian2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta):
    """
    A 2D rotated Gaussian function for curve fitting.

    Args:
        coords (tuple[np.ndarray, np.ndarray]): Tuple of X and Y grid coordinates.
        amplitude (float): The amplitude of the Gaussian.
        x0 (float): The x-coordinate of the center.
        y0 (float): The y-coordinate of the center.
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.
        theta (float): Rotation angle in radians.

    Returns:
        np.ndarray: The flattened 2D Gaussian evaluated on the grid.
    """
    x, y = coords
    x0, y0 = float(x0), float(y0)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + \
        (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + \
        np.sin(2 * theta) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + \
        (np.cos(theta)**2) / (2 * sigma_y**2)
    g = amplitude * np.exp(
        -(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2)
    )
    return g.ravel()


def fit_gaussian_to_arfs(arfs_dict, drop_percent=5):
    """
    Fits a 2D Gaussian to each aRF map.

    The aRF is first centered by subtracting its mean, and the absolute value
    is taken before fitting. Fits with the lowest amplitudes can be dropped.

    Args:
        arfs_dict (dict): A dictionary of aRFs, keyed by session. Each value
            is an array of shape (U, H, W).
        drop_percent (int): The percentage of fits with the lowest amplitudes
            to discard as unreliable.

    Returns:
        dict: A dictionary where keys are session identifiers and values are
            lists of fitted parameter arrays. Failed fits are represented by None.
    """
    fitted_params_dict = {}
    for session, arfs in arfs_dict.items():
        n_units, height, width = arfs.shape
        y_grid, x_grid = np.indices((height, width))
        coords = (x_grid.ravel(), y_grid.ravel())

        fitted_params_list = []
        for i in range(n_units):
            arf_map = arfs[i]
            # Center the aRF and take the absolute value for fitting.
            arf_centered = np.abs(arf_map - np.mean(arf_map))
            data_flat = arf_centered.ravel()

            initial_guess = (np.max(arf_centered), width / 2, height / 2, 10, 10, 0)
            try:
                popt, _ = curve_fit(
                    gaussian2d,
                    coords,
                    data_flat,
                    p0=initial_guess,
                    maxfev=10000
                )
                fitted_params_list.append(popt)
            except RuntimeError:
                fitted_params_list.append(None)

        # Filter out fits with low amplitudes.
        valid_fits = [p for p in fitted_params_list if p is not None]
        if valid_fits and drop_percent > 0:
            amplitudes = [p[0] for p in valid_fits]
            threshold = np.percentile(amplitudes, drop_percent)
            filtered_params = [
                p if p is not None and p[0] >= threshold else None
                for p in fitted_params_list
            ]
        else:
            filtered_params = fitted_params_list

        fitted_params_dict[session] = filtered_params
    return fitted_params_dict


def fit_gaussian_lowpass_to_arfs(arfs_dict, drop_percent=5, sigma=2):
    """
    Applies a low-pass filter and then fits a 2D Gaussian to each aRF map.

    Args:
        arfs_dict (dict): Dictionary of aRF maps, keyed by session.
        drop_percent (int): The percentage of fits with the lowest
            amplitudes to discard.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - A dictionary of the low-pass filtered aRFs.
            - A dictionary of the fitted Gaussian parameters.
    """
    low_pass_arfs = {}
    for session, arfs in arfs_dict.items():
        # Apply Gaussian filter to all aRFs in the session at once.
        smoothed_arfs = gaussian_filter(arfs, sigma=(0, sigma, sigma))
        low_pass_arfs[session] = smoothed_arfs

    # Fit the smoothed aRFs using the standard fitting function.
    fitted_params_dict = fit_gaussian_to_arfs(low_pass_arfs, drop_percent)

    return low_pass_arfs, fitted_params_dict


def main(args):
    """
    Main pipeline to compute aRFs, fit Gaussians, and save results.
    """
    model_paths = create_paths_dict("white_noise", args.session_ids)

    loader = TowardsLoaderBase(args.batch_size, args, 'images')
    loader = loader.load(model_paths['white_noise'])

    print("Computing aRFs via reverse correlation...")
    final_arfs_dict = compute_arfs(
        loader,
        'test',
        skip=args.skip,
        selected_sessions=args.selected_sessions
    )
    print("aRF computation complete.")

    print("Fitting 2D Gaussians to raw aRFs...")
    fitted_params_dict = fit_gaussian_to_arfs(
        final_arfs_dict, drop_percent=args.drop_percent
    )
    print("Fitting to raw aRFs complete.")

    print("Fitting 2D Gaussians to low-pass filtered aRFs...")
    low_pass_arfs, low_pass_params = fit_gaussian_lowpass_to_arfs(
        final_arfs_dict, drop_percent=args.drop_percent, sigma=args.sigma
    )
    print("Fitting to low-pass aRFs complete.")

    # Prepare results for saving.
    save_path = os.path.join('save', 'results', args.model_name, 'aRF.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        with open(save_path, 'rb') as f:
            output_data = pickle.load(f)
        print(f"Loaded existing results from {save_path}. Updating...")
    except FileNotFoundError:
        output_data = {}
        print("No existing results file found. Creating a new one.")

    # Update the dictionary with new or overwritten results.
    output_data.setdefault('arf', {}).update(final_arfs_dict)
    output_data.setdefault('params', {}).update(fitted_params_dict)
    output_data.setdefault('low_pass_arf', {}).update(low_pass_arfs)
    output_data.setdefault('low_pass_params', {}).update(low_pass_params)

    with open(save_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and analyze artificial receptive fields."
    )
    parser.add_argument(
        "--session_ids", type=list, default=None, help="List of session IDs."
    )
    parser.add_argument(
        "--selected_sessions", type=str, nargs='+', required=True,
        help="Specific session keys to process (e.g., '3-1' '4-2')."
    )
    parser.add_argument(
        "--model_name", type=str, default='towards_wang',
        help="Name of the model, used for saving results."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing."
    )
    parser.add_argument(
        "--skip", type=int, default=30,
        help="Number of initial frames to skip per video."
    )
    parser.add_argument(
        "--drop_percent", type=int, default=0,
        help="Percentage of low-amplitude Gaussian fits to drop."
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0,
        help="Standard deviation for the low-pass Gaussian filter."
    )
    # Add other necessary arguments from the original script that are used by loaders.
    parser.add_argument(
        "--frames", type=int, default=60, help="Frames per video."
    )
    parser.add_argument(
        "--n_inputs", type=int, default=500000, help="Number of input samples."
    )
    parser.add_argument(
        "--gray_screen_frames", type=int, default=30,
        help="Number of gray screen frames."
    )

    main_args = parser.parse_args()
    main(main_args)
