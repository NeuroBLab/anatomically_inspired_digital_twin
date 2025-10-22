import argparse
import os
import pickle

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from tqdm import tqdm

from src.engine.build import build_model
from src.data.utils import (
    compute_stats,
    extract_neurons_subset,
)
from src.data.loaders import get_loader
from src.engine.scheduler import Scheduler
from scripts.utils import create_paths_dict, initialize_model_args


def get_model_output(dataloaders, tier, model, n_inputs=500000, skip=0,
                     selected_sessions=None):
    """
    Computes artificial Receptive Fields (aRFs) using reverse correlation.

    This function passes white noise stimuli through the model and calculates
    the stimulus-weighted average of neural responses for each neuron.

    Args:
        dataloaders (dict): Dictionary of data loaders for different tiers.
        tier (str): The data tier to use (e.g., 'test').
        model (torch.nn.Module): The pre-trained model.
        n_inputs (int, optional): Total number of input images to process.
        skip (int, optional): Number of initial time steps in the model's
            response to ignore for each input.
        selected_sessions (list, optional): A list of session IDs to process.
            If None, all sessions are processed.

    Returns:
        dict: A dictionary where keys are session IDs and values are numpy
            arrays of shape (num_neurons, height, width) representing the
            computed receptive fields.
    """
    model.eval()
    output = {}
    with torch.no_grad():
        for session, dataloader in dataloaders[tier].items():
            print(f"Processing session: {session}")
            i = 0
            for data in tqdm(dataloader, desc=f"Session {session}"):
                batch_kwargs = data._asdict() if not isinstance(data, dict) else data
                model_output = model.forward_all(
                    batch_kwargs['videos'],
                    selected_sessions=selected_sessions,
                    **batch_kwargs
                )
                for session_id, session_responses in model_output.items():
                    if session_id not in output:
                        output[session_id] = np.zeros(
                            (session_responses.shape[2], 36, 64)
                        )

                    # Select the last frame from the first channel as the stimulus.
                    videos = batch_kwargs['videos'][:, 0, -1, :, :]

                    # Compute response-weighted stimuli (reverse correlation).
                    # Average neural response over time, then multiply by stimulus.
                    response_mean = session_responses[:, skip:, :].mean(axis=1)
                    output_tensor = (
                        response_mean[:, :, None, None] * videos[:, None, :, :]
                    )

                    output[session_id] += output_tensor.sum(axis=0).cpu().numpy()

                i += videos.shape[0]
                if i >= n_inputs:
                    break
    return output


def gaussian2d(xy_coords, amplitude, x0, y0, sigma_x, sigma_y, theta):
    """
    Computes a 2D rotated Gaussian function.

    Args:
        xy_coords (tuple): A tuple of (x, y) coordinate grids.
        amplitude (float): The amplitude of the Gaussian.
        x0 (float): The x-coordinate of the center.
        y0 (float): The y-coordinate of the center.
        sigma_x (float): The standard deviation along the x-axis.
        sigma_y (float): The standard deviation along the y-axis.
        theta (float): The rotation angle in radians.

    Returns:
        np.ndarray: The flattened 2D Gaussian values.
    """
    x, y = xy_coords
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
    Fits a 2D Gaussian to each receptive field map.

    For each neuron's aRF, the map is centered, its absolute value is taken,
    and a 2D Gaussian is fitted. Fits with the lowest amplitudes are dropped.

    Args:
        arfs_dict (dict): Dictionary of receptive field maps, keyed by session.
        drop_percent (int, optional): The percentage of fits with the lowest
            amplitudes to discard.

    Returns:
        dict: A dictionary where keys are session IDs and values are lists of
            fitted Gaussian parameters. Failed fits are represented by None.
    """
    fitted_params_dict = {}
    for session, arfs in arfs_dict.items():
        num_neurons, height, width = arfs.shape
        y_grid, x_grid = np.indices((height, width))
        x_flat, y_flat = x_grid.ravel(), y_grid.ravel()

        fitted_params_list = []
        for i in range(num_neurons):
            arf_map = arfs[i]
            # Center the map on its baseline and take the absolute value.
            arf_centered = np.abs(arf_map - np.mean(arf_map))
            data_flat = arf_centered.ravel()

            initial_guess = (np.max(arf_centered), width / 2, height / 2, 10, 10, 0)
            try:
                popt, _ = curve_fit(
                    gaussian2d,
                    (x_flat, y_flat),
                    data_flat,
                    p0=initial_guess,
                    maxfev=10000
                )
                fitted_params_list.append(popt)
            except RuntimeError as e:
                print(f"Fit failed for neuron {i} in session {session}: {e}")
                fitted_params_list.append(None)

        # Filter out fits with low amplitudes.
        valid_fits = [p for p in fitted_params_list if p is not None]
        if valid_fits:
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
    Main execution pipeline to compute and save receptive fields.
    """
    args = initialize_model_args(args)
    model_paths = create_paths_dict(args.model_data, args.session_ids)

    neurons_idxs = None
    if args.subset_neurons:
        stats_dataloader = get_loader(
            args, args.dataset, args.batch_size, model_paths,
            input_type=args.input_type, statistics=True
        )
        neurons_idxs = extract_neurons_subset(stats_dataloader, args.subset_metric)

    input_stats = None
    if args.normalize:
        print("Computing normalization statistics...")
        stats_dataloader = get_loader(
            args, args.dataset, args.batch_size, model_paths,
            input_type=args.input_type, statistics=True,
            neurons_subset=neurons_idxs
        )
        input_stats = compute_stats(stats_dataloader, args.normalize)

    model_dataloader = get_loader(
        args, args.dataset, args.batch_size, model_paths,
        input_type=args.input_type, input_stats=input_stats,
        neurons_subset=neurons_idxs
    )

    model = build_model(
        args, model_dataloader, layer=args.layer, brain_area=args.brain_area
    )
    model.to(args.device)
    print(f"Model loaded on {args.device}.")
    print(f"Total trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    scheduler = Scheduler(args, model=model, save_optimizer=False)
    scheduler.restore(force=True)
    print("Model weights restored from checkpoint.")

    input_paths = create_paths_dict("white_noise", None)
    input_dataloader = get_loader(
        args, dataset="white_noise", batch_size=args.batch_size,
        paths_dict=input_paths, input_type=args.input_type,
        input_stats=input_stats, neurons_subset=neurons_idxs
    )

    print("Computing receptive fields...")
    arfs_dict = get_model_output(
        input_dataloader, 'test', model, n_inputs=args.n_inputs,
        selected_sessions=args.selected_sessions
    )
    print("Receptive fields computed.")

    print("Fitting 2D Gaussians to receptive fields...")
    fitted_params_dict = fit_gaussian_to_arfs(arfs_dict, drop_percent=args.drop_percent)
    print("Fitting completed.")

    save_path = os.path.join('save', 'results', args.model_name, f'{args.filename}.pkl')
    try:
        with open(save_path, 'rb') as f:
            output_data = pickle.load(f)
        print(f"Loaded existing results from {save_path}. Will update file.")
    except FileNotFoundError:
        output_data = {'arf': {}, 'params': {}, 'low_pass_arf': {}, 'low_pass_params': {}}
        print(f"No existing results file found. Creating new file at {save_path}.")

    output_data['arf'].update(arfs_dict)
    output_data['params'].update(fitted_params_dict)

    low_pass_fitted_arfs, low_pass_fitted_params = fit_gaussian_lowpass_to_arfs(
        arfs_dict, drop_percent=args.drop_percent, sigma=args.sigma
    )

    output_data['low_pass_arf'].update(low_pass_fitted_arfs)
    output_data['low_pass_params'].update(low_pass_fitted_params)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute neural receptive fields using a pre-trained model."
    )
    parser.add_argument("--filename", type=str, default='aRF',
                        help="Output filename for the results pickle file.")
    parser.add_argument("--model_data", type=str, required=True,
                        help="Identifier for the model's training dataset.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model, used for saving results.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing stimuli.")
    parser.add_argument("--n_inputs", type=int, default=500000,
                        help="Number of white noise images to process.")
    parser.add_argument("--output_dir", type=str, default='save/models',
                        help="Directory for model checkpoints.")
    parser.add_argument("--frames", type=int, default=60,
                        help="Number of frames per video input.")
    parser.add_argument("--skip", type=int, default=30,
                        help="Initial frames to skip in model response.")
    parser.add_argument("--gray_screen_frames", type=int, default=30,
                        help="Number of gray screen frames.")
    parser.add_argument("--save_results", action='store_true',
                        help="Flag to save results (currently always saves).")
    parser.add_argument("--selected_sessions", type=str, nargs='+', default=None,
                        help="List of specific session IDs to analyze.")
    parser.add_argument("--drop_percent", type=float, default=0,
                        help="Percent of fits to drop based on amplitude")
    parser.add_argument("--sigma", type=float, default=1,
                        help="Sigma for Gaussian low-pass filter")

    main_args = parser.parse_args()
    main(main_args)
