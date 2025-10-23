"""
Simulates neural responses to rotated static images using a pre-trained model
of the mouse visual cortex.

This script loads a specified model, computes normalization statistics from the
model's training data, and then processes a new set of image stimuli to
generate simulated neural activity. The response for each stimulus is averaged
over the time dimension to produce a single vector. The output is saved as a
pickle file for later analysis.
"""

import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from src.engine.build import build_model
from src.data.utils import compute_stats, extract_neurons_subset
from src.data.loaders import get_loader
from src.engine.scheduler import Scheduler
from scripts.utils import create_paths_dict, initialize_model_args


def get_model_output(dataloaders, tier, model, skip=0, minibatch_frames=300):
    """
    Processes image data through the model to get neural responses.

    This function iterates through batches of stimuli, processing them in
    temporal chunks to manage memory. For each stimulus, it calculates the
    mean response over the time axis after an initial skip period.

    Args:
        dataloaders (dict): A dictionary of data loaders, keyed by tier
            (e.g., 'test').
        tier (str): The key for the tier to process (e.g., 'test').
        model (torch.nn.Module): The pre-trained model to use for simulation.
        skip (int): Number of initial frames to discard before averaging.
        minibatch_frames (int): The number of frames per chunk for processing.

    Returns:
        dict: A dictionary where keys are session names and values are
            dictionaries mapping model output keys to NumPy arrays of simulated
            neural responses. The shape of each array is
            (num_trials, num_neurons).
    """
    model.eval()
    device = next(model.parameters()).device
    output = {}

    with torch.no_grad():
        for input_session, data_list in dataloaders[tier].items():
            try:
                session_responses = {}
                for data in tqdm(data_list, desc=f"Processing {input_session}"):
                    batch_kwargs = data._asdict() if hasattr(data, '_asdict') else data
                    input_data = batch_kwargs['videos']
                    total_frames = input_data.shape[2]
                    batch_output = {}

                    # Process video frames in chunks to manage GPU memory.
                    for start in range(0, total_frames, minibatch_frames):
                        end = min(start + minibatch_frames, total_frames)
                        input_chunk = torch.from_numpy(input_data[:, :, start:end, ...])
                        output_chunk = model.forward_all(input_chunk.to(device), **batch_kwargs)

                        for model_session, value in output_chunk.items():
                            batch_output.setdefault(model_session, []).append(value.cpu())

                    # Concatenate chunks from the same batch.
                    for model_session, chunks in batch_output.items():
                        full_response = torch.cat(chunks, dim=1)
                        session_responses.setdefault(model_session, []).append(
                            full_response.numpy()[:, skip:, ...].mean(axis=1)
                        )

                # Aggregate results from all batches for the session.
                final_session_dict = {
                    k: np.concatenate(v, axis=0)
                    for k, v in session_responses.items()
                }
                output[input_session] = final_session_dict

            except Exception as e:
                # This allows the script to continue if a session fails, which
                # may be expected for certain datasets.
                print(f"Skipping session {input_session} due to error: {e}")
                continue

    return output


def main(args):
    """
    Main execution function to set up and run the simulation.
    """
    args = initialize_model_args(args)

    # --- Dataloader and statistics setup for the model ---
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
            input_type=args.input_type, neurons_subset=neurons_idxs,
            statistics=True
        )
        input_stats = compute_stats(stats_dataloader, args.normalize)

    model_dataloader = get_loader(
        args, args.dataset, args.batch_size, model_paths,
        input_type=args.input_type, input_stats=input_stats,
        neurons_subset=neurons_idxs
    )

    # --- Model construction ---
    model = build_model(
        args, model_dataloader, layer=args.layer, brain_area=args.brain_area
    )
    model.to(args.device)
    print(f"Model built with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # --- Dataloader for input stimuli ---
    args.session_ids = None
    input_data_path = os.path.join(args.input_data, args.input_data_type)
    input_paths = create_paths_dict(input_data_path, args.session_ids)
    input_paths = {input_data_path.split('/')[0]: input_paths[input_data_path]}

    input_dataloader = get_loader(
        args, args.input_data, args.batch_size, input_paths,
        input_type=args.input_data_type, input_stats=input_stats,
        neurons_subset=neurons_idxs
    )

    # --- Restore model from checkpoint ---
    session = list(model.readout.keys())[0]
    kernels_before_restore = model.readout[session]._features.detach().clone()

    scheduler = Scheduler(args, model=model, save_optimizer=False)
    scheduler.restore(force=True)

    # Sanity check to ensure model weights were loaded correctly.
    kernels_after_restore = model.readout[session]._features.detach().clone()
    if torch.all(torch.eq(kernels_before_restore, kernels_after_restore)):
        raise RuntimeError("Model weights did not change after restoration.")
    print("Model restored successfully from checkpoint.")

    # --- Generate and save model output ---
    model_output = get_model_output(
        input_dataloader,
        'test',
        model,
        skip=args.skip,
        minibatch_frames=args.minibatch_frames
    )

    save_dir = os.path.join('save', 'data', 'hoeller')
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{args.model_name}_{args.input_data}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(model_output, f)
    print(f"Model output saved to {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate simulated neural responses for the Hoeller et al. rotated objects dataset.'
    )
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model.')
    parser.add_argument('--model_data', type=str, required=True, help='Path to the model training data.')
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input stimulus data.')
    parser.add_argument('--input_data_type', type=str, default='all', help='Type of input data to use.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing.')
    parser.add_argument("--normalize", nargs="+", type=str, default=['neurons', 'channels', 'coordinates'], help='Normalization method.')
    parser.add_argument('--skip', type=int, default=50, help='Number of initial frames to skip before averaging.')
    parser.add_argument('--output_dir', type=str, default='save/models', help='Directory for model checkpoints.')
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help='Computation device.')
    parser.add_argument("--minibatch_frames", type=int, default=300, help='Frames per chunk for GPU processing.')

    cli_args = parser.parse_args()
    main(cli_args)
