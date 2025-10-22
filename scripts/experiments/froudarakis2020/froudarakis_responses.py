"""
Simulates neural responses to visual stimuli using a pre-trained model of the
mouse visual cortex.

This script loads a specified model, computes normalization statistics from the
model's training data, and then processes a new set of video stimuli to
generate simulated neural activity. The output is saved as a pickle file for
later analysis.
"""

import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from src.layers.encoder import ResNet50Encoder
from src.engine.build import build_model
from src.data.utils import compute_stats, extract_neurons_subset
from src.data.loaders import get_loader
from src.engine.scheduler import Scheduler
from scripts.utils import create_paths_dict, initialize_model_args


def get_model_output(dataloaders,
                     tier,
                     model,
                     skip=0,
                     minibatch_frames=300,
                     data_key=None,
                     use_amp=True):
    """
    Processes video data through the model to get neural responses.

    This function iterates through batches of video data, processes them in
    temporal chunks to prevent out-of-memory errors, and aggregates the model's
    output.

    Args:
        dataloaders (dict): A dictionary of data loaders, keyed by tier
            (e.g., 'test').
        tier (str): The key for the tier to process (e.g., 'test').
        model (torch.nn.Module): The pre-trained model to use for simulation.
        skip (int): Number of initial frames to discard from the output.
        minibatch_frames (int): The initial number of frames per chunk for
            processing. This is adapted dynamically on OOM errors.
        data_key (str, optional): If provided, specifies a particular data/session key
            for the model's forward pass.
        use_amp (bool): Whether to use automatic mixed precision.

    Returns:
        dict: A dictionary where keys are objects names and values are
            dictionaries mapping training dataset's sessions keys to NumPy arrays of simulated
            neural responses. The shape of each array is
            (num_trials, num_timesteps, num_neurons), where timesteps are
            the result of averaging over 15-frame windows for stimuli at 30fps.
    """
    model.eval()
    device = next(model.parameters()).device
    output = {}

    with torch.cuda.amp.autocast(enabled=use_amp), torch.no_grad():
        for input_session, data_list in dataloaders[tier].items():
            session_outputs = {}

            for data in tqdm(data_list, desc=f"Processing {input_session}"):
                batch_kwargs = data._asdict() if hasattr(data, '_asdict') else data
                input_data = torch.from_numpy(batch_kwargs['videos'])
                total_frames = input_data.shape[2]
                partial_outputs = {}

                # Process video frames in chunks to manage GPU memory.
                pos = 0
                current_chunk_size = minibatch_frames
                while pos < total_frames:
                    end = min(pos + current_chunk_size, total_frames)
                    input_chunk = input_data[:, :, pos:end, ...]
                    model_kwargs = {
                        k: v for k, v in batch_kwargs.items() if k != 'videos'
                    }

                    # Retry loop to handle CUDA Out-of-Memory errors by
                    # reducing the chunk size.
                    while True:
                        try:
                            chunk_gpu = input_chunk.to(device)
                            if data_key is not None:
                                output_chunk = {
                                    key: model(chunk_gpu, data_key=key, **model_kwargs)
                                    for key in data_key
                                }
                            else:
                                output_chunk = model.forward_all(
                                    chunk_gpu,
                                    selected_sessions=data_key,
                                    **model_kwargs
                                )

                            for m_sess, val in output_chunk.items():
                                partial_outputs.setdefault(m_sess, []).append(val.cpu())
                            break  # Success, exit retry loop.

                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            new_size = current_chunk_size // 2
                            if new_size < 1:
                                raise RuntimeError(
                                    "CUDA out of memory even with chunk size of 1."
                                )
                            current_chunk_size = new_size
                            print(f"[OOM] Reducing chunk size to {current_chunk_size} and retrying...")
                    pos = end

                # Concatenate chunks and downsample temporal resolution.
                for m_sess, val_list in partial_outputs.items():
                    concatenated = torch.cat(val_list, dim=1)
                    arr = concatenated.numpy()[:, skip:, ...]

                    # Reshape from (B, F, N) to (B, F//15, 15, N) and average
                    # to get final response shape (B, F//15, N).
                    frames_after_skip = arr.shape[1]
                    if frames_after_skip % 15 != 0:
                        print(f"Warning: {frames_after_skip} frames not divisible by 15. Skipping.")
                        continue
                    arr = arr.reshape(arr.shape[0], -1, 15, arr.shape[2])
                    arr = arr.mean(axis=2)
                    session_outputs.setdefault(m_sess, []).append(arr)

            # Aggregate results from all batches for the session.
            final_session_dict = {
                m_sess: np.concatenate(arr_list, axis=0)
                for m_sess, arr_list in session_outputs.items()
            }
            output[input_session] = final_session_dict

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
            args,
            args.dataset,
            args.batch_size,
            model_paths,
            input_type=args.input_type,
            statistics=True
        )
        neurons_idxs = extract_neurons_subset(stats_dataloader, args.subset_metric)

    input_stats = None
    if args.normalize:
        print("Computing normalization statistics...")
        stats_dataloader = get_loader(
            args,
            args.dataset,
            args.batch_size,
            model_paths,
            input_type=args.input_type,
            neurons_subset=neurons_idxs,
            statistics=True
        )
        input_stats = compute_stats(stats_dataloader, args.normalize)

    model_dataloader = get_loader(
        args,
        args.dataset,
        args.batch_size,
        model_paths,
        input_type=args.input_type,
        input_stats=input_stats,
        neurons_subset=neurons_idxs
    )

    # --- Model construction ---
    if args.resnet:
        model = ResNet50Encoder(args, model_dataloader, layer_hook=args.layer_hook)
    else:
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
        args,
        args.input_data,
        args.batch_size,
        input_paths,
        input_type=args.input_data_type,
        input_stats=input_stats,
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
        minibatch_frames=args.minibatch_frames,
        data_key=args.selected_sessions
    )

    save_dir = os.path.join('save', 'data', 'froudarakis')
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{args.model_name}_{args.input_data}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(model_output, f)
    print(f"Model output saved to {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate simulated neural responses for the Froudarakis et al. object dataset.'
    )
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model.')
    parser.add_argument('--model_data', type=str, required=True, help='Path to the model training data.')
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input stimulus data.')
    parser.add_argument('--input_data_type', type=str, default='all', help='Type of input data to use.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing.')
    parser.add_argument("--normalize", nargs="+", type=str, default=['neurons', 'channels', 'coordinates'], help='Normalization method.')
    parser.add_argument('--skip', type=int, default=30, help='Number of initial frames to skip.')
    parser.add_argument('--output_dir', type=str, default='save/models', help='Directory for model checkpoints.')
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help='Computation device.')
    parser.add_argument("--minibatch_frames", type=int, default=300, help='Frames per chunk for GPU processing.')
    parser.add_argument("--selected_sessions", type=str, nargs='+', default=None, help='Specific sessions to process.')
    
    cli_args = parser.parse_args()
    main(cli_args)
