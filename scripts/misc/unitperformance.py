# -*- coding: utf-8 -*-
"""
Evaluates a pre-trained model's performance on predicting neural responses.

This script loads a specified model checkpoint, prepares evaluation datasets,
and computes various performance metrics. The results, along with model and
dataset metadata, are saved to a pickle file.
"""

import argparse
import os

import numpy as np
import torch

from src.layers.encoder import ResNet50Encoder
from src.metrics import (CCmax, CCnorm, corr, explainable_var,
                                        fev, fev_e, fv_e, mse)
from src.engine.build import build_model
from src.data.loaders import get_loader
from src.data.utils import compute_stats, extract_neurons_subset
from src.engine.scheduler import Scheduler
from src.data.summary import Summary
from scripts.utils import JSONArgs, create_paths_dict, initialize_model_args

# Constants for datasets with repeated trials
TRIALS = 6
REPEATS = 10


def get_model_output(model, inputs, responses, data_key=None,
                     minibatch_frames=300, **kwargs):
    """
    Processes input data through the model in minibatches to get predictions.

    Args:
        model (torch.nn.Module): The model to evaluate.
        inputs (torch.Tensor): The input data tensor.
        responses (torch.Tensor): The ground-truth neural responses.
        data_key (str, optional): Identifier for the specific dataset session.
        minibatch_frames (int, optional): Number of frames per minibatch to
            manage memory usage. Defaults to 300.
        **kwargs: Additional keyword arguments passed to the model.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The model's predictions.
            - The ground-truth responses, transposed for consistency.
    """
    total_frames = inputs.shape[2]
    device = next(model.parameters()).device
    model_outputs = []

    for start in range(0, total_frames, minibatch_frames):
        end = min(start + minibatch_frames, total_frames)
        input_chunk = inputs[:, :, start:end, ...].to(device)

        if data_key is None:
            output_chunk = model.forward_all(input_chunk, **kwargs)
        else:
            output_chunk = model(input_chunk, data_key=data_key, **kwargs)
        model_outputs.append(output_chunk.cpu())

    predictions = torch.cat(model_outputs, dim=1)
    # Transpose responses from (Batch, Neurons, Frames) to (Batch, Frames, Neurons)
    ground_truth = responses.transpose(2, 1)

    return predictions, ground_truth


def eval_model_performance(dataloaders, tier, model, frames, eval_repeated,
                           skip=0, inference=True, save_responses=True,
                           use_amp=False, minibatch_frames=300):
    """
    Evaluates model performance on a given data tier (e.g., 'val', 'test').

    Args:
        dataloaders (dict): Dictionary of data loaders.
        tier (str): The data tier to evaluate ('train', 'val', or 'test').
        model (torch.nn.Module): The model to evaluate.
        frames (int): The number of frames per trial.
        eval_repeated (bool): Whether the data has repeated trials.
        skip (int, optional): Number of initial frames to skip. Defaults to 0.
        inference (bool, optional): Whether to save model predictions.
        save_responses (bool, optional): Whether to save ground-truth responses.
        use_amp (bool, optional): Whether to use automatic mixed precision.
        minibatch_frames (int, optional): Size of minibatches for inference.

    Returns:
        dict: A dictionary containing performance metrics for each data key.
    """
    model.eval()
    output = {}
    with torch.cuda.amp.autocast(enabled=use_amp), torch.no_grad():
        for data_key, loader in dataloaders[tier].items():
            print(f"Evaluating on: {data_key}")
            try:
                session_results = {}
                all_responses, all_predictions = [], []

                for data in loader:
                    batch_args = list(data)
                    batch_kwargs = data._asdict() if hasattr(data, '_asdict') else data
                    pred, resp = get_model_output(
                        model, *batch_args, data_key=data_key,
                        minibatch_frames=minibatch_frames, **batch_kwargs
                    )
                    all_responses.append(resp.numpy()[:, skip:, :])
                    all_predictions.append(pred.numpy()[:, skip:, :])

                if eval_repeated:
                    # Reshape for repeated trial analysis.
                    # From (Trials*Repeats, Frames, Neurons) to
                    # (Trials, Repeats, Neurons, Frames)
                    responses = np.concatenate(all_responses).reshape(
                        TRIALS, REPEATS, frames, -1).swapaxes(2, 3)
                    predictions = np.concatenate(all_predictions).reshape(
                        TRIALS, REPEATS, frames, -1).swapaxes(2, 3)

                    session_results['CCmax'] = CCmax(responses)
                    session_results['ev'] = explainable_var(responses)
                    session_results['fev'] = fev(responses)
                    session_results['oracle_score'] = loader.dataset.neurons.pearson
                    session_results['corr_to_ave'] = corr(
                        predictions.mean(axis=1), responses.mean(axis=1), axis=(0, 2)
                    )
                    session_results['fve'] = fv_e(responses, predictions)
                    session_results['feve'] = fev_e(responses, predictions)
                    session_results['CCnorm'] = CCnorm(responses, predictions)
                    axis = (0, 1, 3)
                else:
                    responses = np.concatenate(all_responses, axis=0)
                    predictions = np.concatenate(all_predictions, axis=0)
                    axis = (0, 1)

                session_results['corr'] = corr(predictions, responses, axis=axis)
                session_results['mse'] = np.mean((predictions - responses) ** 2, axis=axis)
                session_results['rmse'] = np.sqrt(session_results['mse'])

                if inference:
                    session_results['prediction'] = predictions
                if save_responses:
                    session_results['responses'] = responses

                output[data_key] = session_results

            except Exception as e:
                print(f"Could not evaluate {data_key} due to an error: {e}")
                continue
    return output


def eval_model_performance_sensorium(dataloaders, tier, model, frames, skip=0,
                                     inference=True, save_responses=True):
    """
    Evaluates model performance specifically for the Sensorium dataset structure.

    Args:
        dataloaders (dict): Dictionary of data loaders.
        tier (str): The data tier to evaluate ('val').
        model (torch.nn.Module): The model to evaluate.
        frames (int): The number of frames per trial.
        skip (int, optional): Number of initial frames to skip. Defaults to 0.
        inference (bool, optional): Whether to save model predictions.
        save_responses (bool, optional): Whether to save ground-truth responses.

    Returns:
        dict: A dictionary containing performance metrics for each data key (session id).
    """
    model.eval()
    output = {}
    with torch.no_grad():
        for data_key, loader in dataloaders[tier].items():
            session_results = {}
            oracle_hashes = loader.dataset.trial_info.oracle_hashes
            oracle_idxs = np.where(loader.dataset.tiers == 'oracle')[0]
            unique_oracle_hashes = oracle_hashes[oracle_idxs]

            # Group responses and predictions by oracle hash
            grouped_outputs = {h: {'response': [], 'prediction': []}
                               for h in np.unique(unique_oracle_hashes)}

            for i, data in enumerate(loader):
                hash_val = unique_oracle_hashes[i]
                batch_args = list(data)
                batch_kwargs = data._asdict() if hasattr(data, '_asdict') else data
                pred, resp = get_model_output(
                    model, *batch_args, data_key=data_key, **batch_kwargs
                )
                grouped_outputs[hash_val]['response'].append(resp.numpy()[:, skip:, :])
                grouped_outputs[hash_val]['prediction'].append(pred.numpy()[:, skip:, :])

            # Calculate metrics on single trials
            all_responses = np.concatenate(
                [np.concatenate(v['response']) for v in grouped_outputs.values()], axis=0
            )
            all_predictions = np.concatenate(
                [np.concatenate(v['prediction']) for v in grouped_outputs.values()], axis=0
            )
            session_results['corr'] = corr(all_predictions, all_responses, axis=(0, 1))
            session_results['mse'] = np.mean((all_predictions - all_responses) ** 2, axis=(0, 1))
            session_results['rmse'] = np.sqrt(session_results['mse'])

            # Calculate metrics on trial-averaged responses
            avg_responses = np.concatenate(
                [np.mean(v['response'], axis=0) for v in grouped_outputs.values()], axis=0
            )
            avg_predictions = np.concatenate(
                [np.mean(v['prediction'], axis=0) for v in grouped_outputs.values()], axis=0
            )
            session_results['corr_to_ave'] = corr(avg_predictions, avg_responses, axis=(0, 1))

            output[data_key] = session_results
    return output


def main(args):
    """
    Main function to run the model evaluation pipeline.
    """
    # --- 1. Initialization and Setup ---
    args = initialize_model_args(args)
    summary = Summary(args.model_name, args.input_data, args.input_type)
    model_paths = create_paths_dict(args.model_data, args.session_ids)

    # --- 2. Compute Dataset Statistics ---
    neurons_idxs = None
    if args.subset_neurons:
        stats_loader = get_loader(
            args, args.dataset, args.batch_size, model_paths,
            input_type=args.input_type, statistics=True
        )
        neurons_idxs = extract_neurons_subset(stats_loader, args.subset_metric)

    input_stats = None
    if args.normalize:
        print("Computing normalization statistics...")
        stats_loader = get_loader(
            args, args.dataset, args.batch_size, model_paths,
            input_type=args.input_type, statistics=True,
            neurons_subset=neurons_idxs
        )
        input_stats = compute_stats(stats_loader, args.normalize)

    directions, singular_values = None, None

    summary._add_dataset_info(neurons_idxs, input_stats, directions, singular_values)

    # --- 3. Load Data for Model Training Context ---
    model_dataloader = get_loader(
        args, args.model_data, args.batch_size, model_paths,
        input_type=args.input_type, input_stats=input_stats,
        neurons_subset=neurons_idxs
    )
    summary._add_training_info(args)
    summary._add_dataloader_info(model_dataloader)

    # --- 4. Build and Restore Model ---
    if args.resnet:
        model = ResNet50Encoder(args, model_dataloader, layer_hook=args.layer_hook)
        args.device = "cpu"
    else:
        model = build_model(
            args, model_dataloader, layer=args.layer, brain_area=args.brain_area
        )
    model.to(args.device)
    print(f"Total trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    scheduler = Scheduler(args, model=model, save_optimizer=False)
    session = list(model.readout.keys())[0]
    kernels_before = model.readout[session].features.detach().clone()
    scheduler.restore(force=True)
    kernels_after = model.readout[session].features.detach().clone()

    if torch.all(torch.eq(kernels_before, kernels_after)):
        raise RuntimeError("Model weights were not restored correctly.")
    print("Model restored successfully.")

    if args.add_readout_info:
        summary._add_readout_info(model)

    # --- 5. Load Data for Evaluation ---
    input_data_path = os.path.join(args.input_data, args.input_type) \
        if args.input_data == 'stringer' else args.input_data
    input_paths = create_paths_dict(input_data_path, args.session_ids)
    input_paths = {input_data_path.split('/')[0]: input_paths[input_data_path]}

    input_dataloader = get_loader(
        args, args.input_data, args.batch_size, input_paths,
        input_type=args.input_data_type, input_stats=input_stats,
        neurons_subset=neurons_idxs
    )

    # --- 6. Run Evaluation and Save Results ---
    frames_to_eval = args.frames - args.skip

    if args.input_data == 'microns30':
        print("Evaluating validation performance...")
        val_performance = eval_model_performance(
            input_dataloader, 'val', model, frames=frames_to_eval,
            eval_repeated=True, skip=args.skip, inference=args.inference,
            save_responses=args.save_responses, use_amp=args.use_amp,
            minibatch_frames=args.minibatch_frames
        )
        summary._add_results_data(val_performance, 'val')

        print("Evaluating test performance...")
        test_performance = eval_model_performance(
            input_dataloader, 'test', model, frames=frames_to_eval,
            eval_repeated=False, skip=args.skip, inference=args.inference,
            save_responses=args.save_responses, use_amp=args.use_amp,
            minibatch_frames=args.minibatch_frames
        )
        summary._add_results_data(test_performance, 'test')

    elif args.input_data == 'sensorium':
        print("Evaluating validation performance...")
        val_performance = eval_model_performance_sensorium(
            input_dataloader, 'val', model, frames=frames_to_eval,
            skip=args.skip, inference=args.inference,
            save_responses=args.save_responses
        )
        summary._add_results_data(val_performance, 'val')

    if args.eval_train:
        print("Evaluating train performance...")
        train_performance = eval_model_performance(
            input_dataloader, 'train', model, frames=frames_to_eval,
            eval_repeated=False, skip=args.skip, inference=args.inference,
            save_responses=args.save_responses, use_amp=args.use_amp,
            minibatch_frames=args.minibatch_frames
        )
        summary._add_results_data(train_performance, 'train')

    # --- 7. Save Summary ---
    output_path = os.path.join('save', 'results', args.model_name)
    os.makedirs(output_path, exist_ok=True)
    summary.save_to_pickle(os.path.join(output_path, f'{args.filename}.pkl'))
    print(f"Evaluation summary saved to {output_path}/{args.filename}.pkl")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a neural network model.")
    parser.add_argument("--output_dir", type=str, default="save/models",
                        help="Directory where models are saved.")
    parser.add_argument("--model_data", type=str, required=True,
                        help="Path to the dataset used for model training context.")
    parser.add_argument("--input_data", type=str, required=True,
                        help="Path to the dataset for evaluation.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model to evaluate.")
    parser.add_argument("--input_data_type", type=str, required=True,
                        help="Type of input data (e.g., 'natural_movie_one').")
    parser.add_argument("--filename", type=str, default='summary',
                        help="Output filename for the summary pickle file.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for data loaders.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use for computation ('cuda' or 'cpu').")
    parser.add_argument("--frames", type=int, default=150,
                        help="Total number of frames in a trial.")
    parser.add_argument("--skip", type=int, default=50,
                        help="Number of initial frames to skip from analysis.")
    parser.add_argument("--minibatch_frames", type=int, default=300,
                        help="Frames per minibatch for memory management.")
    parser.add_argument("--add_readout_info", action='store_true',
                        help="Include readout weights in the summary.")
    parser.add_argument("--removeSpont", action='store_true',
                        help="Flag to remove spontaneous activity (if applicable).")
    parser.add_argument("--inference", type=bool, default=True,
                        help="Save model predictions in the output.")
    parser.add_argument("--save_responses", action='store_true',
                        help="Save ground-truth responses in the output.")
    parser.add_argument("--eval_train", action='store_true',
                        help="Also evaluate performance on the training set.")
    parser.add_argument("--use_amp", action='store_true',
                        help="Use Automatic Mixed Precision for evaluation.")
    parser.add_argument("--no_behavior", action='store_true',
                        help="Flag to exclude behavioral data.")
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_arguments()
    main(cli_args)
