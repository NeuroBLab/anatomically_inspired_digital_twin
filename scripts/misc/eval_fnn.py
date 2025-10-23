import argparse
import os

import numpy as np
import torch
import tqdm
from fnn import microns

from src.metrics import (CCmax, CCnorm, corr, explainable_var,
                                        fev, fev_e, fv_e)
from src.data.loaders import TowardsLoader, get_loader
from src.data.utils import compute_stats
from src.data.summary import Summary
from scripts.utils import create_paths_dict

# Constants for repeated trial structure in the validation set
TRIALS = 6
REPEATS = 10


def get_model_output(model, **kwargs):
    """Generates model predictions using the fnn.microns model interface.

    This function serves as a wrapper to align the fnn model's `predict` method
    with the data structure used in the evaluation pipeline.

    Args:
        model: An instance of the pre-trained fnn.microns model.
        **kwargs: A dictionary of keyword arguments containing the input data,
            expected to include 'videos' and 'responses', and optionally
            'behavior' and 'pupil_center'.

    Returns:
        A tuple containing:
            - np.ndarray: The model's predictions with a restored batch dimension.
            - np.ndarray: The ground-truth neural responses.
    """
    stimuli = np.squeeze(kwargs['videos'].numpy().astype('uint8'))
    behavior = None
    if 'behavior' in kwargs:
        behavior = np.squeeze(kwargs['behavior'].numpy())

    perspective = None
    if 'pupil_center' in kwargs:
        perspective = np.squeeze(kwargs['pupil_center'].numpy())

    model_output = model.predict(
        stimuli=stimuli,
        perspectives=perspective,
        modulations=behavior
    )
    # Restore the batch dimension for consistency with the pipeline.
    model_output = np.expand_dims(model_output, axis=0)

    ground_truth = kwargs['responses'].numpy()

    return model_output, ground_truth


def eval_model_performance(dataloaders, tier, frames, eval_repeated, skip=0,
                           inference=True, save_responses=True, use_amp=False):
    """Evaluates model performance for a given data tier (e.g., 'val', 'test').

    Args:
        dataloaders (dict): A dictionary of data loaders for each tier.
        tier (str): The tier to evaluate (e.g., 'val', 'test').
        frames (int): The number of frames per trial after skipping.
        eval_repeated (bool): If True, assumes data has a repeated trial
            structure and computes metrics accordingly (e.g., CCmax).
        skip (int, optional): Number of initial frames to skip in each trial.
        inference (bool, optional): If True, saves model predictions.
        save_responses (bool, optional): If True, saves ground-truth responses.
        use_amp (bool, optional): If True, uses automatic mixed precision.

    Returns:
        dict: A dictionary where keys are data session IDs and values are
              dictionaries of computed performance metrics.
    """
    output = {}
    with torch.cuda.amp.autocast(enabled=use_amp), torch.no_grad():
        for data_key, loader in dataloaders[tier].items():
            session_result = {}
            responses, predictions = [], []
            session, scan = int(data_key[0]), int(data_key[-1])

            # A specific scan in the dataset is known to be problematic.
            if session == 7 and scan == 4:
                continue

            session_model, _ = microns.scan(session, scan)
            for data in tqdm.tqdm(loader, desc=f"Evaluating {data_key}"):
                batch_kwargs = data._asdict() if hasattr(data, '_asdict') else data
                pred_chunk, gt_chunk = get_model_output(
                    session_model, **batch_kwargs
                )
                responses.append(gt_chunk[:, skip:, :])
                predictions.append(pred_chunk[:, skip:, :])

            if eval_repeated:
                # Reshape for repeated trial analysis.
                # Shape: (Trials, Repeats, Neurons, Frames)
                responses = np.concatenate(responses).reshape(
                    TRIALS, REPEATS, frames, -1).swapaxes(2, 3)
                predictions = np.concatenate(predictions).reshape(
                    TRIALS, REPEATS, frames, -1).swapaxes(2, 3)

                session_result['CCmax'] = CCmax(responses)
                session_result['ev'] = explainable_var(responses)
                session_result['fev'] = fev(responses)
                session_result['oracle_score'] = loader.dataset.neurons.pearson
                corr_to_ave = corr(
                    responses.mean(axis=1),
                    predictions.mean(axis=1),
                    axis=(0, 2)
                )
                session_result['corr_to_ave'] = corr_to_ave
                session_result['fve'] = fv_e(responses, predictions)
                session_result['feve'] = fev_e(responses, predictions)
                ccnorm = CCnorm(responses, predictions)
                session_result['CCnorm'] = ccnorm

                print(f"\nSession {session}, Scan {scan} Results:")
                print(f"  Median CCnorm: {np.nanmedian(ccnorm):.4f}")
                print(f"  Mean Corr-to-Avg: {np.nanmean(corr_to_ave):.4f}")

                final_resp, final_pred = responses, predictions
                axis = (0, 1, 3)
            else:
                final_resp = np.concatenate(responses, axis=0)
                final_pred = np.concatenate(predictions, axis=0)
                axis = (0, 1)

            session_result['corr'] = corr(final_pred, final_resp, axis=axis)
            session_result['mse'] = np.mean((final_pred - final_resp)**2, axis=axis)
            session_result['rmse'] = np.sqrt(session_result['mse'])

            if inference:
                session_result['prediction'] = final_pred
            if save_responses:
                session_result['responses'] = final_resp

            output[data_key] = session_result
    return output


def main(args):
    """Main function to run the evaluation pipeline."""
    summary = Summary(args.model_name, args.input_data, args.input_type)
    model_paths = create_paths_dict(args.model_data, args.session_ids)

    input_stats = None
    if args.normalize:
        print("Computing normalization statistics...")
        stats_dataloader = get_loader(
            args,
            args.dataset,
            args.batch_size,
            model_paths,
            input_type=args.input_type,
            statistics=True,
        )
        input_stats = compute_stats(stats_dataloader, args.normalize)

    loader = TowardsLoader(
        args.batch_size,
        args,
        args.input_type,
        input_stats
    )
    loader = loader.load(model_paths['microns30'])
    summary.add_dataloader_info(loader)

    print("\nEvaluating validation performance...")
    val_performance = eval_model_performance(
        loader,
        'val',
        eval_repeated=True,
        frames=args.frames - args.skip,
        skip=args.skip,
        inference=args.inference,
        save_responses=args.save_responses,
        use_amp=args.use_amp,
    )
    summary.add_results(val_performance, 'val')

    print("\nEvaluating test performance...")
    test_performance = eval_model_performance(
        loader,
        'test',
        eval_repeated=False,
        frames=args.frames - args.skip,
        skip=args.skip,
        inference=args.inference,
        save_responses=args.save_responses,
        use_amp=args.use_amp,
    )
    summary.add_results(test_performance, 'test')

    path = os.path.join('save', 'results', args.model_name)
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, f'{args.filename}.pkl')
    summary.save_to_pickle(output_file)
    print(f"\nEvaluation summary saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate FNN model performance on the Microns dataset."
    )
    parser.add_argument('--dataset', type=str, default='microns30')
    parser.add_argument('--input_data', type=str, default='microns30')
    parser.add_argument('--input_type', type=str, default='clips')
    parser.add_argument('--model_data', type=str, default='microns30')
    parser.add_argument('--model_name', type=str, default='towards_wang')
    parser.add_argument("--filename", type=str, default='summary')
    parser.add_argument('--session_ids', type=list, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--skip', type=int, default=50)
    parser.add_argument('--include_behavior', action='store_true')
    parser.add_argument('--include_pupil_centers', action='store_true')
    parser.add_argument("--inference", type=bool, default=True)
    parser.add_argument('--save_responses', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument(
        '--normalize',
        nargs="+",
        type=str,
        default=[],
        choices=['batch', 'channels', 'neurons', 'height', 'width',
                 'coordinates', 'frames'],
        help="Normalize input data, keeping specified dimensions."
    )
    main(parser.parse_args())
