import argparse
import os

import numpy as np
import torch

from src.metrics import (
    CCmax,
    CCnorm,
    corr,
    explainable_var,
    fev,
    fev_e,
    fv_e,
)
from src.engine.build import build_model
from src.data.utils import compute_stats, extract_neurons_subset
from src.data.loaders import get_loader
from src.engine.scheduler import Scheduler
from src.data.summary import Summary
from scripts.utils import create_paths_dict, initialize_model_args

# Constants for reshaping data from repeated trials.
TRIALS = 6
REPEATS = 10


def get_model_output(
    model, data_key, minibatch_frames, inputs, responses, **kwargs
):
    """
    Generates model predictions for a given input, processing in minibatches.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        data_key (str): The identifier for the current data session.
        minibatch_frames (int): The number of frames to process in each batch
            to manage memory usage.
        inputs (torch.Tensor): The input data tensor (e.g., images).
        responses (torch.Tensor): The ground-truth neural responses.
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


def eval_model_performance(
    dataloaders,
    tier,
    model,
    frames,
    eval_repeated,
    skip=0,
    inference=True,
    save_responses=True,
    use_amp=False,
    minibatch_frames=300,
):
    """
    Evaluates the model's performance on a specific data tier (e.g., 'val').

    Args:
        dataloaders (dict): A dictionary of data loaders for each tier.
        tier (str): The data tier to evaluate (e.g., 'val', 'test').
        model (torch.nn.Module): The model to be evaluated.
        frames (int): The number of frames per trial.
        eval_repeated (bool): Whether the data includes repeated trials,
            requiring specialized metrics.
        skip (int, optional): Number of initial frames to skip in evaluation.
        inference (bool, optional): If True, stores model predictions.
        save_responses (bool, optional): If True, stores ground-truth responses.
        use_amp (bool, optional): If True, uses automatic mixed precision.
        minibatch_frames (int, optional): Frame count for minibatch processing.

    Returns:
        dict: A dictionary containing performance metrics for each data session.
    """
    model.eval()
    output_metrics = {}
    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            for data_key, dataloader in dataloaders[tier].items():
                print(f"Evaluating session: {data_key}")
                try:
                    session_metrics = {}
                    all_responses = []
                    all_predictions = []

                    for data in dataloader:
                        batch_kwargs = (
                            data._asdict()
                            if not isinstance(data, dict)
                            else data
                        )
                        predictions, ground_truth = get_model_output(
                            model,
                            data_key,
                            minibatch_frames,
                            **batch_kwargs,
                        )
                        all_responses.append(
                            ground_truth.cpu().numpy()[:, skip:, :]
                        )
                        all_predictions.append(
                            predictions.cpu().numpy()[:, skip:, :]
                        )

                    if eval_repeated:
                        # Reshape from (T*R, F, N) to (T, R, N, F)
                        responses = (
                            np.concatenate(all_responses)
                            .reshape(TRIALS, REPEATS, frames, -1)
                            .swapaxes(2, 3)
                        )
                        predictions = (
                            np.concatenate(all_predictions)
                            .reshape(TRIALS, REPEATS, frames, -1)
                            .swapaxes(2, 3)
                        )

                        session_metrics["CCmax"] = CCmax(responses)
                        session_metrics["ev"] = explainable_var(responses)
                        session_metrics["fev"] = fev(responses)
                        session_metrics["oracle_score"] = (
                            dataloader.dataset.neurons.pearson
                        )
                        session_metrics["corr_to_ave"] = corr(
                            predictions.mean(axis=1),
                            responses.mean(axis=1),
                            axis=(0, 2),
                        )
                        session_metrics["fve"] = fv_e(responses, predictions)
                        session_metrics["feve"] = fev_e(responses, predictions)
                        session_metrics["CCnorm"] = CCnorm(
                            responses, predictions
                        )
                        axis = (0, 1, 3)
                    else:
                        responses = np.concatenate(all_responses, axis=0)
                        predictions = np.concatenate(all_predictions, axis=0)
                        axis = (0, 1)

                    session_metrics["corr"] = corr(
                        predictions, responses, axis=axis
                    )
                    session_metrics["mse"] = np.mean(
                        (predictions - responses) ** 2, axis=axis
                    )
                    session_metrics["rmse"] = np.sqrt(session_metrics["mse"])

                    if inference:
                        session_metrics["prediction"] = predictions
                    if save_responses:
                        session_metrics["responses"] = responses

                    output_metrics[data_key] = session_metrics

                except Exception as e:
                    print(
                        f"Could not evaluate {data_key} due to an error: {e}"
                    )
                    continue
    return output_metrics


def main(args):
    """
    Main function to run the model evaluation pipeline.
    """
    args = initialize_model_args(args)
    summary = Summary(args.model_name, args.input_data, args.input_type)

    # --- Data and Model Setup ---
    model_paths = create_paths_dict(args.model_data, args.session_ids)

    neurons_idxs = None
    if args.subset_neurons:
        stats_dataloader = get_loader(
            args,
            args.dataset,
            args.batch_size,
            model_paths,
            input_type=args.input_type,
            statistics=True,
        )
        neurons_idxs = extract_neurons_subset(
            stats_dataloader, args.subset_metric
        )

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
            neurons_subset=neurons_idxs,
        )
        input_stats = compute_stats(stats_dataloader, args.normalize)

    summary._add_dataset_info(neurons_idxs, input_stats)

    model_dataloader = get_loader(
        args,
        args.model_data,
        args.batch_size,
        model_paths,
        input_type=args.input_type,
        input_stats=input_stats,
        neurons_subset=neurons_idxs,
    )

    summary._add_training_info(args)
    summary._add_dataloader_info(model_dataloader)

    model = build_model(
        args,
        model_dataloader,
        layer=args.layer,
        brain_area=args.brain_area,
    )
    model.to(args.device)

    # --- Load Evaluation Data ---
    input_data_path = (
        os.path.join(args.input_data, args.input_type)
        if args.input_data == "stringer"
        else args.input_data
    )
    input_paths = create_paths_dict(input_data_path, args.session_ids)
    input_paths = {input_data_path.split("/")[0]: input_paths[input_data_path]}

    input_dataloader = get_loader(
        args,
        args.input_data,
        args.batch_size,
        input_paths,
        input_type=args.input_data_type,
        input_stats=input_stats,
        neurons_subset=neurons_idxs,
    )

    # --- Restore Model from Checkpoint ---
    scheduler = Scheduler(args, model=model, save_optimizer=False)
    session = list(model.readout.keys())[0]
    kernels_before_restore = model.readout[session].features.detach().clone()
    scheduler.restore(force=True)
    kernels_after_restore = model.readout[session].features.detach().clone()

    # Verify that the model weights have changed after restoring.
    if torch.all(torch.eq(kernels_before_restore, kernels_after_restore)):
        raise RuntimeError("Model was not restored correctly from checkpoint.")
    else:
        print("Model restored successfully from checkpoint.")

    if args.add_readout_info:
        summary._add_readout_info(model)

    # --- Performance Evaluation ---
    print("\nEvaluating validation performance...")
    val_performance = eval_model_performance(
        input_dataloader,
        "val",
        model,
        eval_repeated=args.input_data == "microns30",
        frames=args.frames - args.skip,
        skip=args.skip,
        inference=args.inference,
        save_responses=args.save_responses,
        use_amp=args.use_amp,
        minibatch_frames=args.minibatch_frames,
    )
    summary._add_results_data(val_performance, "val")

    print("\nEvaluating test performance...")
    test_performance = eval_model_performance(
        input_dataloader,
        "test",
        model,
        eval_repeated=False,
        frames=args.frames - args.skip,
        skip=args.skip,
        inference=args.inference,
        save_responses=args.save_responses,
        use_amp=args.use_amp,
        minibatch_frames=args.minibatch_frames,
    )
    summary._add_results_data(test_performance, "test")

    if args.eval_train:
        print("\nEvaluating train performance...")
        train_performance = eval_model_performance(
            input_dataloader,
            "train",
            model,
            eval_repeated=False,
            frames=args.frames - args.skip,
            skip=args.skip,
            inference=args.inference,
            save_responses=args.save_responses,
            use_amp=args.use_amp,
            minibatch_frames=args.minibatch_frames,
        )
        summary._add_results_data(train_performance, "train")

    # --- Save Results ---
    output_path = os.path.join("save", "results", args.model_name)
    os.makedirs(output_path, exist_ok=True)
    summary_file = os.path.join(output_path, f"{args.filename}.pkl")
    summary.save_to_pickle(summary_file)
    print(f"\nEvaluation summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-trained model of mouse visual cortex."
    )
    parser.add_argument(
        "--model_data", type=str, required=True, help="Path to model data."
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to input data for evaluation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model, used for saving results.",
    )
    parser.add_argument(
        "--input_data_type",
        type=str,
        required=True,
        help="Type of input data (e.g., 'stimuli').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="save/models",
        help="Directory for model outputs.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="summary",
        help="Filename for the output summary pickle file.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Computation device ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for data loaders."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=150,
        help="Total number of frames per trial.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=50,
        help="Number of initial frames to skip from evaluation.",
    )
    parser.add_argument(
        "--minibatch_frames",
        type=int,
        default=300,
        help="Number of frames per minibatch for inference.",
    )
    parser.add_argument(
        "--add_readout_info",
        action="store_true",
        help="Add readout layer information to the summary.",
    )
    parser.add_argument(
        "--removeSpont",
        action="store_true",
        help="Flag to remove spontaneous activity.",
    )
    parser.add_argument(
        "--no_inference",
        dest="inference",
        action="store_false",
        help="Do not save model predictions in the summary.",
    )
    parser.add_argument(
        "--save_responses",
        action="store_true",
        help="Save ground-truth responses in the summary.",
    )
    parser.add_argument(
        "--eval_train",
        action="store_true",
        help="Also evaluate performance on the training set.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision for evaluation.",
    )
    parser.add_argument(
        "--no_behavior",
        action="store_true",
        help="Flag to exclude behavioral data.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
