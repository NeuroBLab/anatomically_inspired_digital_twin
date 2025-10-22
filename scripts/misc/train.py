import json
import os
from shutil import rmtree

import torch

from src.layers.encoder import ResNet50Encoder
from src.engine.build import build_model
from src.data.loaders import get_loader
from src.data.utils import compute_stats, extract_neurons_subset
from src.engine.training import Trainer, TwoPartTrainer
from scripts.utils import create_data_paths, set_random_seed
from train_args import get_parser


def update_runs_file():
    """Creates or increments a run number in 'save/models/runs.txt'.

    This function ensures each experiment has a unique, sequential run number,
    which is used for naming output directories.

    Returns:
        int: The new, unique run number for the current experiment.
    """
    runs_file = os.path.join("save", "models", "runs.txt")
    os.makedirs(os.path.dirname(runs_file), exist_ok=True)

    new_number = 1
    if os.path.isfile(runs_file):
        try:
            with open(runs_file, "r") as f:
                numbers = [
                    int(line.strip())
                    for line in f
                    if line.strip().isdigit()
                ]
                if numbers:
                    new_number = max(numbers) + 1
        except (IOError, ValueError):
            # If file is empty, corrupt, or unreadable, start from 1.
            pass

    with open(runs_file, "a") as f:
        f.write(str(new_number) + "\n")

    return new_number


def main(args):
    """Sets up and runs the model training pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments specifying the
            training configuration.
    """
    set_random_seed(args.seed)

    try:
        # --- Setup output directory and save configuration ---
        run_no = update_runs_file()
        if args.model_name is None:
            args.model_name = str(run_no)

        args.output_dir = os.path.join(args.output_dir, args.model_name)

        if args.save_results:
            if args.clear_output_dir and os.path.isdir(args.output_dir):
                rmtree(args.output_dir)

            if os.path.isdir(args.output_dir) and not args.restore:
                overwrite = input(
                    "Model name exists and --restore is False. "
                    "Overwrite? (y/n): "
                )
                if overwrite.lower() != "y":
                    print("Exiting. Please specify a new `model_name`.")
                    return

            os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
            args_path = os.path.join(args.output_dir, "args.json")
            with open(args_path, "w") as f:
                json.dump(vars(args), f, indent=4)

        if args.wandb_name is None:
            args.wandb_name = f"model_{args.model_name}"

        # --- Load data and compute statistics ---
        paths_dict = {
            dat: create_data_paths(dat, args.session_ids)
            for dat in args.dataset
        }

        neurons_idxs = None
        if args.subset_neurons:
            stats_loader = get_loader(
                args,
                args.dataset,
                args.batch_size,
                paths_dict,
                input_type=args.input_type,
                statistics=True,
            )
            neurons_idxs = extract_neurons_subset(
                stats_loader, args.subset_metric
            )

        input_stats = None
        if args.normalize:
            stats_loader = get_loader(
                args,
                args.dataset,
                args.batch_size,
                paths_dict,
                input_type=args.input_type,
                statistics=True,
                neurons_subset=neurons_idxs,
            )
            input_stats = compute_stats(stats_loader, args.normalize)

        directions = None
        singular_values = None
        loss_requires_cov = args.loss in [
            "SinglePCALoss",
            "PoissonSinglePCALoss",
        ]
        epoch_loss_requires_cov = args.epoch_loss in ["CosineSimilarityLoss"]
        if args.evaluate_directions or loss_requires_cov or epoch_loss_requires_cov:
            stats_loader = get_loader(
                args,
                args.dataset,
                args.batch_size,
                paths_dict,
                input_type=args.input_type,
                statistics=True,
                neurons_subset=neurons_idxs,
            )

        dataloaders = get_loader(
            args,
            args.dataset,
            args.batch_size,
            paths_dict,
            input_type=args.input_type,
            input_stats=input_stats,
            neurons_subset=neurons_idxs,
        )

        # --- Build model ---
        if args.resnet:
            model = ResNet50Encoder(
                args, dataloaders, layer_hook=args.layer_hook
            )
            model.to(args.device)
            print(model)
            print(
                "Total number of parameters: ",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
        else:
            model = build_model(
                args,
                dataloaders,
                layer=args.layer,
                brain_area=args.brain_area,
            )

        # --- Initialize trainer ---
        if not args.two_part_model:
            trainer = Trainer(
                args,
                loss_name=args.loss,
                epoch_loss_name=args.epoch_loss,
                eval_metric=args.eval_metric,
                eval_mode=args.eval_mode,
                avg_loss=args.avg_loss,
                neuron_norm=args.neuron_norm,
                use_wandb=args.use_wandb,
                wandb_username="darioliscai",
                val_loader_name="val",
                regularize=args.regularize,
                input_stats=input_stats,
                directions=directions,
                singular_values=singular_values,
                neuron_idxs=neurons_idxs,
            )
        else:
            trainer = TwoPartTrainer(
                args,
                loss_name=args.loss,
                use_wandb=args.use_wandb,
                wandb_username="darioliscai",
                val_loader_name="val",
                regularize=args.regularize,
            )

        # --- Start training ---
        print("Starting training procedure...")
        trainer.train(
            args, model, dataloaders, max_iter=args.epochs, use_amp=args.use_amp
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if args.save_results and os.path.isdir(args.output_dir):
            delete_folder = input(
                "Do you want to delete the created output folder? (y/n): "
            )
            if delete_folder.lower() == "y":
                rmtree(args.output_dir)
                print(f"Removed directory: {args.output_dir}")
            else:
                print("Output folder was not deleted.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
