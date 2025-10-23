import argparse
import ast
import sys

def get_parser(allow_missing_required=False):
    """Constructs the argument parser for the training script.

    Conditionally adds arguments based on the selected model core and readout
    to avoid conflicts and keep the interface clean.

    Args:
        allow_missing_required (bool, optional): If True, allows missing
            required arguments, which is useful for interactive environments
            like Jupyter notebooks. Defaults to False.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser()

    # --- Storing settings ---
    parser.add_argument(
        "-s",
        "--save_results",
        action="store_true",
        help="Save model checkpoints and plots to the output directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="save",
        help="Path to directory where files will be saved.",
    )
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="Overwrite content in the specified output directory.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore model, optimizer, and scheduler from a checkpoint.",
    )
    parser.add_argument(
        "--restore_params_only",
        action="store_true",
        help="Restore only model parameters from a checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to restore and save.",
    )

    # --- Dataset settings ---
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["microns30"],
        choices=["microns", "sensorium", "microns30", "sensorium_old"],
        help="Name of the dataset(s) to use.",
    )
    parser.add_argument(
        "--session_ids",
        nargs="+",
        type=str,
        default=None,
        help="Specific session IDs to use for training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Cache data to speed up data loading.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames to sample from each training video.",
    )
    parser.add_argument(
        "--val_frames",
        type=int,
        default=150,
        help="Number of frames for validation/test videos.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation.",
    )
    parser.add_argument(
        "--eval_gpu",
        action="store_true",
        help="Compute evaluation metrics on the GPU.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="all",
        choices=["all", "natural", "parametric", "spontaneous", "clips"],
        help="Type of input data to use for training.",
    )
    parser.add_argument(
        "--removeSpont",
        action="store_true",
        help="Subtract spontaneous activity from neural responses.",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=1024,
        help="Number of components for covariance decomposition.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=1,
        choices=[0, 1],
        help="Resize image mode: 0=no resize, 1=resize to (1, 36, 64).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        nargs="+",
        default=None,
        help="Layer(s) to extract features from.",
    )
    parser.add_argument(
        "--brain_area",
        type=str,
        nargs="+",
        default=None,
        help="Brain area(s) to filter the data.",
    )
    parser.add_argument(
        "--subset_neurons",
        action="store_true",
        help="Use only a subset of neurons based on a quality metric.",
    )
    parser.add_argument(
        "--subset_metric",
        type=str,
        default="fev",
        choices=["fev", "CCmax"],
        help="Metric for selecting neuron subset ('fev' or 'CCmax').",
    )
    parser.add_argument(
        "--normalize",
        nargs="+",
        type=str,
        default=[],
        help="Normalize input data across specified dimensions.",
    )
    parser.add_argument(
        "--include_behavior",
        action="store_true",
        help="Include behavioral data (e.g., running speed) in training.",
    )
    parser.add_argument(
        "--include_pupil_centers",
        action="store_true",
        help="Include pupil center coordinates in training.",
    )
    parser.add_argument(
        "--include_pupil_centers_as_channels",
        action="store_true",
        help="Include pupil centers as input channels.",
    )
    parser.add_argument(
        "--augment_data",
        action="store_true",
        help="Perform data augmentation on input stimuli.",
    )

    # --- Training settings ---
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--two_part_model",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use TwoPartModel training procedure.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="PoissonLoss",
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--avg_loss", action="store_true", help="Average the loss."
    )
    parser.add_argument(
        "--evaluate_directions",
        action="store_true",
        help="Train on cosine similarity between data and prediction directions.",
    )
    parser.add_argument(
        "--epoch_loss",
        type=str,
        default=None,
        help="Loss function to apply at the end of each epoch.",
    )
    parser.add_argument(
        "--neuron_norm",
        action="store_true",
        help="Normalize loss by the number of neurons.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable automatic mixed precision training.",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="corr",
        help="Metric for the learning rate scheduler.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="max",
        choices=["max", "min"],
        help="Scheduler evaluation mode ('max' or 'min').",
    )
    parser.add_argument(
        "--lr_init", type=float, default=5e-3, help="Initial learning rate."
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Warmup epochs for learning rate scheduler.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="LR scheduler patience: epochs with no improvement before LR decay.",
    )
    parser.add_argument(
        "--beta", type=float, default=1, help="Beta value for the loss function."
    )
    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        default=0.3,
        help="Factor to decay learning rate by.",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=4,
        help="Number of times learning rate is allowed to decay.",
    )
    parser.add_argument(
        "--loss_accum_batch_n",
        type=int,
        default=None,
        help="Number of sessions to accumulate gradients over.",
    )
    parser.add_argument(
        "--weight_decay_core",
        type=float,
        default=0.01,
        help="AdamW weight decay for the core module.",
    )
    parser.add_argument(
        "--weight_decay_readout",
        type=float,
        default=0.01,
        help="AdamW weight decay for the readout module.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=-1,
        help="Offset to start training from.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed training information."
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=None,
        help="If set, applies gradient clipping with this max norm.",
    )
    parser.add_argument(
        "--regularize",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable regularization in the loss function.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N frames when evaluating correlation.",
    )
    parser.add_argument(
        "--detach_core", action="store_true", help="Freeze core parameters."
    )
    parser.add_argument(
        "--normalize_per_area",
        action="store_true",
        help="Normalize loss based on the number of neurons per brain area.",
    )

    # --- Weights & Biases settings ---
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--wandb_project", default="model_runs", help="W&B project name."
    )
    parser.add_argument(
        "--wandb_entity", default="NeuroBLab", help="W&B entity name."
    )
    parser.add_argument(
        "--wandb_name", default=None, help="W&B run name."
    )

    # --- Model settings ---
    parser.add_argument(
        "--resnet", action="store_true", help="Use a ResNet50 encoder."
    )
    parser.add_argument(
        "--layer_hook",
        type=str,
        default="layer1",
        help="Layer to hook for feature extraction from ResNet50.",
    )
    parser.add_argument(
        "--core", type=str, default=None, help="The core module to use."
    )
    parser.add_argument(
        "--readout", type=str, default=None, help="The readout module to use."
    )
    parser.add_argument(
        "--shifter",
        action="store_true",
        help="Apply a shifter module to the readout.",
    )
    parser.add_argument(
        "--modulator",
        action="store_true",
        help="Apply a modulator for behavior integration.",
    )
    parser.add_argument(
        "--gru", action="store_true", help="Apply a GRU to the core module."
    )
    parser.add_argument(
        "--nonlinearity",
        type=str,
        choices=["elu", "identity", "exp", "learnablesoftplus"],
        default="elu",
        help="Final nonlinearity of the model.",
    )

    # Parse known args to conditionally add model-specific hyperparameters
    temp_args, _ = parser.parse_known_args()

    # --- Core hyperparameters ---
    if temp_args.core == "conv3d":
        parser.add_argument(
            "--hidden_channels",
            nargs="+",
            type=int,
            default=[16, 32, 64, 128],
        )
        parser.add_argument("--input_kernel", nargs="+", type=int, default=(9, 9, 9))
        parser.add_argument("--hidden_kernel", nargs="+", type=int, default=(9, 7, 7))
        parser.add_argument("--layers", type=int, default=4)
        parser.add_argument("--batch_norm", type=int, default=1, choices=[0, 1])
        parser.add_argument("--layer_norm", action="store_true")
        parser.add_argument("--gamma_input_spatial", type=float, default=0)
        parser.add_argument("--gamma_input_temporal", type=float, default=0)
        parser.add_argument("--padding", type=int, default=1, choices=[0, 1])
        parser.add_argument("--use_residuals", action="store_true")
        parser.add_argument("--pooling", action="store_true")
        parser.add_argument("--pooling_layers", nargs="+", type=int, default=None)
        parser.add_argument("--dropout_rate", type=float, default=0)

    if temp_args.core == "factorizedconv3d":
        parser.add_argument(
            "--hidden_channels",
            nargs="+",
            type=int,
            default=[16, 32, 64, 128],
        )
        parser.add_argument("--spatial_input_kernel", nargs="+", type=int, default=11)
        parser.add_argument("--temporal_input_kernel", nargs="+", type=int, default=11)
        parser.add_argument("--spatial_hidden_kernel", nargs="+", type=int, default=5)
        parser.add_argument("--temporal_hidden_kernel", nargs="+", type=int, default=5)
        parser.add_argument("--layers", type=int, default=4)
        parser.add_argument("--batch_norm", type=int, default=1, choices=[0, 1])
        parser.add_argument("--gamma_input_spatial", type=float, default=0)
        parser.add_argument("--gamma_input_temporal", type=float, default=0)
        parser.add_argument("--padding", type=int, default=1, choices=[0, 1])
        parser.add_argument("--use_residuals", type=int, default=0, choices=[0, 1])
        parser.add_argument("--stack", nargs="+", type=int, default=None)
        parser.add_argument("--two_streams", action="store_true")
        parser.add_argument("--pooling", action="store_true")
        parser.add_argument("--pooling_layers", nargs="+", type=int, default=-1)
        parser.add_argument("--dropout_rate", type=float, default=0)
        parser.add_argument("--intermediate_bn", action="store_true")
        parser.add_argument(
            "--behavior_integration_mode",
            type=str,
            choices=["mul", "sum"],
            default="mul",
        )

    if temp_args.core == "stackedconv2d":
        parser.add_argument("--hidden_channels", type=int, default=64)
        parser.add_argument("--input_kernel", nargs="+", type=int, default=7)
        parser.add_argument("--hidden_kernel", nargs="+", type=int, default=5)
        parser.add_argument("--layers", type=int, default=3)
        parser.add_argument("--batch_norm", type=int, default=1, choices=[0, 1])
        parser.add_argument("--gamma_input", type=float, default=0)
        parser.add_argument("--gamma_hidden", type=float, default=0)
        parser.add_argument("--stack", type=int, nargs="+", default=None)
        parser.add_argument("--padding", type=int, default=1, choices=[0, 1])

    # --- Readout hyperparameters ---
    if temp_args.readout in ["gaussian", "twopartgaussian", "factorized"]:
        parser.add_argument("--disable_grid_predictor", action="store_true")
        parser.add_argument(
            "--grid_predictor_dim", type=int, default=2, choices=[2, 3]
        )
        parser.add_argument("--grid_predictor_hidden_layers", type=int, default=1)
        parser.add_argument("--grid_predictor_hidden_features", type=int, default=30)
        parser.add_argument("--readout_bias", type=int, default=0, choices=[0, 1])
        parser.add_argument("--readout_reg_weight", type=float, default=0.0076)
        parser.add_argument("--dispersion_reg", type=float, default=0.0076)
        parser.add_argument("--regularize_per_layer", action="store_true")
        parser.add_argument("--init_gaussian", action="store_true")
        if temp_args.readout == "gaussian":
            parser.add_argument("--num_samples", type=int, default=1)

    parser.add_argument(
        "--brain_area_to_layer",
        type=ast.literal_eval,
        default=None,
        help="Dict mapping brain areas to layers, e.g., \"{'V1': [1, 2]}\"",
    )

    # --- Optional module hyperparameters ---
    if temp_args.shifter:
        parser.add_argument("--hidden_channels_shifter", type=int, default=5)
        parser.add_argument("--shift_layers", type=int, default=3)
        parser.add_argument("--gamma_shifter", type=float, default=0)

    if temp_args.modulator:
        parser.add_argument("--concat_behavior", action="store_true")
        parser.add_argument("--modulator_layers", nargs="+", type=int, default=None)
        parser.add_argument(
            "--modulator_output_channels", nargs="+", type=int, default=None
        )
        parser.add_argument("--modulator_hidden_channels", type=int, default=None)
        parser.add_argument("--modulator_hidden_layers", type=int, default=1)
        parser.add_argument("--modulator_bias", type=int, default=1)
        parser.add_argument("--modulator_dropout", type=float, default=0.1)
        parser.add_argument("--modulator_gamma", type=float, default=0)
        parser.add_argument(
            "--modulator_activation",
            type=str,
            choices=["relu", "tanh", "sigmoid"],
            default="tanh",
        )
        parser.add_argument("--modulator_temporal_conv", action="store_true")
        parser.add_argument("--modulator_temporal_kernel", type=int, default=5)

    if temp_args.gru:
        parser.add_argument("--rec_channels", type=int, default=None)
        parser.add_argument("--rec_input_kern", type=int, default=3)
        parser.add_argument("--rec_kern", type=int, default=3)
        parser.add_argument("--gamma_rec", type=float, default=0)
        parser.add_argument("--rec_padding", type=int, default=1, choices=[0, 1])

    if temp_args.augment_data:
        parser.add_argument("--alpha_affine", type=float, default=0.5)
        parser.add_argument("--rotation_max", type=float, default=20)
        parser.add_argument("--scale_max", type=float, default=0.2)
        parser.add_argument("--translation_max", type=float, default=0.1)
        parser.add_argument("--augment_per_frame", action="store_true")

    del temp_args

    # Handle missing required arguments when running in an interactive session
    if allow_missing_required and "ipykernel" in sys.modules:
        temp_args_dict, _ = vars(parser.parse_known_args([]))
        for action in parser._actions:
            if action.required and action.dest not in temp_args_dict:
                temp_args_dict[action.dest] = None
        return argparse.Namespace(**temp_args_dict)

    return parser