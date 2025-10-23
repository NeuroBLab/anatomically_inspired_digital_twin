import inspect
import os
import random
import warnings
from functools import partial

import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm
from neuralpredictors.training import LongCycler

from src import metrics as measures
from src.engine.scheduler import Scheduler

class Trainer:
    """
    Manages the training and evaluation of a PyTorch model.

    This class encapsulates the training loop, loss computation, evaluation metrics,
    checkpointing, and integration with Weights & Biases for experiment tracking.
    """

    def __init__(
        self,
        args,
        loss_name: str = "PoissonLoss",
        epoch_loss_name: str = None,
        eval_metric: str = "corr",
        eval_mode: str = "max",
        scale_loss: bool = True,
        avg_loss: bool = False,
        per_neuron: bool = False,
        neuron_norm: bool = True,
        loss_accum_batch_n: int = None,
        maximize: bool = True,
        regularize: bool = True,
        use_wandb: bool = False,
        wandb_username: str = None,
        chpt_save_step: int = 1,
        val_loader_name: str = "val",
        input_stats: dict = None,
        directions: dict = None,
        singular_values: dict = None,
        neuron_idxs: dict = None,
    ):
        """
        Initializes the Trainer.

        Args:
            args: An object containing command-line arguments and configuration.
            loss_name (str): Name of the loss function from `src.metrics`.
            epoch_loss_name (str, optional): Name of the epoch-level loss function.
            eval_metric (str): The primary metric for evaluation and scheduling.
            eval_mode (str): Mode for the scheduler ('min' or 'max').
            scale_loss (bool): Whether to scale the loss by the dataset size.
            avg_loss (bool): Whether to average the loss over neurons.
            per_neuron (bool): Whether the loss is computed per neuron.
            neuron_norm (bool): Whether to normalize metrics by neuron stats.
            loss_accum_batch_n (int, optional): Number of batches for gradient accumulation.
            maximize (bool): Whether the goal is to maximize the evaluation metric.
            regularize (bool): Whether to apply model regularization.
            use_wandb (bool): Whether to use Weights & Biases for logging.
            wandb_username (str, optional): Your Weights & Biases username.
            chpt_save_step (int): Frequency of saving checkpoints (in epochs).
            val_loader_name (str): The key for the validation dataloader.
            input_stats (dict, optional): Precomputed statistics for the input data.
            directions (dict, optional): Precomputed directions (e.g., PCA components).
            singular_values (dict, optional): Precomputed singular values.
            neuron_idxs (dict, optional): Indices for specific neurons.
        """
        # Loss and evaluation settings
        self.loss_name = loss_name
        self.epoch_loss_name = epoch_loss_name
        self.eval_metric = eval_metric
        self.eval_mode = eval_mode
        self.maximize = maximize

        # Loss function options
        self.avg_loss = avg_loss
        self.scale_loss = scale_loss
        self.per_neuron = per_neuron
        self.neuron_norm = neuron_norm
        self.beta = args.beta
        self.loss_accum_batch_n = loss_accum_batch_n
        self.regularize = regularize
        self.normalize_per_area = args.normalize_per_area
        if self.normalize_per_area and "BA" not in self.eval_metric:
            warnings.warn(
                "It is recommended to switch to a BA_avg metric when using area normalization."
            )

        # Training configuration
        self.device = args.device
        self.detach_core = args.detach_core
        self.max_norm = args.max_norm
        self.skip = args.skip
        self.n_frames = args.frames - args.skip
        self.n_components = args.n_components

        # General options
        self.seed = args.seed
        self.verbose = args.verbose
        self.save_checkpoints = args.save_results
        self.checkpoint_save_path = args.output_dir
        self.chpt_save_step = chpt_save_step
        self.val_loader_name = val_loader_name

        # Experiment tracking
        self.use_wandb = use_wandb
        self.wandb_username = wandb_username
        self.wandb_api = None

        # Data-related attributes
        self.input_stats = input_stats
        self.directions = directions
        self.singular_values = singular_values
        self.neuron_idxs = neuron_idxs

    def full_objective(
        self,
        model,
        dataloader,
        loss_function,
        data_key,
        regularize: bool = True,
        scale_loss: bool = True,
        *args,
        **kwargs,
    ):
        """
        Calculates the full objective, including loss and regularization.

        Args:
            model (torch.nn.Module): The model being trained.
            dataloader (dict): Dictionary of dataloaders.
            loss_function (callable): The loss function.
            data_key (str): The key for the current dataloader.
            regularize (bool): Whether to add regularization terms.
            scale_loss (bool): Whether to scale the loss.
            *args: Positional arguments for the model forward pass.
            **kwargs: Keyword arguments for the model forward pass.

        Returns:
            torch.Tensor: The final computed loss.
        """
        model_output, original_data = self.get_model_output(
            model, data_key, *args, **kwargs
        )
        loss = self.evaluate_loss(
            model,
            model_output,
            original_data,
            loss_function,
            dataloader,
            data_key,
            regularize,
            scale_loss,
            self.normalize_per_area,
        )
        return loss

    def get_model_output(self, model, data_key, *args, **kwargs):
        """
        Performs a forward pass and formats the model output and target data.

        Args:
            model (torch.nn.Module): The model.
            data_key (str): The identifier for the current data session.
            *args: Positional arguments for the model, typically (inputs, targets).
            **kwargs: Keyword arguments for the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the model's
                prediction and the ground truth responses.
        """
        model_output = model(args[0], data_key=data_key, **kwargs)
        original_data = args[1].transpose(2, 1)
        return model_output, original_data

    def evaluate_loss(
        self,
        model,
        model_output,
        original_data,
        loss_function,
        dataloader,
        data_key,
        regularize=True,
        scale_loss=True,
        normalize_per_area=False,
    ) -> float:
        """
        Computes the loss, optionally with scaling, regularization, and area normalization.

        Args:
            model (torch.nn.Module): The model.
            model_output (torch.Tensor): The model's predictions.
            original_data (torch.Tensor): The ground truth data.
            loss_function (callable): The loss function to use.
            dataloader (dict): Dictionary of dataloaders.
            data_key (str): The key for the current dataloader.
            regularize (bool): If True, adds regularization terms to the loss.
            scale_loss (bool): If True, scales the loss by the dataset size.
            normalize_per_area (bool): If True, normalizes the loss for each brain area.

        Returns:
            float: The computed loss value.
        """
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / original_data.shape[0])
            if scale_loss
            else 1
        )
        regularizers = self.evaluate_regularizer(model, data_key) if regularize else 0

        if normalize_per_area:
            brain_areas = dataloader[data_key].dataset.neurons.brain_area
            unique_areas = np.unique(brain_areas)

            loss_values = torch.stack(
                [
                    loss_function(
                        model_output[..., brain_areas == area],
                        original_data[..., brain_areas == area],
                        directions=self.directions.get(data_key),
                        singular_values=self.singular_values.get(data_key),
                        stats=self.input_stats[data_key].get("responses")
                        if self.input_stats
                        else None,
                        n_components=self.n_components,
                    )
                    / np.sum(brain_areas == area)
                    for area in unique_areas
                ]
            )
            loss = loss_values.mean() * len(brain_areas)
        else:
            loss = loss_function(
                model_output,
                original_data,
                directions=self.directions.get(data_key),
                singular_values=self.singular_values.get(data_key),
                stats=self.input_stats[data_key].get("responses")
                if self.input_stats
                else None,
                n_components=self.n_components,
            )
        return loss_scale * loss + regularizers

    def evaluate_regularizer(self, model, data_key):
        """
        Computes the regularization loss for the model components.

        Args:
            model (torch.nn.Module): The model.
            data_key (str): The identifier for the current data session.

        Returns:
            torch.Tensor: The total regularization loss.
        """
        regularizers = 0
        if not self.detach_core:
            core_reg = model.core.regularizer()
            regularizers += sum(core_reg) if isinstance(core_reg, tuple) else core_reg
        regularizers += model.readout.regularizer(data_key)
        if hasattr(model, "shifter") and model.shifter is not None:
            regularizers += model.shifter.regularizer(data_key)
        return regularizers

    def initialize_loss_function(self, loss_name):
        """
        Initializes a loss function module from its name.

        Args:
            loss_name (str): The name of the loss function class in `src.metrics`.

        Returns:
            An instance of the specified loss function class.
        """
        loss_cls = getattr(measures, loss_name)
        kwargs = {
            "per_neuron": self.per_neuron,
            "avg": self.avg_loss,
            "neuron_norm": self.neuron_norm,
        }

        init_signature = inspect.signature(loss_cls.__init__)
        if "beta" in init_signature.parameters:
            kwargs["beta"] = self.beta

        return loss_cls(**kwargs)

    @staticmethod
    def get_lr(optimizer):
        """
        Retrieves the current learning rate from the optimizer.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            float: The current learning rate.
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def initialize_wandb(self, args, timeout=180):
        """
        Initializes a Weights & Biases run.

        Args:
            args: The object containing run configurations.
            timeout (int): Timeout for the wandb service.
        """
        self.wandb_api = wandb.Api(timeout=timeout)
        config = vars(args)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=config,
            settings=wandb.Settings(
                api_key=self.wandb_api.api_key, _service_wait=timeout
            ),
        )
        wandb.define_metric(name="Epoch", hidden=True)
        wandb.define_metric(name="Batch", hidden=True)

    @staticmethod
    def update_wandb(epoch_loss, batch_no_tot, epoch, lr, results):
        """
        Logs metrics to Weights & Biases.

        Args:
            epoch_loss (float): The total training loss for the epoch.
            batch_no_tot (int): The total number of batches processed.
            epoch (int): The current epoch number.
            lr (float): The current learning rate.
            results (dict): A dictionary of evaluation results.
        """
        wandb_dict = {
            "train loss": epoch_loss,
            "Batch": batch_no_tot,
            "Epoch": epoch,
            "Learning rate": lr,
        }
        wandb_dict.update({k: v for k, v in results["metrics"].items()})
        wandb.log(wandb_dict)

    def training_step(
        self,
        model,
        dataloader,
        loss_function,
        scaler,
        optimizer,
        batch_args,
        batch_kwargs,
        data_key,
        batch_no,
        optim_step_count,
        use_amp,
    ):
        """
        Performs a single training step, including forward pass, loss calculation,
        and backward pass.

        Args:
            model (torch.nn.Module): The model.
            dataloader (dict): The training dataloaders.
            loss_function (callable): The loss function.
            scaler (torch.cuda.amp.GradScaler): Gradient scaler for AMP.
            optimizer (torch.optim.Optimizer): The optimizer.
            batch_args (list): Positional arguments from the dataloader.
            batch_kwargs (dict): Keyword arguments from the dataloader.
            data_key (str): The key for the current dataloader.
            batch_no (int): The current batch number within the epoch.
            optim_step_count (int): Number of batches for gradient accumulation.
            use_amp (bool): Whether to use automatic mixed precision.

        Returns:
            torch.Tensor: The loss for the current step.
        """
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = self.full_objective(
                model,
                dataloader,
                loss_function,
                data_key,
                self.regularize,
                self.scale_loss,
                *batch_args,
                **batch_kwargs,
            )
            scaler.scale(loss).backward()

        if (batch_no + 1) % optim_step_count == 0:
            scaler.unscale_(optimizer)
            if self.max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        return loss

    def eval_step_gpu(self, model, dataloaders, criterion, switch_mode: bool = True):
        """
        Performs an evaluation step on the GPU, computing metrics without moving
        data to the CPU.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloaders (dict): A dictionary of validation dataloaders.
            criterion (callable): The loss function for evaluation.
            switch_mode (bool): Whether to switch the model to `eval()` mode.

        Returns:
            dict: A dictionary containing aggregated metrics and per-session results.
        """
        training_status = model.training
        if switch_mode:
            model.eval()

        output = {"metrics": {}, "sessions": {}}
        with torch.no_grad():
            for data_key, loader in dataloaders.items():
                brain_areas = dataloaders[data_key].dataset.neurons.brain_area
                session = {
                    "responses": [],
                    "prediction": [],
                    "metrics": {"loss": 0},
                }
                for data in loader:
                    batch_args = list(data)
                    batch_kwargs = data._asdict() if not isinstance(data, dict) else data

                    model_output, original_data = self.get_model_output(
                        model, data_key, *batch_args, **batch_kwargs
                    )

                    batch_loss = self.evaluate_loss(
                        model,
                        model_output,
                        original_data,
                        criterion,
                        dataloaders,
                        data_key,
                        regularize=False,
                        scale_loss=False,
                        normalize_per_area=self.normalize_per_area,
                    )

                    session["metrics"]["loss"] += batch_loss.detach().item()
                    session["responses"].append(original_data[:, self.skip :, :])
                    session["prediction"].append(model_output[:, self.skip :, :])

                responses = torch.cat(session["responses"], dim=0)
                prediction = torch.cat(session["prediction"], dim=0)

                repeats = responses.shape[0] % 60 == 0

                session["responses"] = responses.view(-1, responses.shape[-1])
                session["prediction"] = prediction.view(-1, prediction.shape[-1])

                session["metrics"].update(
                    self.compute_metrics_gpu(
                        session["responses"],
                        session["prediction"],
                        repeats,
                        data_key,
                        brain_areas,
                    )
                )
                output["sessions"][data_key] = session

        output["metrics"] = self.compute_aggregate_metrics_gpu(output["sessions"])
        model.train(training_status)
        return output

    def compute_metrics_gpu(self, responses, prediction, repeats, data_key, brain_areas):
        """
        Computes evaluation metrics on the GPU.

        Args:
            responses (torch.Tensor): Ground truth neural responses.
            prediction (torch.Tensor): Model's predicted responses.
            repeats (bool): Whether the data contains repeated trials.
            data_key (str): Identifier for the data session.
            brain_areas (np.ndarray): Array of brain area labels for each neuron.

        Returns:
            dict: A dictionary of computed metrics.
        """
        metrics = {}
        unique_areas = np.unique(brain_areas)
        corr_brain_area = {
            area: measures.corr_gpu(
                prediction[:, brain_areas == area],
                responses[:, brain_areas == area],
                axis=0,
            ).mean()
            for area in unique_areas
        }

        metrics["corr_BA_avg"] = torch.stack(list(corr_brain_area.values())).mean()
        metrics["corr"] = measures.corr_gpu(prediction, responses, axis=0).mean()
        for area, val in corr_brain_area.items():
            metrics[f"corr_{area}"] = val

        metrics["mse"] = measures.mse_gpu(
            responses, prediction, normalize=self.neuron_norm
        )
        metrics["rmse"] = torch.sqrt(metrics["mse"])
        metrics["poissonloss"] = measures.poissonloss_gpu(
            prediction, responses, normalize=self.neuron_norm
        )

        if repeats:
            responses_r = responses.view(6, 10, self.n_frames, -1).permute(0, 1, 3, 2)
            prediction_r = prediction.view(6, 10, self.n_frames, -1).permute(0, 1, 3, 2)

            corr_to_ave_ba = {
                area: measures.corr_gpu(
                    prediction_r.mean(dim=1)[:, brain_areas == area, :],
                    responses_r.mean(dim=1)[:, brain_areas == area, :],
                    axis=(0, 2),
                ).mean()
                for area in unique_areas
            }
            ccnorm_ba = {
                area: measures.CCnorm_gpu(
                    responses_r[..., brain_areas == area, :],
                    prediction_r[..., brain_areas == area, :],
                ).nanmean()
                for area in unique_areas
            }

            metrics["corr_to_ave_BA_avg"] = torch.stack(
                list(corr_to_ave_ba.values())
            ).mean()
            metrics["CCnorm_BA_avg"] = torch.stack(list(ccnorm_ba.values())).mean()
            metrics["corr_to_ave"] = measures.corr_gpu(
                prediction_r.mean(dim=1), responses_r.mean(dim=1), axis=(0, 2)
            ).mean()
            metrics["CCnorm"] = measures.CCnorm_gpu(responses_r, prediction_r).nanmean()

            for area in unique_areas:
                metrics[f"corr_to_ave_{area}"] = corr_to_ave_ba[area]
                metrics[f"CCnorm_{area}"] = ccnorm_ba[area]

            metrics["fv_e"] = measures.fv_e_gpu(responses_r, prediction_r).nanmean()
            metrics["feve"] = measures.fev_e_gpu(responses_r, prediction_r).nanmean()

        return metrics

    def compute_aggregate_metrics_gpu(self, sessions):
        """
        Aggregates metrics from multiple sessions (GPU version).

        Args:
            sessions (dict): A dictionary of session results.

        Returns:
            dict: A dictionary of metrics averaged across all sessions.
        """
        metrics_keys = list(next(iter(sessions.values()))["metrics"].keys())
        aggregate_metrics = {
            metric: torch.tensor(
                [session["metrics"][metric] for session in sessions.values()]
            )
            .nanmean()
            .item()
            for metric in metrics_keys
        }
        return aggregate_metrics

    @staticmethod
    def save_model(model, checkpoint_save_path, name, clean: bool = False):
        """
        Saves the model's state dictionary to a file.

        Args:
            model (torch.nn.Module): The model to save.
            checkpoint_save_path (str): The directory to save the checkpoint in.
            name (str or int): The name of the checkpoint file (e.g., 'final' or epoch number).
            clean (bool): If True, removes previous epoch checkpoints.
        """
        if clean:
            for filename in os.listdir(checkpoint_save_path):
                if filename.startswith("epoch_") and filename.endswith(".pth"):
                    os.remove(os.path.join(checkpoint_save_path, filename))

        if isinstance(name, int):
            name = f"epoch_{name}"
        torch.save(model.state_dict(), os.path.join(checkpoint_save_path, f"{name}.pth"))

    def _set_seed(self):
        """
        Sets random seeds for reproducibility.
        """
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def train(
        self,
        args,
        model: nn.Module,
        dataloaders: dict,
        seed: int = None,
        epoch: int = 0,
        max_iter: int = 500,
        use_amp: bool = True,
    ):
        """
        The main training loop.

        Args:
            args: Configuration object.
            model (torch.nn.Module): The model to be trained.
            dataloaders (dict): Dictionary of dataloaders ('train', 'val', etc.).
            seed (int, optional): Random seed for reproducibility.
            epoch (int): The starting epoch number.
            max_iter (int): The maximum number of epochs to train for.
            use_amp (bool): Whether to use automatic mixed precision.

        Returns:
            dict: The state dictionary of the best model.
        """
        if seed is not None:
            self._set_seed()

        model.train()

        # Initialize training components
        loss_function = self.initialize_loss_function(self.loss_name)
        n_iterations = len(LongCycler(dataloaders["train"]))

        param_groups = [
            {"params": model.core.parameters(), "weight_decay": args.weight_decay_core},
            {
                "params": model.readout.parameters(),
                "weight_decay": args.weight_decay_readout,
            },
        ]
        if hasattr(model, "shifter") and model.shifter is not None:
            param_groups.append(
                {"params": model.shifter.parameters(), "weight_decay": 0.01}
            )

        optimizer = torch.optim.AdamW(param_groups, lr=args.lr_init)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        scheduler = Scheduler(
            args,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            mode=self.eval_mode,
            max_reduce=args.lr_decay_steps,
            lr_patience=args.patience,
            factor=args.lr_decay_factor,
        )

        if args.restore:
            restore_all = not args.restore_params_only
            epoch = scheduler.restore(
                load_optimizer=restore_all, load_scheduler=restore_all
            )

        optim_step_count = (
            len(dataloaders["train"])
            if args.loss_accum_batch_n is None
            else args.loss_accum_batch_n
        )

        if self.use_wandb:
            self.initialize_wandb(args)

        # --- Training Loop ---
        batch_no_tot = 0
        while (epoch := epoch + 1) < max_iter + 1:
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0

            pbar = tqdm(
                enumerate(LongCycler(dataloaders["train"])),
                total=n_iterations,
                desc=f"Epoch {epoch}",
            )
            for batch_no, (data_key, data) in pbar:
                batch_no_tot += 1
                batch_args = list(data)
                batch_kwargs = data._asdict() if not isinstance(data, dict) else data

                loss = self.training_step(
                    model,
                    dataloaders["train"],
                    loss_function,
                    scaler,
                    optimizer,
                    batch_args,
                    batch_kwargs,
                    data_key,
                    batch_no,
                    optim_step_count,
                    use_amp,
                )

                epoch_loss += loss.detach()
                del loss, data, batch_args, batch_kwargs

            # --- Evaluation Step ---
            eval_results = self.eval_step_gpu(
                model,
                dataloaders[self.val_loader_name],
                loss_function,
            )

            lr = self.get_lr(optimizer)
            early_stop = scheduler.step(eval_results["metrics"][self.eval_metric], epoch)

            if self.use_wandb:
                self.update_wandb(epoch_loss, batch_no_tot, epoch, lr, eval_results)

            if self.verbose:
                print(
                    f"\nEpoch {epoch}, lr={lr:.6f}, train_loss={epoch_loss:.4f}, "
                    f"val_loss={eval_results['metrics']['loss']:.4f}, "
                    f"val_{self.eval_metric}={eval_results['metrics'][self.eval_metric]:.4f}"
                )

            del eval_results
            if early_stop:
                print("Early stopping triggered.")
                break

        # --- Finalization ---
        scheduler.restore()  # Restore the best model
        if self.save_checkpoints:
            self.save_model(model, self.checkpoint_save_path, "final", clean=True)

        return model.state_dict()
