import os
import typing as t
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer


class Scheduler:
    """Manages LR scheduling, checkpointing, and early stopping."""

    def __init__(
        self,
        args,
        model: nn.Module,
        optimizer: Optimizer,
        scaler: t.Optional[GradScaler] = None,
        mode: t.Literal["min", "max"] = "max",
        max_reduce: int = 2,
        lr_patience: int = 10,
        factor: float = 0.3,
        min_epochs: int = 0,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        module_names: t.Optional[t.List[str]] = None,
    ):
        """Initializes the Scheduler.

        Args:
            args: Namespace object from argparse. Expected to have attributes:
                `warmup_epochs`, `lr_init`, `output_dir`, `device`, `verbose`,
                and `model_name`.
            model: The PyTorch model to be managed.
            optimizer: The optimizer for the model.
            scaler: (Optional) Gradient scaler for mixed-precision training.
            mode: One of {'min', 'max'}. In 'min' mode, the learning rate will
                be reduced when the quantity monitored has stopped decreasing;
                in 'max' mode it will be reduced when the quantity has stopped
                increasing.
            max_reduce: Maximum number of learning rate reductions before
                early stopping is triggered.
            lr_patience: Number of epochs with no improvement after which
                learning rate will be reduced.
            factor: Factor by which the learning rate will be reduced.
                `new_lr = lr * factor`.
            min_epochs: Minimum number of epochs to run before early stopping
                can be triggered.
            save_optimizer: If True, save optimizer and scaler states in the
                checkpoint.
            save_scheduler: If True, save the scheduler's state in the
                checkpoint.
            module_names: (Optional) A list of module names in the model to
                save. If None, the entire model state is saved.
        """
        assert mode in ("min", "max"), f"Mode must be 'min' or 'max', not {mode}."
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.mode = mode
        self.max_reduce = max_reduce
        self.lr_patience = lr_patience
        self.factor = factor
        self.min_epochs = min_epochs
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.module_names = module_names

        # Extract parameters from args
        self.warmup_epochs = args.warmup_epochs
        self.initial_lr = args.lr_init
        self.device = args.device
        self.verbose = args.verbose
        self.name = args.model_name

        # State tracking variables
        self.num_reduce = 0
        self.lr_wait = 0
        self.best_value = np.inf if mode == "min" else -np.inf
        self.best_epoch = 0

        self.checkpoint_dir = os.path.join(args.output_dir, "ckpt")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.warmup_epochs > 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.initial_lr / self.warmup_epochs

    def _get_model_state_dict_for_saving(self) -> OrderedDict:
        """Filters model state_dict based on self.module_names."""
        full_state_dict = self.model.state_dict()
        if self.module_names is None:
            return full_state_dict

        # Filter parameters to save only specified modules
        filtered_parameters = OrderedDict()
        for key, value in full_state_dict.items():
            module_name = key.split(".")[0]
            if module_name in self.module_names:
                filtered_parameters[key] = value
        return filtered_parameters

    def save_checkpoint(self, value: float, epoch: int):
        """Saves the model checkpoint.

        Args:
            value: The validation metric value for the current epoch.
            epoch: The current epoch number.
        """
        filename = os.path.join(self.checkpoint_dir, f"model_state_{self.name}.pt")
        ckpt = {
            "epoch": epoch,
            "value": value,
            "model": self._get_model_state_dict_for_saving(),
        }
        if self.save_optimizer:
            ckpt["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                ckpt["scaler"] = self.scaler.state_dict()
        if self.save_scheduler:
            ckpt["scheduler"] = self.state_dict()

        torch.save(ckpt, f=filename)
        if self.verbose:
            print(f"\nCheckpoint saved to {filename}.")

    def restore(
        self,
        force: bool = False,
        load_optimizer: bool = False,
        load_scheduler: bool = False,
    ) -> int:
        """Loads the best model checkpoint from the checkpoint directory.

        Args:
            force: If True, raise an error if no checkpoint is found.
            load_optimizer: If True, load optimizer and scaler states.
            load_scheduler: If True, load the scheduler's state.

        Returns:
            The epoch number of the loaded checkpoint, or 0 if not found.
        """
        epoch = 0
        filename = os.path.join(self.checkpoint_dir, f"model_state_{self.name}.pt")
        pth_filename = os.path.join(self.checkpoint_dir, f"model_state_{self.name}.pth")

        if os.path.exists(filename):
            ckpt = torch.load(filename, map_location=self.device)
            epoch = ckpt["epoch"]

            # Update state_dict for flexible loading (e.g., partial models)
            state_dict = self.model.state_dict()
            state_dict.update(ckpt["model"])
            self.model.load_state_dict(state_dict)

            if load_optimizer and "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                if self.scaler is not None and "scaler" in ckpt:
                    self.scaler.load_state_dict(ckpt["scaler"])
            if load_scheduler and "scheduler" in ckpt:
                self.load_state_dict(ckpt["scheduler"])

            if self.verbose:
                print(
                    f"\nLoaded checkpoint from epoch {epoch} "
                    f"(value: {ckpt['value']:.04f}).\n"
                )
        elif os.path.exists(pth_filename):
            # Handle loading from a .pth file, which may only contain model weights
            ckpt = torch.load(pth_filename, map_location=self.device)
            state_dict = self.model.state_dict()
            state_dict.update(ckpt)
            self.model.load_state_dict(state_dict)
            if self.verbose:
                print("\nLoaded pretrained model weights from .pth file.")
        elif force:
            raise FileNotFoundError(f"Checkpoint not found in {self.checkpoint_dir}.")

        return epoch

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a dictionary."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "model", "scaler")
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the scheduler's state.

        Args:
            state_dict: Scheduler state.
        """
        self.__dict__.update(state_dict)

    def is_better(self, value: t.Union[float, torch.Tensor]) -> bool:
        """Checks if the current value is better than the best recorded value."""
        return (self.mode == "min" and value < self.best_value) or (
            self.mode == "max" and value > self.best_value
        )

    def reduce_lr(self):
        """Reduces learning rate for all parameter groups."""
        self.num_reduce += 1
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = old_lr * self.factor
            param_group["lr"] = new_lr
            if self.verbose:
                print(
                    f"Reducing learning rate of group {i} from {old_lr:.4e} to "
                    f"{new_lr:.4e} (reduction #{self.num_reduce})."
                )

    def step(self, value: t.Union[float, np.ndarray, torch.Tensor], epoch: int) -> bool:
        """Performs a scheduler step.

        Should be called after each validation epoch.

        Args:
            value: The validation metric to monitor.
            epoch: The current epoch number.

        Returns:
            A boolean indicating whether to terminate training.
        """
        terminate = False

        # Linear learning rate warmup
        if 1 < epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            lr = self.initial_lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        if self.is_better(value):
            self.best_value = float(value)
            self.best_epoch = epoch
            self.lr_wait = 0
            self.save_checkpoint(value=float(value), epoch=epoch)
        elif epoch > self.min_epochs:
            self.lr_wait += 1
            if self.lr_wait >= self.lr_patience:
                if self.num_reduce >= self.max_reduce:
                    terminate = True
                    if self.verbose:
                        print(
                            f"\nEarly stopping: Model has not improved after "
                            f"{self.num_reduce} LR reductions."
                        )
                else:
                    # Restore the best model state before reducing learning rate
                    self.restore(load_optimizer=False, load_scheduler=False)
                    self.reduce_lr()
                    self.lr_wait = 0

        return terminate