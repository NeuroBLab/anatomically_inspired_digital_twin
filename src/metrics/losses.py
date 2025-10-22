import warnings

import numpy as np
import torch
from torch import linalg as LA
from torch import nn
from torch.nn.functional import cosine_similarity


class PoissonLoss(nn.Module):
    """Computes the Poisson negative log-likelihood loss."""

    def __init__(
        self,
        bias: float = 1e-7,
        per_neuron: bool = False,
        avg: bool = False,
        full_loss: bool = False,
        neuron_norm: bool = True,
    ):
        """Initializes the PoissonLoss module.

        Args:
            bias (float, optional): A small value added to the predicted rate
                for numerical stability. Defaults to 1e-7.
            per_neuron (bool, optional): If True, returns a loss value for each
                neuron. Defaults to False.
            avg (bool, optional): If True, the loss is averaged over the batch.
                If False, it is summed. Defaults to False.
            full_loss (bool, optional): If True, includes the Stirling
                approximation term in the loss calculation. Defaults to False.
            neuron_norm (bool, optional): If True and `per_neuron` is False,
                the total loss is normalized by the number of neurons.
                Defaults to True.
        """
        super().__init__()
        self.bias = bias
        self.full_loss = full_loss
        self.per_neuron = per_neuron
        self.avg = avg
        self.neuron_norm = neuron_norm
        if self.avg:
            warnings.warn(
                "PoissonLoss is averaged per batch. It's recommended to use "
                "`sum` instead for stability."
            )

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the loss.

        Args:
            output (torch.Tensor): The predicted rate from the model of shape
                (batch_size, n_timesteps, n_neurons).
            target (torch.Tensor): The ground truth data of the same shape.

        Returns:
            torch.Tensor: The computed scalar or vector loss.
        """
        _, _, n_neurons = output.shape
        target = target.detach()
        loss = nn.PoissonNLLLoss(
            log_input=False, full=self.full_loss, eps=self.bias, reduction="none"
        )(output, target)

        if not self.per_neuron:
            if self.avg:
                loss = loss.mean()
            else:
                loss = loss.sum() / n_neurons if self.neuron_norm else loss.sum()
        else:
            loss = loss.view(-1, loss.shape[-1])
            loss = loss.mean(dim=0) if self.avg else loss.sum(dim=0)

        assert not (
            torch.isnan(loss).any() or torch.isinf(loss).any()
        ), "NaN or inf value encountered in PoissonLoss!"
        return loss


class MSELoss(nn.Module):
    """Computes the Mean Squared Error loss, normalized by number of neurons."""

    def __init__(self, neuron_norm: bool = True):
        """Initializes the MSELoss module.

        Args:
            neuron_norm (bool, optional): If True, the standard MSE value is
                additionally divided by the number of neurons. Defaults to True.
        """
        super().__init__()
        self.neuron_norm = neuron_norm

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the loss.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        _, _, n_neurons = output.shape
        target = target.detach()
        loss = nn.MSELoss(reduction="mean")(output, target)
        if self.neuron_norm:
            loss = loss / n_neurons
        return loss


class RMSELoss(nn.Module):
    """Computes the Root Mean Squared Error loss."""

    def __init__(
        self,
        per_neuron: bool = False,
        avg: bool = False,
        neuron_norm: bool = True,
    ):
        """Initializes the RMSELoss module.

        Args:
            per_neuron (bool, optional): If True, returns a loss value for each
                neuron. Defaults to False.
            avg (bool, optional): If True, the loss is averaged over the batch.
                If False, it is summed. Defaults to False.
            neuron_norm (bool, optional): If True and `per_neuron` is False,
                the total loss is normalized by the number of neurons.
                Defaults to True.
        """
        super().__init__()
        self.per_neuron = per_neuron
        self.avg = avg
        self.neuron_norm = neuron_norm

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the loss.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.

        Returns:
            torch.Tensor: The computed scalar or vector loss.
        """
        _, _, n_neurons = output.shape
        target = target.detach()
        loss = torch.sqrt(nn.MSELoss(reduction="none")(output, target))

        if not self.per_neuron:
            if self.avg:
                loss = loss.mean()
            else:
                loss = loss.sum() / n_neurons if self.neuron_norm else loss.sum()
        else:
            loss = loss.view(-1, loss.shape[-1])
            loss = loss.mean(dim=0) if self.avg else loss.sum(dim=0)
        return loss


class CrossCorrelationMSE(nn.Module):
    """Computes MSE between the neuron-neuron cross-correlation matrices."""

    def __init__(self, neuron_norm: bool = True):
        """Initializes the CrossCorrelationMSE module.

        Args:
            neuron_norm (bool, optional): If True, normalizes the loss by the
                number of unique neuron pairs. Defaults to True.
        """
        super().__init__()
        self.neuron_norm = neuron_norm

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the loss.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        _, _, n_neurons = output.shape
        epsilon = 1e-6
        target = target.detach()

        # Reshape to (n_neurons, n_samples)
        target = target.reshape(-1, n_neurons).T
        output = output.reshape(-1, n_neurons).T

        # Disable automatic mixed precision for stability in matrix operations
        with torch.cuda.amp.autocast(enabled=False):
            target = target.float()
            output = output.float()

            # Center data
            target_centered = target - torch.mean(target, dim=1, keepdim=True)
            output_centered = output - torch.mean(output, dim=1, keepdim=True)

            # Compute covariance matrices
            cov_target = torch.matmul(
                target_centered, target_centered.T
            ) / (target.size(1) - 1)
            cov_output = torch.matmul(
                output_centered, output_centered.T
            ) / (output.size(1) - 1)

            # Compute correlation from covariance
            target_std = torch.sqrt(torch.diag(cov_target)).unsqueeze(1)
            output_std = torch.sqrt(torch.diag(cov_output)).unsqueeze(1)
            corr_target = cov_target / (
                torch.matmul(target_std, target_std.T) + epsilon
            )
            corr_output = cov_output / (
                torch.matmul(output_std, output_std.T) + epsilon
            )

            loss = nn.MSELoss(reduction="sum")(corr_output, corr_target)
            if self.neuron_norm:
                # Normalize by number of off-diagonal elements
                loss = loss / (n_neurons * (n_neurons - 1) / 2)

        assert not (
            torch.isnan(loss).any() or torch.isinf(loss).any()
        ), "NaN or inf value encountered in CrossCorrelationMSE!"
        return loss


class PoissonCorrelationLoss(nn.Module):
    """A composite loss combining Poisson NLL and Cross-Correlation MSE."""

    def __init__(
        self,
        per_neuron: bool = False,
        avg: bool = False,
        alpha: float = 1.0,
        beta: float = 1.0,
        neuron_norm: bool = True,
    ):
        """Initializes the PoissonCorrelationLoss module.

        Args:
            per_neuron (bool, optional): Passed to PoissonLoss. Defaults to False.
            avg (bool, optional): Passed to PoissonLoss. Defaults to False.
            alpha (float, optional): Weight for the PoissonLoss component.
                Defaults to 1.0.
            beta (float, optional): Weight for the CrossCorrelationMSE component.
                Defaults to 1.0.
            neuron_norm (bool, optional): Passed to both component losses.
                Defaults to True.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.poisson_loss_fn = PoissonLoss(
            per_neuron=per_neuron, avg=avg, neuron_norm=neuron_norm
        )
        self.cc_loss_fn = CrossCorrelationMSE(neuron_norm=neuron_norm)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the combined loss.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        poisson_loss = self.alpha * self.poisson_loss_fn(output, target)
        cc_loss = self.beta * self.cc_loss_fn(output, target)
        return poisson_loss + cc_loss


class CCALoss(nn.Module):
    """Computes Canonical Correlation Analysis (CCA) loss."""

    def __init__(self):
        """Initializes the CCALoss module."""
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the negative sum of canonical correlations.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        _, _, n_neurons = output.shape
        epsilon = 1e-6
        target = target.detach()

        # Reshape to (n_neurons, n_samples)
        target = target.reshape(-1, n_neurons).T
        output = output.reshape(-1, n_neurons).T

        with torch.cuda.amp.autocast(enabled=False):
            target = target.float()
            output = output.float()

            # Center the data
            target_centered = target - target.mean(dim=1, keepdim=True)
            output_centered = output - output.mean(dim=1, keepdim=True)

            # Compute covariance matrices
            c_xx = torch.matmul(
                output_centered, output_centered.T
            ) / (output.size(1) - 1) + epsilon * torch.eye(
                output.size(0), device=output.device
            )
            c_yy = torch.matmul(
                target_centered, target_centered.T
            ) / (target.size(1) - 1) + epsilon * torch.eye(
                target.size(0), device=target.device
            )
            c_xy = (
                torch.matmul(output_centered, target_centered.T)
                / (output.size(1) - 1)
            )

            # Compute inverse square root of covariance matrices
            c_xx_inv_sqrt = LA.inv(LA.sqrtm(c_xx))
            c_yy_inv_sqrt = LA.inv(LA.sqrtm(c_yy))

            # Compute the matrix for SVD
            t_matrix = torch.matmul(c_xx_inv_sqrt, torch.matmul(c_xy, c_yy_inv_sqrt))

            # Sum of singular values are the canonical correlations
            cca_corr = torch.sum(LA.svd(t_matrix, compute_uv=False))

            # Loss is negative correlation (to be minimized)
            loss = -cca_corr

        assert not (
            torch.isnan(loss).any() or torch.isinf(loss).any()
        ), "NaN or inf value encountered in CCALoss!"
        return loss


class PCALoss(nn.Module):
    """Computes RMSE between principal components of the output and target."""

    def __init__(
        self,
        num_eigenvectors: int = 32,
        project: bool = True,
        neuron_norm: bool = True,
    ):
        """Initializes the PCALoss module.

        Args:
            num_eigenvectors (int, optional): Number of top principal
                components to compare. Defaults to 32.
            project (bool, optional): If True, compares the data projected onto
                the PCs. If False, compares the PC vectors directly.
                Defaults to True.
            neuron_norm (bool, optional): If True, normalizes the final RMSE
                by the number of eigenvectors. Defaults to True.
        """
        super().__init__()
        self.num_eigenvectors = num_eigenvectors
        self.project = project
        self.neuron_norm = neuron_norm

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the loss.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        _, _, n_neurons = output.shape
        target = target.detach()

        # Reshape to (n_neurons, n_samples)
        target = target.reshape(-1, n_neurons).T
        output = output.reshape(-1, n_neurons).T

        with torch.cuda.amp.autocast(enabled=False):
            target = target.float()
            output = output.float()

            # Center the data
            target_centered = target - target.mean(dim=1, keepdim=True)
            output_centered = output - output.mean(dim=1, keepdim=True)

            # Compute covariance matrices
            cov_x = torch.matmul(
                output_centered, output_centered.T
            ) / (output_centered.size(1) - 1)
            cov_y = torch.matmul(
                target_centered, target_centered.T
            ) / (target_centered.size(1) - 1)

            # Eigendecomposition to find principal components
            eigvals_x, eigvecs_x = LA.eigh(cov_x)
            eigvals_y, eigvecs_y = LA.eigh(cov_y)

            # Sort eigenvectors by eigenvalues in descending order
            sorted_indices_x = torch.argsort(eigvals_x, descending=True)
            sorted_indices_y = torch.argsort(eigvals_y, descending=True)

            pcs_x = eigvecs_x[:, sorted_indices_x][:, : self.num_eigenvectors]
            pcs_y = eigvecs_y[:, sorted_indices_y][:, : self.num_eigenvectors]

            if self.project:
                # Project data onto the principal components
                projected_x = torch.matmul(pcs_x.T, output_centered)
                projected_y = torch.matmul(pcs_y.T, target_centered)
                loss_input_x, loss_input_y = projected_x, projected_y
            else:
                # Compare the principal component vectors directly
                loss_input_x, loss_input_y = pcs_x, pcs_y

            mse = nn.MSELoss(reduction="sum")(loss_input_x, loss_input_y)
            if self.neuron_norm:
                rmse_loss = torch.sqrt(mse / self.num_eigenvectors)
            else:
                rmse_loss = torch.sqrt(mse)

        assert not (
            torch.isnan(rmse_loss).any() or torch.isinf(rmse_loss).any()
        ), "NaN or inf value encountered in PCALoss!"
        return rmse_loss


class PoissonPCALoss(nn.Module):
    """A composite loss combining Poisson NLL and PCA-based RMSE."""

    def __init__(
        self,
        per_neuron: bool = False,
        avg: bool = False,
        alpha: float = 1.0,
        beta: float = 100.0,
        num_eigenvectors: int = 32,
        project: bool = True,
        neuron_norm: bool = True,
    ):
        """Initializes the PoissonPCALoss module.

        Args:
            per_neuron (bool, optional): Passed to PoissonLoss. Defaults to False.
            avg (bool, optional): Passed to PoissonLoss. Defaults to False.
            alpha (float, optional): Weight for the PoissonLoss component.
                Defaults to 1.0.
            beta (float, optional): Weight for the PCALoss component.
                Defaults to 100.0.
            num_eigenvectors (int, optional): Passed to PCALoss. Defaults to 32.
            project (bool, optional): Passed to PCALoss. Defaults to True.
            neuron_norm (bool, optional): Passed to both component losses.
                Defaults to True.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.poisson_loss_fn = PoissonLoss(
            per_neuron=per_neuron, avg=avg, neuron_norm=neuron_norm
        )
        self.pca_loss_fn = PCALoss(
            num_eigenvectors=num_eigenvectors,
            project=project,
            neuron_norm=neuron_norm,
        )

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the combined loss.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        poisson_loss = self.alpha * self.poisson_loss_fn(output, target)
        pca_loss = self.beta * self.pca_loss_fn(output, target)
        return poisson_loss + pca_loss


class SinglePCALoss(nn.Module):
    """Computes MSE between data projected onto pre-computed directions."""

    def __init__(self, neuron_norm: bool = True):
        """Initializes the SinglePCALoss module.

        Args:
            neuron_norm (bool, optional): If True, normalizes the loss by the
                number of projection directions. Defaults to True.
        """
        super().__init__()
        self.neuron_norm = neuron_norm

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        directions: torch.Tensor,
        singular_values: torch.Tensor,
        stats: dict,
    ) -> torch.Tensor:
        """Computes the loss.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.
            directions (torch.Tensor): Pre-computed projection directions (e.g., PCs).
            singular_values (torch.Tensor): Singular values associated with directions.
            stats (dict): A dictionary with 'mean' and 'std' of the target data
                for normalization.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        device = output.device
        _, _, n_neurons = output.shape
        n_dims, _ = directions.shape
        target = target.detach()

        mean = torch.from_numpy(stats["mean"]).to(device)
        std = stats["std"]

        # Compute precision for whitening, with a floor for stability
        threshold = 0.01 * np.nanmean(std)
        response_precision = np.ones_like(std) / (threshold + 1e-8)
        stable_std_idx = std > threshold
        response_precision[stable_std_idx] = 1 / (std[stable_std_idx] + 1e-8)
        response_precision = torch.from_numpy(response_precision).to(device)

        # Whiten the data
        target_whitened = (
            target.reshape(-1, n_neurons).T - mean
        ) * response_precision
        output_whitened = (
            output.reshape(-1, n_neurons).T - mean
        ) * response_precision

        # Project whitened data onto scaled directions
        scaled_directions = directions.to(device) / singular_values.to(device).reshape(
            n_dims, -1
        )
        target_projected = torch.matmul(scaled_directions, target_whitened)
        output_projected = torch.matmul(scaled_directions, output_whitened)

        loss = nn.MSELoss(reduction="sum")(output_projected, target_projected)
        if self.neuron_norm:
            loss = loss / n_dims
        return loss


class PoissonSinglePCALoss(nn.Module):
    """Composite loss combining Poisson NLL and SinglePCALoss."""

    def __init__(
        self,
        per_neuron: bool = False,
        avg: bool = False,
        alpha: float = 0.5,
        beta: float = 100.0,
        neuron_norm: bool = True,
    ):
        """Initializes the PoissonSinglePCALoss module.

        Args:
            per_neuron (bool, optional): Passed to PoissonLoss. Defaults to False.
            avg (bool, optional): Passed to PoissonLoss. Defaults to False.
            alpha (float, optional): Weight for the PoissonLoss component.
                The SinglePCALoss weight is `(1 - alpha) * beta`. Defaults to 0.5.
            beta (float, optional): Scaling factor for the SinglePCALoss component.
                Defaults to 100.0.
            neuron_norm (bool, optional): Passed to both component losses.
                Defaults to True.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.poisson_loss_fn = PoissonLoss(
            per_neuron=per_neuron, avg=avg, neuron_norm=neuron_norm
        )
        self.pca_loss_fn = SinglePCALoss(neuron_norm=neuron_norm)

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        directions: torch.Tensor,
        stats: dict,
    ) -> torch.Tensor:
        """Computes the combined loss.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.
            directions (torch.Tensor): Passed to SinglePCALoss.
            stats (dict): Passed to SinglePCALoss.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        poisson_loss = self.alpha * self.poisson_loss_fn(output, target)
        pca_loss = (
            (1 - self.alpha)
            * self.beta
            * self.pca_loss_fn(output, target, directions, stats)
        )
        return poisson_loss + pca_loss


class CosineSimilarityLoss(nn.Module):
    """Computes 1 minus the cosine similarity between two sets of vectors."""

    def __init__(self):
        """Initializes the CosineSimilarityLoss module."""
        super().__init__()

    @staticmethod
    def _extract_components(x: torch.Tensor, n_components: int) -> torch.Tensor:
        """Extracts top principal components via SVD.

        Args:
            x (torch.Tensor): Input data of shape (n_samples, n_features).
            n_components (int): Number of components to extract.

        Returns:
            torch.Tensor: Top `n_components` principal axes (eigenvectors).
        """
        x_centered = x - x.mean(dim=0, keepdim=True)
        _, _, v_h = torch.linalg.svd(x_centered, full_matrices=False)
        return v_h[:n_components]

    @staticmethod
    def _compute_cosine_similarity(
        output: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """Computes cosine similarity between two tensors.

        Args:
            output (torch.Tensor): First tensor.
            target (torch.Tensor): Second tensor.
            reduction (str, optional): Reduction method ('mean' or 'sum').
                Defaults to 'mean'.

        Returns:
            torch.Tensor: The reduced similarity score.
        """
        if output.shape != target.shape:
            raise ValueError("Input tensors must have the same shape.")
        similarities = cosine_similarity(output, target, dim=1)
        return similarities.mean() if reduction == "mean" else similarities.sum()

    def forward(
        self,
        output: torch.Tensor,
        directions: torch.Tensor,
        extract: bool = False,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the loss.

        Args:
            output (torch.Tensor): Model output. Can be either raw data or
                pre-extracted components.
            directions (torch.Tensor): The target components to compare against.
            extract (bool, optional): If True, principal components are
                extracted from `output` before comparison. Defaults to False.
            reduction (str, optional): Reduction method for similarity scores.
                Defaults to 'mean'.

        Returns:
            torch.Tensor: The computed scalar loss (1 - similarity).
        """
        device = output.device
        n_components = directions.shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            if extract:
                output_directions = self._extract_components(output, n_components)
            else:
                output_directions = output
            target_directions = directions.to(device)

            similarity = self._compute_cosine_similarity(
                output_directions, target_directions, reduction=reduction
            )
        return 1.0 - similarity


class PoissonCosineLoss(nn.Module):
    """Composite loss of Poisson NLL and cosine similarity of PCs."""

    def __init__(
        self,
        per_neuron: bool = False,
        avg: bool = False,
        alpha: float = 0.5,
        beta: float = 1e4,
        neuron_norm: bool = True,
    ):
        """Initializes the PoissonCosineLoss module.

        Args:
            per_neuron (bool, optional): Passed to PoissonLoss. Defaults to False.
            avg (bool, optional): Passed to PoissonLoss. Defaults to False.
            alpha (float, optional): Weight for the PoissonLoss component.
                The CosineLoss weight is `(1 - alpha) * beta`. Defaults to 0.5.
            beta (float, optional): Scaling factor for the CosineLoss component.
                Defaults to 1e4.
            neuron_norm (bool, optional): Passed to PoissonLoss. Defaults to True.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.poisson_loss_fn = PoissonLoss(
            per_neuron=per_neuron, avg=avg, neuron_norm=neuron_norm
        )
        self.cosine_loss_fn = CosineSimilarityLoss()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        n_components: int,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the combined loss.

        Args:
            output (torch.Tensor): The predicted values from the model.
            target (torch.Tensor): The ground truth data.
            n_components (int): Number of principal components to extract and compare.
            reduction (str, optional): Reduction method for similarity.
                Defaults to 'mean'.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        target = target.detach()

        with torch.cuda.amp.autocast(enabled=False):
            # Reshape to (n_samples, n_neurons) for component extraction
            target_reshaped = target.reshape(-1, target.shape[-1])
            output_reshaped = output.reshape(-1, output.shape[-1])

            # Extract directions (PCs) from target data
            target_directions = self.cosine_loss_fn._extract_components(
                target_reshaped, n_components
            )

            # Compute cosine similarity loss
            cosine_loss = self.cosine_loss_fn(
                output_reshaped,
                target_directions,
                extract=True,
                reduction=reduction,
            )

        poisson_loss = self.alpha * self.poisson_loss_fn(output, target)
        scaled_cosine_loss = (1 - self.alpha) * self.beta * cosine_loss
        return poisson_loss + scaled_cosine_loss