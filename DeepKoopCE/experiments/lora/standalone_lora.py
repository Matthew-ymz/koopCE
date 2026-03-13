"""
Standalone LoRA (Low-Rank Approximation) for Koopman Operator Learning

This is a self-contained implementation extracted from NeuralKoopmanSVD project.
No external dependencies on the original project structure.

Reference:
    Jeong et al., "Efficient Parametric SVD of Koopman Operator for Stochastic 
    Dynamical Systems", NeurIPS 2025.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


# =============================================================================
# Mathematical Helper Functions
# =============================================================================

def compute_second_moment(f: torch.Tensor, seq_nesting: bool = False) -> torch.Tensor:
    """
    Computes the second moment matrix M = 1/N * f^T @ f.

    If seq_nesting is True, applies partial stop-gradients for sequential nesting:
      - Lower Triangle (i > j): <f_i, sg(f_j)> (Current optimizes against fixed past)
      - Upper Triangle (i < j): <sg(f_i), f_j> (Future optimizes against fixed current)
      - Diagonal: <f_i, f_i> (Full gradient)

    Args:
        f: Tensor of shape (Batch, Modes).
        seq_nesting: If True, applies sequential nesting gradient blocking.

    Returns:
        Tensor of shape (Modes, Modes).
    """
    batch_size = f.shape[0]

    if not seq_nesting:
        return (f.T @ f) / batch_size
    else:
        # Lower: i > j
        M_lower = torch.tril(f.T @ f.detach(), diagonal=-1)
        # Upper: i < j
        M_upper = torch.triu(f.detach().T @ f, diagonal=1)
        # Diagonal
        M_diag = torch.diag((f * f).sum(dim=0))
        return (M_lower + M_upper + M_diag) / batch_size


def kostic_regularization(
    M: torch.Tensor, h_mean: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the metric distortion regularization (Kostic et al.).

    Loss = ||M - I||_F^2 + 2 * ||mean(h)||^2 (if h_mean provided)

    Args:
        M: Covariance/Moment matrix (Modes, Modes).
        h_mean: Optional mean vector of features (Modes,).
    
    Returns:
        Regularization loss.
    """
    k = M.shape[0]
    eye = torch.eye(k, device=M.device, dtype=M.dtype)
    reg = ((M - eye) ** 2).sum()

    if h_mean is not None:
        reg = reg + 2 * (h_mean**2).sum()
    return reg


# =============================================================================
# LoRA Loss Function
# =============================================================================

class NestedLoRALoss(nn.Module):
    """
    Nested Low-Rank Approximation (LoRA) Loss for Koopman Operator Learning.

    This loss minimizes the low-rank approximation error of the Koopman operator
    without requiring matrix inversions or singular value decompositions.

    The objective is:
        L(f, g) = -2 * Tr(T[f, g]) + Tr(M_ρ₀[f] @ M_ρ₁[g])

    Where:
        - T[f, g] = E[f @ g^T]: Cross-correlation between current and future features
        - M_ρ₀[f] = E[f @ f^T]: Auto-correlation of current features
        - M_ρ₁[g] = E[g @ g^T]: Auto-correlation of future features

    Args:
        n_modes (int): The rank k of the approximation (number of singular modes).
        nesting (str, optional): The nesting strategy to learn ordered singular functions.
            - 'jnt' (Joint): Optimizes all modes simultaneously with a mask.
            - 'seq' (Sequential): Optimizes modes iteratively via stop-gradient.
            - None: Standard LoRA without ordering guarantees. Defaults to 'jnt'.
        reg_weight (float): Weight for Kostic regularization. Defaults to 0.0.

    References:
        Jeong et al., "Efficient Parametric SVD of Koopman Operator", NeurIPS 2025.
    """

    def __init__(
        self, 
        n_modes: int, 
        nesting: str = "jnt", 
        reg_weight: float = 0.0
    ):
        super().__init__()
        assert nesting in [None, "jnt", "seq"], \
            "nesting must be one of [None, 'jnt', 'seq']"
        
        self.nesting = nesting
        self.reg_weight = reg_weight
        self.n_modes = n_modes

        # Create joint nesting masks if needed
        if nesting == "jnt":
            vec_mask, mat_mask = self._create_joint_masks(n_modes)
            self.register_buffer("vec_mask", vec_mask.unsqueeze(0))
            self.register_buffer("mat_mask", mat_mask)
        else:
            self.vec_mask = None
            self.mat_mask = None

    def _create_joint_masks(self, n_modes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates weight masks for joint nesting strategy.
        
        The masks implement a progressive weighting scheme where earlier modes
        receive higher weights, encouraging the network to learn them first.
        """
        weights = np.ones(n_modes) / n_modes
        vec_mask_np = np.cumsum(weights[::-1])[::-1].copy()
        vec_tensor = torch.tensor(vec_mask_np, dtype=torch.float32)
        mat_tensor = torch.minimum(
            vec_tensor.unsqueeze(1), 
            vec_tensor.unsqueeze(0)
        )
        return vec_tensor, mat_tensor

    def forward(
        self, 
        f: torch.Tensor, 
        g: torch.Tensor, 
        svals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the LoRA loss.

        Args:
            f: Encoder output for current state (batch_size, n_modes).
            g: Encoder output for future state (batch_size, n_modes).
            svals: Optional learned singular values (n_modes,).

        Returns:
            Scalar loss value.
        """
        # Scale by learned singular values if provided
        if svals is not None:
            scale = svals.unsqueeze(0).sqrt()
            f = f * scale
            g = g * scale

        # 1. Correlation Term: -2 * Tr(T[f, g])
        if self.nesting == "jnt":
            corr = -2 * (self.vec_mask * f * g).mean(dim=0).sum()
        else:
            corr = -2 * (f * g).mean(dim=0).sum()

        # 2. Metric Term: Tr(M_f @ M_g)
        is_seq = (self.nesting == "seq")
        m_f = compute_second_moment(f, seq_nesting=is_seq)
        m_g = compute_second_moment(g, seq_nesting=is_seq)

        if self.nesting == "jnt":
            metric = (self.mat_mask * m_f * m_g).sum()
        else:
            metric = (m_f * m_g).sum()

        loss = corr + metric

        # 3. Regularization (Optional)
        if self.reg_weight > 0:
            reg = kostic_regularization(m_f, f.mean(0)) + \
                  kostic_regularization(m_g, g.mean(0))
            loss += self.reg_weight * reg

        return loss


# =============================================================================
# Simple MLP Encoder
# =============================================================================

class MLPEncoder(nn.Module):
    """
    Multi-Layer Perceptron Encoder for Koopman feature extraction.

    Args:
        input_dim: Dimension of input state.
        output_dim: Dimension of output features (n_modes).
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name (e.g., 'ReLU', 'Tanh', 'LeakyReLU').
        use_batchnorm: Whether to use batch normalization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "ReLU",
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Resolve activation function
        if hasattr(nn, activation):
            act_fn = getattr(nn, activation)
        else:
            raise ValueError(f"Activation '{activation}' not found in torch.nn")

        # Build network
        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            curr_dim = h_dim

        # Final layer
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_dim) or (batch, ...).

        Returns:
            Features of shape (batch, output_dim).
        """
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)


# =============================================================================
# Koopman Model with LoRA
# =============================================================================

class KoopmanLoRAModel(nn.Module):
    """
    Koopman Operator Model with LoRA Loss.

    This model learns a low-rank approximation of the Koopman operator using
    two encoders (current and future state) and optionally learns singular values.

    Args:
        input_dim: Dimension of system state.
        n_modes: Number of Koopman modes (rank of approximation).
        hidden_dims: Hidden layer dimensions for encoders.
        activation: Activation function.
        use_batchnorm: Whether to use batch normalization.
        shared_encoder: If True, uses same encoder for current and future.
        learn_svals: If True, learns singular values as parameters.
        has_centering: If True, fixes first singular value to 1.0.
    """

    def __init__(
        self,
        input_dim: int,
        n_modes: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "ReLU",
        use_batchnorm: bool = False,
        shared_encoder: bool = False,
        learn_svals: bool = False,
        has_centering: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_modes = n_modes
        self.learn_svals = learn_svals
        self.has_centering = has_centering

        # Create encoders
        self.encoder_f = MLPEncoder(
            input_dim, n_modes, hidden_dims, activation, use_batchnorm
        )
        
        if shared_encoder:
            self.encoder_g = self.encoder_f
        else:
            self.encoder_g = MLPEncoder(
                input_dim, n_modes, hidden_dims, activation, use_batchnorm
            )

        # Learned singular values (optional)
        if learn_svals:
            num_learnable = n_modes - 1 if has_centering else n_modes
            init_vals = torch.sort(
                torch.randn(num_learnable), descending=True
            ).values
            self.svals_params = nn.Parameter(init_vals)
        else:
            self.register_parameter("svals_params", None)

    @property
    def svals(self) -> Optional[torch.Tensor]:
        """Returns the processed singular values."""
        if not self.learn_svals:
            return None
        
        # Apply sigmoid to enforce positivity and boundedness
        vals = torch.sigmoid(self.svals_params)
        
        # Prepend 1.0 if centering is enabled
        if self.has_centering:
            ones = torch.ones(1, device=vals.device, dtype=vals.dtype)
            vals = torch.cat([ones, vals])
        
        return vals

    def forward(self, x: torch.Tensor, lagged: bool = False) -> torch.Tensor:
        """
        Encode input state to feature space.

        Args:
            x: Input state (batch, input_dim).
            lagged: If True, uses lagged encoder (for future state).

        Returns:
            Features (batch, n_modes).
        """
        if lagged:
            return self.encoder_g(x)
        else:
            return self.encoder_f(x)

    def encode_trajectory(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both current and future states.

        Args:
            x: Current states (batch, input_dim).
            y: Future states (batch, input_dim).

        Returns:
            Tuple of (features_x, features_y).
        """
        f = self.forward(x, lagged=False)
        g = self.forward(y, lagged=True)
        return f, g


# =============================================================================
# Training Utilities
# =============================================================================

def train_koopman_lora(
    model: KoopmanLoRAModel,
    loss_fn: NestedLoRALoss,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 100,
    log_interval: int = 10,
) -> List[float]:
    """
    Training loop for Koopman LoRA model.

    Args:
        model: KoopmanLoRAModel instance.
        loss_fn: NestedLoRALoss instance.
        train_loader: DataLoader yielding (x, y) pairs.
        optimizer: Optimizer instance.
        device: Device to train on.
        n_epochs: Number of training epochs.
        log_interval: How often to print loss.

    Returns:
        List of loss values per epoch.
    """
    model.to(device)
    loss_fn.to(device)
    model.train()
    
    loss_history = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward pass
            f, g = model.encode_trajectory(x, y)

            # Compute loss
            loss = loss_fn(f, g, svals=model.svals)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")
            if model.svals is not None:
                print(f"  Singular values: {model.svals.detach().cpu().numpy()}")

    return loss_history


# =============================================================================
# Inference Utilities
# =============================================================================

@torch.no_grad()
def compute_koopman_spectrum(
    model: KoopmanLoRAModel,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Compute the Koopman spectrum (singular values and eigenvalues).

    Args:
        model: Trained KoopmanLoRAModel.
        data_loader: DataLoader for computing statistics.
        device: Device.

    Returns:
        Dictionary with keys 'singular_values', 'eigenvalues', 'K_matrix'.
    """
    model.eval()
    model.to(device)

    # Accumulate second moment matrices
    M_f_sum = 0
    M_g_sum = 0
    T_fg_sum = 0
    n_samples = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        f, g = model.encode_trajectory(x, y)
        
        # Scale by singular values if available
        if model.svals is not None:
            scale = model.svals.sqrt().unsqueeze(0)
            f = f * scale
            g = g * scale

        batch_size = f.shape[0]
        M_f_sum += f.T @ f
        M_g_sum += g.T @ g
        T_fg_sum += f.T @ g
        n_samples += batch_size

    # Normalize
    M_f = M_f_sum / n_samples
    M_g = M_g_sum / n_samples
    T_fg = T_fg_sum / n_samples

    # CCA: Whitening + SVD
    def matrix_inv_sqrt(M, eps=1e-6):
        vals, vecs = torch.linalg.eigh(M)
        vals = torch.clamp(vals, min=eps)
        return vecs @ torch.diag(vals.pow(-0.5)) @ vecs.T

    M_f_inv_half = matrix_inv_sqrt(M_f)
    M_g_inv_half = matrix_inv_sqrt(M_g)
    
    O = M_f_inv_half @ T_fg @ M_g_inv_half
    U, S, Vh = torch.linalg.svd(O, full_matrices=False)

    # Koopman matrix approximation (EDMD style)
    K_matrix = torch.linalg.pinv(M_f) @ T_fg
    eigvals, eigvecs = torch.linalg.eig(K_matrix)

    return {
        'singular_values': S.cpu().numpy(),
        'eigenvalues': eigvals.cpu().numpy(),
        'K_matrix': K_matrix.cpu().numpy(),
        'M_f': M_f.cpu().numpy(),
        'M_g': M_g.cpu().numpy(),
        'T_fg': T_fg.cpu().numpy(),
    }


if __name__ == "__main__":
    print("Standalone LoRA Implementation for Koopman Operator Learning")
    print("=" * 70)
    print("\nThis module provides:")
    print("  - NestedLoRALoss: The core loss function")
    print("  - MLPEncoder: Simple feature extractor")
    print("  - KoopmanLoRAModel: Complete model with training support")
    print("  - Training and inference utilities")
    print("\nSee the accompanying notebook for usage examples.")
