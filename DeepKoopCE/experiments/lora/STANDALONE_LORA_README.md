# Standalone LoRA Implementation for Koopman Operator Learning

This directory contains a **self-contained implementation** of the LoRA (Low-Rank Approximation) algorithm for learning Koopman operators, extracted from the [NeuralKoopmanSVD](https://github.com/MinchanJeong/NeuralKoopmanSVD) project.

## 📁 Files

- **`standalone_lora.py`**: Complete standalone implementation (no external dependencies on the original project)
- **`demo_standalone_lora.ipynb`**: Interactive Jupyter notebook with examples and visualizations

## 🎯 What is LoRA?

LoRA (Low-Rank Approximation) is an efficient method for learning Koopman operators that:

- **Learns feature functions** φ(x) such that K·φ(x_t) ≈ φ(x_{t+1})
- **Avoids matrix inversions** and SVD during training
- **Supports nested optimization** for ordered singular functions
- **Enables parametric learning** of singular values

## 🚀 Quick Start

### Installation

```bash
# Required packages
pip install numpy torch matplotlib

# Optional: for notebook
pip install jupyter
```

### Basic Usage

```python
from standalone_lora import (
    NestedLoRALoss,
    KoopmanLoRAModel,
    train_koopman_lora,
    compute_koopman_spectrum
)

# 1. Create model
model = KoopmanLoRAModel(
    input_dim=2,           # System state dimension
    n_modes=8,             # Number of Koopman modes
    hidden_dims=[128, 128],
    learn_svals=True,      # Learn singular values
)

# 2. Create loss function
loss_fn = NestedLoRALoss(
    n_modes=8,
    nesting='jnt',         # Joint nesting strategy
    reg_weight=0.0,
)

# 3. Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history = train_koopman_lora(
    model, loss_fn, train_loader, optimizer, device, n_epochs=100
)

# 4. Analyze spectrum
spectrum = compute_koopman_spectrum(model, test_loader, device)
print("Singular values:", spectrum['singular_values'])
print("Eigenvalues:", spectrum['eigenvalues'])
```

## 📊 Demo Notebook

The `demo_standalone_lora.ipynb` notebook provides complete examples:

1. **Data Generation**: Logistic map and 2D nonlinear systems
2. **Model Training**: Step-by-step training process
3. **Visualization**: Loss curves, singular values, eigenvalues
4. **Spectrum Analysis**: Feature orthogonality and mode importance
5. **Feature Functions**: Learned Koopman basis functions (for 1D systems)

To run:

```bash
jupyter notebook demo_standalone_lora.ipynb
```

## 🔧 Core Components

### 1. NestedLoRALoss

The LoRA loss function minimizes:

```
L(f, g) = -2·Tr(T[f, g]) + Tr(M_f · M_g)
```

Where:
- `T[f, g] = E[f·g^T]`: Cross-correlation
- `M_f = E[f·f^T]`: Auto-correlation of current features
- `M_g = E[g·g^T]`: Auto-correlation of future features

**Nesting Strategies:**
- `'jnt'` (Joint): Progressive weighting for all modes (recommended)
- `'seq'` (Sequential): Stop-gradient based sequential optimization
- `None`: Standard LoRA without ordering

### 2. KoopmanLoRAModel

Dual-encoder architecture:
- **encoder_f**: Encodes current state x_t
- **encoder_g**: Encodes future state x_{t+1}
- **Optional**: Learned singular values

### 3. Training Utilities

- `train_koopman_lora()`: Complete training loop
- `compute_koopman_spectrum()`: Post-training spectral analysis

## 📈 Example Systems

### 1. Logistic Map (Chaotic)

```python
x_{t+1} = 4·x_t·(1 - x_t) + noise
```

### 2. 2D Nonlinear System

```python
x1' = λ·x1
x2' = μ·x2 + (λ² - μ)·x1²
```

Both systems are included in the demo notebook.

## 🎨 Visualization Examples

The demo generates:
- Training loss curves
- Singular value spectra
- Eigenvalue distribution (complex plane)
- Feature orthogonality heatmaps
- Learned feature functions (for 1D systems)

## ⚙️ Hyperparameters

### Model Configuration

```python
KoopmanLoRAModel(
    input_dim=1,              # State dimension
    n_modes=8,                # Number of modes (rank)
    hidden_dims=[128, 128],   # MLP hidden layers
    activation='LeakyReLU',   # Activation function
    use_batchnorm=False,      # Batch normalization
    shared_encoder=False,     # Share f and g encoders
    learn_svals=True,         # Parametric singular values
    has_centering=False,      # Fix first σ to 1.0
)
```

### Loss Configuration

```python
NestedLoRALoss(
    n_modes=8,                # Must match model
    nesting='jnt',            # 'jnt', 'seq', or None
    reg_weight=0.0,           # Kostic regularization
)
```

### Recommended Settings

| System Complexity | n_modes | hidden_dims | nesting |
|-------------------|---------|-------------|---------|
| Simple (1D)       | 5-8     | [64, 64]    | 'jnt'   |
| Medium (2-3D)     | 8-12    | [128, 128]  | 'jnt'   |
| Complex (>3D)     | 12-20   | [256, 256]  | 'seq'   |

## 📚 Key References

1. **Jeong et al., "Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems"**, NeurIPS 2025
   - Introduces the LoRA method and nesting strategies

2. **Wu & Noe, "Variational Approach for Learning Markov Processes"**, NeurIPS 2019
   - VAMP method (related to LoRA objective)

3. **Kostic et al., "Learning Dynamical Systems via Koopman Operator Regression in Reproducing Kernel Hilbert Spaces"**, NeurIPS 2022
   - Metric regularization techniques

## 🔬 Theory Background

### Koopman Operator

For a dynamical system `x' = F(x)`, the Koopman operator K acts on observables:

```
K[φ](x) = φ(F(x))
```

LoRA learns finite-dimensional approximation:

```
K ≈ Σᵢ σᵢ·ψᵢ ⊗ φᵢ
```

Where:
- `φᵢ`: Right singular functions (current state)
- `ψᵢ`: Left singular functions (future state)
- `σᵢ`: Singular values (mode importance)

### Why Dual Encoders?

The Koopman operator's SVD requires learning **pairs** of functions (φ, ψ) that satisfy:

```
E[φᵢ(x)·ψⱼ(x')] = σᵢ·δᵢⱼ
```

Hence:
- `encoder_f` learns φ (acts on current state)
- `encoder_g` learns ψ (acts on future state)

## 🐛 Troubleshooting

### Issue: Loss not decreasing

- **Solution**: Try smaller learning rate (1e-4) or larger batch size
- Check data normalization
- Verify nesting='jnt' is enabled

### Issue: Singular values collapse

- **Solution**: Enable regularization (`reg_weight=0.1`)
- Use nesting strategy ('jnt' or 'seq')
- Increase network capacity

### Issue: Poor feature orthogonality

- **Solution**: Increase training epochs
- Use joint nesting ('jnt')
- Add Kostic regularization

## 💡 Tips for Best Results

1. **Data Requirements**
   - Need sufficient trajectories to cover state space
   - Longer trajectories capture better dynamics
   - Balance training set size vs. diversity

2. **Model Selection**
   - Start with `n_modes=8` and adjust based on singular value spectrum
   - Use `learn_svals=True` for better convergence
   - Try `'jnt'` nesting first (most stable)

3. **Training**
   - Monitor both loss and VAMP-2 score
   - Check that singular values are decreasing
   - Verify eigenvalues stay within unit circle (stable systems)

## 📄 License

This implementation is extracted from [NeuralKoopmanSVD](https://github.com/MinchanJeong/NeuralKoopmanSVD).
Please refer to the original repository for licensing information.

## 🙏 Acknowledgments

- Original implementation by Minchan Jeong et al.
- Based on NeurIPS 2025 paper on efficient Koopman operator learning
- Extracted and documented for standalone use

## 📧 Questions?

For questions about:
- **Original method**: See [NeuralKoopmanSVD](https://github.com/MinchanJeong/NeuralKoopmanSVD)
- **This standalone version**: Open an issue or refer to the demo notebook

---

**Happy Koopman Learning! 🎉**
