# Tables

## Structured Result Sources

| Source | Type | What it supports |
|---|---|---|
| `DeepKoopCE/deepkoop_func.py` | code | Primary autoencoder-Koopman pipeline, loss terms, and coarse-graining rule |
| `DeepKoopCE/model_save/*.pth` | checkpoints | Re-evaluated spectral statistics and rollout losses |
| `DeepKoopCE/experiments/toy/toy_example.ipynb` | notebook | Toy-system generator, training setup, and symbolic-regression outputs |
| `DeepKoopCE/experiments/air_quality/air_data.ipynb` | notebook | Air-quality dimensions, sequence horizon, and macro-series visualization logic |
| `experiments/kuramoto/kuramoto_2group.ipynb` | notebook | Two-group Kuramoto score, EC, coarse equations, and macro dynamics |
| `experiments/kuramoto/kuramoto_grid_sweep_analysis.ipynb` | notebook | 36-combination Kuramoto sweep and qualitative EC trends |
| `experiments/toy_nonlinear/example_nonlinear_yb.ipynb` | notebook | Exact lifted sparse-identification recovery on a nonlinear oscillator |
| `experiments/air_quality/classical_sparse/air_data.ipynb` | notebook | Classical sparse coarse-graining baseline on air-quality data |
| `experiments/resdmd/ResDMD.ipynb` | notebook | Residual-based Koopman spectral diagnostics |
| `neural_science/experiments/compare_stages.ipynb` | notebook | Exploratory stage-wise neural time-series identification |
| `neural_science/experiments/exp_stage2.ipynb` | notebook | Exploratory continuous/discrete sparse stage modeling |
| `results/pm25_macro.png`, `results/pm25_macro2.png` | figure | Qualitative macro-variable overlays for air-quality data |
| `results/sindy_results.png` | figure | Qualitative baseline visualization for the standalone SINDy pipeline |

## Checkpoint Architecture Inventory

| Checkpoint | Input Dim | Hidden Dim | Koopman Dim | Notes |
|---|---:|---:|---:|---|
| `air127_ce0.pth` | 127 | 512 | 256 | PM2.5-only checkpoint |
| `air127_ce005_type1.pth` | 127 | 512 | 256 | PM2.5-only CE variant |
| `air127_ce005_type1_seed42.pth` | 127 | 512 | 256 | PM2.5-only CE variant with explicit seed tag |
| `air127_ce005_type2.pth` | 127 | 512 | 256 | PM2.5-only alternative CE objective |
| `air127_ce005_type3.pth` | 127 | 512 | 256 | PM2.5-only checkpoint; exact objective unresolved in current code |
| `air127_ce0_seed42.pth` | 127 | 512 | 256 | PM2.5-only no-CE checkpoint with explicit seed tag |
| `air254_ce0_seed42.pth` | 254 | 512 | 512 | Joint PM2.5+O3 checkpoint |
| `air254_ce0_seed42_sp30.pth` | 254 | 512 | 512 | Joint PM2.5+O3 checkpoint with `sp30` tag |
| `toy_ce0_seed42_sp30.pth` | 2 | 8 | 3 | Toy checkpoint |
| `toy_ce0_seed42_sp30_l1.pth` | 2 | 8 | 3 | Toy checkpoint with extra regularization tag |
| `toy_type1_k3_h4_sp30_ace0.01_a1_2_a3_0.001_lr0.001_seed42.pth` | 2 | 4 | 3 | Toy checkpoint with stronger spectral shaping tag |
| `toy_type1_k3_h8_sp30_ace0_a1_2_a3_0.0001_lr0.001_seed42.pth` | 2 | 8 | 3 | Toy checkpoint whose filename still carries a `type1` tag |

## Re-evaluated Air Checkpoints

Evaluation protocol:
- loaded saved checkpoints with the current `Lusch` implementation
- used the repository air-quality dataset (`dataset_yrd.nc`)
- PM2.5-only inputs used 127 stations; joint inputs concatenated PM2.5 and O3 into 254 dimensions
- train/test split used the first `60000` time steps for training and the remainder for testing, matching the notebook logic
- `val_loss10` averages the first 10 validation batches under the current predictive loss with `alpha_CE=0`
- `forecast_loss_2048` evaluates the repository rollout probe on the first 2048 held-out windows
- `EC50` uses the first 50 singular values when the latent dimension exceeds 50

| Checkpoint | EC50 | Max Singular Value | `val_loss10` | `forecast_loss_2048` | Interpretation |
|---|---:|---:|---:|---:|---|
| `air127_ce0.pth` | 5.2391 | 1.3222 | 812.10 | 192.16 | Stable PM2.5-only baseline |
| `air127_ce0_seed42.pth` | 4.9270 | 1.4706 | 796.55 | 185.90 | Strongest stable PM2.5-only no-CE checkpoint |
| `air127_ce005_type1.pth` | 4.7485 | 2.5176 | 1.84e11 | 4.30e8 | Unstable CE run with exploding rollout loss |
| `air127_ce005_type1_seed42.pth` | 5.3994 | 1.3106 | 819.86 | 195.74 | Stable CE run with higher EC than the seed-matched no-CE model |
| `air127_ce005_type2.pth` | 5.3729 | 1.3046 | 821.44 | 200.01 | Stable alternative CE objective with similar behavior to type1 |
| `air127_ce005_type3.pth` | 4.8996 | 2.0498 | 5.89e4 | 1.73e4 | Less catastrophic than the unstable type1 run, but still far worse than stable models |
| `air254_ce0_seed42.pth` | 3.5801 | 1.8775 | 16637.12 | 2116.91 | Older joint PM2.5+O3 checkpoint with weak rollout quality |
| `air254_ce0_seed42_sp30.pth` | 4.3823 | 1.8298 | 1517.99 | 317.14 | Stronger joint PM2.5+O3 checkpoint; likely better aligned with the notebook horizon |

## Re-evaluated Toy Checkpoints

Evaluation protocol:
- regenerated the repository toy system with `lambda=0.1`, `mu=0.9`
- used 100 training trajectories of length 100 and 20 test trajectories of length 80
- formed 31-step windows (`Sp=30`) as in the notebook
- `val_loss10` averages the first 10 validation batches
- `forecast_loss` is the repository 30-step rollout probe

| Checkpoint | EC | Max Singular Value | `val_loss10` | `forecast_loss` | Interpretation |
|---|---:|---:|---:|---:|---|
| `toy_ce0_seed42_sp30.pth` | 0.9426 | 0.0399 | 0.011132 | 0.001823 | Stable toy baseline with a highly contracted spectrum |
| `toy_ce0_seed42_sp30_l1.pth` | 0.9980 | 1.0178 | 0.008179 | 0.001081 | Best saved rollout among the available toy checkpoints |
| `toy_type1_k3_h4_sp30_ace0.01_a1_2_a3_0.001_lr0.001_seed42.pth` | 0.7970 | 1.8281 | 0.011544 | 0.001869 | Stronger spectral shaping with weaker predictive quality |
| `toy_type1_k3_h8_sp30_ace0_a1_2_a3_0.0001_lr0.001_seed42.pth` | 0.9069 | 0.3801 | 0.010377 | 0.001852 | Moderate spectral reshaping with similar rollout quality to baseline |

## Notebook-Derived Quantitative Artifacts

| Artifact | Observed Numbers | Why it matters |
|---|---|---|
| Lifted nonlinear oscillator sparse identification | `mse = 3.281e-32`, `nnz = 9` | Demonstrates near-exact recovery when the observable library is well aligned with the dynamics |
| Two-group Kuramoto sparse model | `score = 0.9501695514710947`, `EC = 4.771670063172314`, `delta_gamma(rank=2) = 4.489796119502212` | Shows that coarse observables can collapse the system to low-dimensional linear recurrences |
| Kuramoto sweep | `36` parameter combinations scanned; top observed EC values around `5.69` in the printed summary | Supports the claim that EC varies systematically with coupling/noise settings |
| Classical air sparse model | `score = 0.9335436039342788`, `EC = 3.942484161651143`, `delta_gamma(rank=1) = 3.6916663157826157` | Provides a hand-crafted baseline for air-quality coarse-graining |
| Neural stage-wise sparse models | scores around `0.9914`, `0.9897`, `0.9966` in one stage notebook; `0.6401` for another stage-2 configuration | Shows an exploratory application branch with mixed stability |
