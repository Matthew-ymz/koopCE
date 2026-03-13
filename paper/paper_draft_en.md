# KoopCE: Learning Spectrally Structured Koopman Embeddings for Coarse-Grained Dynamics

## Abstract

We reconstruct the main research story of the current repository from code, checkpoints, notebooks, and saved figures. The strongest jointly supported contribution is a deep autoencoder-Koopman pipeline that learns a nonlinear lifting, propagates latent states with a linear operator, and extracts macro variables through singular-vector coarse-graining. The training objective combines reconstruction, multi-step prediction, latent linearity, robustness, sparsity, and an optional singular-spectrum regularizer that the repository refers to as CE. The current evidence is strongest on two settings: a synthetic nonlinear map and air-quality time series from 127 monitoring stations. On seed-matched PM2.5-only checkpoints, mild CE variants increase a 50-mode spectrum-entropy summary from `4.9270` to `5.3994` or `5.3729`, while 5-step rollout loss changes only modestly from `185.90` to `195.74` or `200.01`. However, more aggressive CE-tagged checkpoints become unstable, with forecast losses as large as `1.73e4` to `4.30e8`. On the toy system, the best saved checkpoint reaches a 30-step rollout loss of `1.081e-3`, and symbolic regression on learned latent coordinates recovers simple quadratic observables consistent with the designed nonlinear map. Additional sparse-identification notebooks on Kuramoto and lifted oscillator systems reinforce the interpretation claim by showing that low-dimensional macro laws can become simple and nearly exact once the observables are well chosen. Overall, the repository provides credible evidence that spectral shaping is useful for discovering coarse-grained coordinates, but the benefit is highly sensitive to checkpoint selection and remains less mature than the interpretability pipeline itself.

## Introduction

### Problem Context

Many nonlinear dynamical systems admit useful low-dimensional descriptions, but discovering those descriptions from raw trajectories is difficult. A practical pipeline must do more than forecast accurately: it should also expose macro variables, reveal a stable linear or nearly linear latent evolution, and support downstream equation discovery. The current repository explores exactly this intersection. It contains a deep learned lifting model, classical sparse-identification utilities, residual Koopman diagnostics, a standalone LoRA implementation for Koopman spectra, and exploratory stage-wise neural time-series analysis.

The repository is heterogeneous, so a narrow paper framing is necessary. The most defensible framing is the deep autoencoder-Koopman branch with post hoc spectral coarse-graining. This branch has all of the ingredients needed for a coherent paper story: a dedicated model implementation, saved checkpoints, synthetic and air-quality experiments, macro-variable visualizations, and follow-up symbolic regression analyses. Other branches are valuable, but they function more naturally as supporting evidence or future extensions than as co-equal primary contributions.

### Contribution Summary

This report supports four calibrated claims.

1. The repository implements a clear end-to-end pipeline for learning nonlinear Koopman embeddings, linear latent evolution, and low-dimensional macro variables.
2. The learned Koopman matrix is explicitly analyzed through its singular spectrum, and the leading left singular vectors are used as coarse-graining coefficients for macro-state extraction.
3. Moderate CE-style spectral shaping can increase the spread of singular-spectrum contributions on PM2.5 data without catastrophically damaging rollout quality, but stronger spectral shaping can destabilize the model.
4. The repository's classical sparse-identification notebooks provide strong downstream evidence that well-chosen observables can collapse complicated systems into simple low-dimensional equations.

## Method

### Problem Formulation

Let `x_t in R^d` denote the observed micro-state at time step `t`. The goal is to learn a representation in which the dynamics are easier to evolve linearly and easier to summarize macroscopically. The repository pursues this through a learned lifting `g_theta`, a linear latent operator `K`, a decoder `h_theta`, and a post hoc singular-vector projection that defines macro variables.

### Notation

For a window `x_{t:t+S} = (x_t, x_{t+1}, ..., x_{t+S})`, define

`z_t = g_theta(x_t),   z_hat_{t+tau} = K^tau z_t,   x_hat_t = h_theta(z_t)`.

The latent dimension is `k`, the macro dimension is `r <= k`, and the learned linear operator is a matrix `K in R^{k x k}`.

### Model Overview

The observed deep branch uses a multilayer perceptron encoder and decoder around a linear latent evolution module. The encoder maps each micro-state to a latent state, the latent state is repeatedly advanced by the same linear operator, and the decoder maps latent predictions back to the original observation space. In contrast to purely predictive latent models, the repository treats the latent linear operator as an object of scientific interest: its singular values and singular vectors are used to score spectral concentration and to construct coarse-grained macro variables.

### Module Decomposition

The encoder and decoder are standard feed-forward networks. The unusual component is the latent evolution module, which performs repeated one-step multiplication by the same trainable linear map. This enforces a Koopman-style inductive bias: the nonlinear observation space is first lifted into a latent coordinate system, and the latent sequence is then modeled with a linear recurrence.

After training, the singular value decomposition

`K = U Sigma V^T`

is used twice. First, the singular values summarize how strongly different latent directions contribute to the learned dynamics. Second, the leading columns of `U` define a coarse-graining matrix

`C = U_{:, 1:r}`.

Given latent states `z_t`, the macro variables are

`y_t = z_t C`.

This is the central bridge between learned prediction and interpretable macro dynamics in the repository.

### Objective And Training

The current loss implementation can be written as

`L = alpha_1 (L_pred + L_rec) + L_lin + alpha_2 L_inf + alpha_3 ||K||_1 + alpha_CE L_CE`.

Here `L_pred` is decoder-space prediction loss on future steps, `L_rec` is reconstruction loss, `L_lin` enforces consistency between encoded future states and linearly evolved latent states, `L_inf` adds an infinity-norm robustness term, and `||K||_1` regularizes the latent operator. The observed CE objectives are singular-value based:

`L_CE^(1) = - sum_i sigma_i(K),`

`L_CE^(2) = - sum_i sigma_i(K) log sigma_i(K).`

Only these two forms are implemented in the current code. A stored checkpoint labeled as a third type exists, but its exact training objective is not recoverable from the present implementation and is therefore treated only as an empirical artifact, not as a method definition.

### Inference And Macro Analysis

Prediction proceeds by encoding the current state, applying the learned linear operator repeatedly in latent space, and decoding the resulting latent sequence. Macro analysis is performed after training by selecting a truncation rank `r`, projecting latent states onto `U_{:,1:r}`, and then either visualizing the macro trajectories or fitting symbolic equations to them.

The repository also defines a singular-spectrum summary used throughout the notebooks. For singular values `sigma_1 >= sigma_2 >= ...`, it computes

`a_i = (1/i) sum_{j=1}^i log sigma_j,   Delta_i = a_i - a_{i+1},`

followed by a Shannon-entropy score

`EC = - sum_i p_i log_2 p_i,   p_i = Delta_i / sum_m Delta_m.`

This quantity is best interpreted as an entropy-style measure of how distributed the positive spectrum contributions are across scales.

## Experiments

### Datasets And Splits

The repository provides three main evidence sources for this report.

The first is a synthetic two-dimensional nonlinear map used to test whether the learned latent coordinates expose simple lifted observables. The saved notebook configuration uses `lambda = 0.1`, `mu = 0.9`, 100 training trajectories of length 100, and 20 test trajectories with 30-step rollout evaluation.

The second is air-quality data for the Yangtze River Delta. The shipped dataset contains 70,128 time steps and 127 stations, with at least PM2.5 and O3 variables available. The deep branch uses either PM2.5 alone as a 127-dimensional input or concatenated PM2.5 and O3 as a 254-dimensional input. The notebook split uses the first 60,000 time steps for training and the remainder for testing.

The third is a collection of classical sparse-identification notebooks, especially a lifted nonlinear oscillator, a two-group Kuramoto model, and a 36-combination Kuramoto sweep. These are not matched baselines for the deep branch, but they provide direct evidence about coarse-graining and interpretability.

### Compared Methods

The repository does not contain a clean external baseline table. Accordingly, the main comparisons in this report are internal:

- no-CE versus CE-tagged stored checkpoints on PM2.5-only air-quality data
- two joint PM2.5+O3 checkpoints with different apparent training maturity
- several toy checkpoints with different regularization or architecture tags
- classical sparse-identification pipelines that act as downstream analysis tools rather than direct competitors to the learned model

This is a limitation, but it is still sufficient to support calibrated conclusions about spectral shaping, macro extraction, and failure modes.

### Metrics

For checkpoint reevaluation, we use the repository loss functions and held-out rollout probes. On air-quality checkpoints, `val_loss10` averages the first 10 validation batches and `forecast_loss_2048` evaluates 5-step rollouts on the first 2048 held-out windows. On the toy system, `forecast_loss` is the 30-step rollout probe used in the notebook logic. We also report EC-style singular-spectrum summaries and maximum singular values. For the classical notebooks, we use the printed sparse-model score, MSE, and macro-equation outputs already embedded in the saved notebook results.

### Implementation Details

The PM2.5-only deep checkpoints use a `127 -> 512 -> 256` encoder profile and a `256 x 256` latent operator. The joint PM2.5+O3 checkpoints use a `254 -> 512 -> 512` profile with a `512 x 512` latent operator. The toy checkpoints use a three-dimensional latent space with hidden width 8 or 4. The deep air notebook uses a horizon `Sp = 30`, batch size 256, and a 5-step forecast probe. The toy notebook uses `Sp = 30`, 10 training epochs, and a 30-step forecast probe.

Because the repository lacks a unified run ledger, the reevaluated checkpoint tables should be read as standardized probes of stored artifacts rather than as a fully controlled hyperparameter sweep.

## Results

### PM2.5-Only Air-Quality Checkpoints

#### Experiment Purpose

The main question is whether CE-style spectral shaping can improve the structure of the learned latent spectrum without breaking the predictive behavior of the deep Koopman model.

#### Setup Summary

We reevaluated the stored PM2.5-only checkpoints on the repository air dataset using the same model class and predictive losses. The cleanest comparison is between the seed-tagged no-CE checkpoint and the seed-tagged type-1 CE checkpoint, with the type-2 checkpoint serving as an additional stable reference and the remaining CE-tagged runs serving as negative examples.

#### Main Findings

| Variant | EC50 | Max SV | Probe Validation Loss | 5-Step Forecast Loss |
|---|---:|---:|---:|---:|
| PM25-noCE-seed | 4.9270 | 1.4706 | 796.55 | 185.90 |
| PM25-CE1-seed | 5.3994 | 1.3106 | 819.86 | 195.74 |
| PM25-CE2 | 5.3729 | 1.3046 | 821.44 | 200.01 |
| PM25-CE1-aggressive | 4.7485 | 2.5176 | 1.84e11 | 4.30e8 |
| PM25-CE3-aggressive | 4.8996 | 2.0498 | 5.89e4 | 1.73e4 |

The stable CE variants increase EC by roughly `0.45` relative to the seed-matched no-CE checkpoint while only slightly increasing rollout loss. In contrast, aggressive CE-tagged checkpoints produce much larger top singular values and dramatically worse forecast behavior.

#### Interpretation

These results support a nuanced interpretation. Mild spectral shaping appears able to redistribute singular-spectrum contributions in a way that is visible to the EC summary while preserving broadly similar rollout quality. However, once the leading singular values become too dominant, the latent linear operator appears to amplify rollout errors rather than organize the dynamics more cleanly.

#### Limitation Or Caveat

The repository does not provide a clean training log for every stored checkpoint. Therefore, this section compares stored artifacts rather than a perfectly controlled seed-and-hyperparameter grid. In particular, the exact provenance of the aggressive CE-tagged checkpoints is incomplete, and the type-3 objective is not implemented in the current code.

#### Takeaway

The strongest supported empirical claim in the repository is not that CE uniformly improves prediction, but that conservative spectral shaping can improve macro-spectrum structure at modest predictive cost, while aggressive spectral shaping is unstable.

### Toy Nonlinear Map And Symbolic Latents

#### Experiment Purpose

The toy experiment tests whether the learned latent representation becomes scientifically interpretable, not only whether it predicts accurately.

#### Setup Summary

The repository generates a two-dimensional nonlinear map with a quadratic coupling term, trains a three-dimensional latent model on 31-step windows, and then fits symbolic regressors to the learned macro coordinates.

#### Main Findings

| Variant | EC | Max SV | Probe Validation Loss | 30-Step Forecast Loss |
|---|---:|---:|---:|---:|
| Toy-noCE | 0.9426 | 0.0399 | 0.011132 | 0.001823 |
| Toy-L1 | 0.9980 | 1.0178 | 0.008179 | 0.001081 |
| Toy-strong-spectrum | 0.7970 | 1.8281 | 0.011544 | 0.001869 |
| Toy-moderate-spectrum | 0.9069 | 0.3801 | 0.010377 | 0.001852 |

The best saved rollout is obtained by the regularized toy checkpoint labeled here as `Toy-L1`. More importantly, the notebook's symbolic regression output shows that the learned latent coordinates can be approximated by simple quadratic expressions of the first micro-state, closely matching the designed nonlinear lifting.

#### Interpretation

The toy experiment supports the repository's interpretability story more strongly than its optimization story. All stored toy checkpoints predict reasonably well, but the deeper scientific signal is that the learned latent coordinates collapse to a simple quadratic basis that is consistent with the exact synthetic map.

#### Limitation Or Caveat

The saved toy checkpoints confound multiple changes, including different hidden widths and regularization tags. As a result, they do not isolate the effect of CE cleanly. The symbolic-regression outputs are therefore more convincing than the small differences in rollout loss.

#### Takeaway

The toy branch provides direct evidence that the learned representation can expose interpretable observables, which is central to the repository's broader coarse-graining objective.

### Multi-Variable Air Quality And Macro Extraction

#### Experiment Purpose

This experiment asks whether the same deep Koopman machinery can extend from a PM2.5-only view to a joint PM2.5+O3 view while still supporting meaningful macro variables.

#### Setup Summary

The joint checkpoints use 254-dimensional inputs obtained by concatenating PM2.5 and O3 observations across 127 stations. We reevaluated two stored no-CE checkpoints and combined this with the saved macro-overlay figures and notebook logic for macro-series extraction.

#### Main Findings

The stronger joint checkpoint reduces 5-step forecast loss from `2116.91` to `317.14` relative to the older stored checkpoint and also raises EC50 from `3.5801` to `4.3823`. The saved macro figures show one- and two-dimensional macro series tracking broad PM2.5 station-level tendencies rather than high-frequency station-specific fluctuations.

#### Interpretation

This branch suggests that the repository's macro-extraction mechanism scales beyond a single pollutant and can capture shared slow structure across coupled pollutants and stations. The large performance gap between the two stored checkpoints also suggests that horizon alignment and training maturity matter substantially in this higher-dimensional setting.

#### Limitation Or Caveat

Unlike the PM2.5-only branch, the joint branch currently lacks matched CE and no-CE checkpoints, so it cannot directly answer whether CE helps in the multi-variable setting.

#### Takeaway

Joint air-quality modeling is feasible and qualitatively interpretable in the current repository, but the PM2.5-only branch remains the cleaner source of quantitative evidence.

### Classical Sparse Macro-Dynamics

#### Experiment Purpose

The purpose of the classical notebooks is to test whether low-dimensional macro laws become simple once the observable set is sufficiently aligned with the underlying dynamics.

#### Setup Summary

We use the saved outputs from a lifted nonlinear oscillator, a two-group Kuramoto model, and a Kuramoto grid sweep. These notebooks rely on hand-crafted libraries and sparse regression rather than end-to-end learned lifting, but they are closely aligned with the repository's macro-dynamics objective.

#### Main Findings

The lifted nonlinear oscillator notebook reports `mse = 3.281e-32` and recovers the target equations almost exactly. The two-group Kuramoto notebook reports `score = 0.9501695514710947`, `EC = 4.771670063172314`, and a two-dimensional macro dynamics in which the macro variables evolve independently as simple linear recurrences. The larger Kuramoto sweep explores 36 parameter combinations and reports top EC values around `5.69`, together with heatmaps and spectrum comparisons across coupling and noise settings.

#### Interpretation

These notebooks show that once the observables are well chosen, coarse-grained macro dynamics can be both low-dimensional and interpretable. This is precisely the downstream use case that motivates the deep branch: the learned Koopman representation is valuable not only as a predictor, but as a generator of analysis-ready macro variables.

#### Limitation Or Caveat

These classical notebooks are not direct baselines for the deep model because they rely on hand-crafted libraries and different data-generation protocols. They validate the analysis pipeline more directly than the end-to-end learning claim.

#### Takeaway

The strongest classical evidence in the repository is that sparse identification becomes exceptionally effective once the system is represented in a suitable macro coordinate system.

### Figure-Centered Analysis

The saved air-quality macro overlays are important because they visually connect the learned macro variables to the common tendency shared across multiple PM2.5 station trajectories. This matters more than raw loss values alone: it suggests that the macro coordinate is capturing a system-wide mode rather than simply memorizing local fluctuations.

The saved classical sparse-identification figure serves a different purpose. It shows that the repository also maintains a conventional pipeline for identifying dynamical laws from trajectories, complete with trajectory comparison and error visualization. This strengthens the overall story by showing that the repository is not limited to black-box latent prediction.

The Kuramoto sweep figures provide the clearest qualitative sensitivity study in the repository. They compare spectrum shape, EC, coarse-graining coefficients, and frequency summaries across many coupling/noise settings. Even without a single consolidated metric table, these visuals support the claim that macro-spectrum structure changes systematically with interaction strength and noise.

## Conclusion

The current repository most credibly supports a paper about learning spectrally structured Koopman embeddings and then extracting coarse-grained macro dynamics from the learned latent operator. The method is clear in code, the macro-extraction rule is explicit, and the downstream symbolic-analysis pipeline is well developed.

The empirical evidence is promising but uneven. The PM2.5-only branch supports a calibrated positive claim: moderate CE-style spectral shaping can increase singular-spectrum diversity at limited predictive cost. The same branch also supplies the strongest negative result: aggressive spectral shaping can destabilize rollout severely. The toy and classical notebooks strengthen the interpretability story by showing that coarse-grained coordinates can recover simple or nearly exact equations. What is still missing is a unified, seed-averaged experiment ledger and a cleaner provenance record for every stored checkpoint.

In its current state, the repository is best presented as a strong research prototype for Koopman-based coarse-graining rather than a fully finalized benchmark paper. That is still a meaningful contribution: it already contains enough code and evidence to motivate a serious manuscript, provided that the claims remain tightly aligned with the artifacts that actually exist.
