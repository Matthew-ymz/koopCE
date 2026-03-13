# Open Questions

- Recover the exact training cells or historical code that produced the `air127_ce005_type3.pth` checkpoint. The current repository implements only `type1` and `type2` CE losses.
- Reconstruct a cleaner provenance map for the `air127_*` checkpoints. Their architectures are recoverable, but a unified config/log file for training hyperparameters is missing.
- Decide whether the main paper should remain focused on the deep autoencoder-Koopman branch, or explicitly broaden to a repository-level "Koopman coarse-graining toolkit" story. The current draft chooses the narrower framing.
- Build a seed-averaged comparison table for PM2.5-only and PM2.5+O3 models. Current evidence is based on stored checkpoints rather than a full experiment ledger.
- Pair the saved macro figures with numeric summaries such as correlation, variance explained, or station-group consistency. The current figure evidence is mostly qualitative.
- Clarify whether the standalone LoRA module is meant as a future baseline, a separate paper direction, or an implementation note. The repository currently lacks a matched comparison against the main deep Koopman branch.
- Decide how much of the stage-wise neural time-series branch should appear in the final manuscript. It is empirically interesting, but not yet integrated into the same metric space.
- Clean up root-level metadata and narrative files that still point to upstream or partially unrelated projects. This would reduce ambiguity for future paper writing.
