# Paper

**TARVI: Transcription-Factor Aided RNA Velocity Inference with Supervised
Latent Time and Velocity-Pseudotime Blending**

Chandula Adhikari, Sandeep Dassanayake, Damayanthi Herath
Department of Computer Engineering, University of Peradeniya, Sri Lanka

📄 [`TARVI__Transcription_Factor_Aided_RNA_Velocity_Inference.pdf`](TARVI__Transcription_Factor_Aided_RNA_Velocity_Inference.pdf)
(6 pp.)

> 🏆 **Best Paper Award** — *Biomedical Engineering and Instrumentation* track,
> MERCon 2026 (Moratuwa Engineering Research Conference, IEEE), University of
> Moratuwa, Sri Lanka, 13–14 August 2026.
> ([certificate](../docs/assets/mercon_certificate.jpg))

## Abstract

Reconstructing cell-fate trajectories from single-cell RNA sequencing
(scRNA-seq) is fundamental to understanding cellular development and disease.
RNA velocity infers cellular dynamics from unspliced-to-spliced mRNA ratios, but
existing methods share three key limitations: transcription rates are static
gene-level constants that ignore cell-state regulation; latent cell time is
weakly supervised and poorly calibrated; and locally inferred velocity vectors
are often inconsistent with global trajectories, producing reversed or
discontinuous flows. We present TARVI (Transcription-Factor-Aided RNA Velocity
Inference), a deep probabilistic framework that extends the VeloVI variational
autoencoder to address all three limitations, through cell-specific
transcription rates driven by transcription factor–target regulatory priors
(ChEA/ENCODE), a supervised residual latent-time head, novel velocity–time
consistency and temporal-smoothness losses, and a velocity–pseudotime blending
step. Evaluated on 5 datasets against 4 baselines with 8 metrics, TARVI matches
the strongest baseline on cross-boundary direction accuracy (CBDir: 0.933 vs.
VeloVI's 0.931) while improving Spearman pseudotime correlation by 126% over
VeloVI (0.747 vs. 0.331), recovering coherent directional flows and smooth
developmental gradients where competing methods fail.

**Index terms:** RNA velocity, variational autoencoder, transcription factor,
latent time, deep generative model

## Contributions

- A masked TF→gene weight matrix (ChEA/ENCODE priors) producing a cell-specific
  transcription rate `α_i`, jointly optimised end-to-end against the ODE
  likelihood.
- A residual latent-time head supervised by a stop-gradient ODE-time signal,
  with a graph-Laplacian temporal-smoothness regulariser, plus a velocity–time
  consistency loss that directly penalises the angle between each cell's velocity
  and its displacement toward later cells.
- A velocity–pseudotime blending step that fuses the supervised time head with
  velocity-graph diffusion pseudotime for a globally grounded temporal
  coordinate.
- Evaluation on five biologically diverse datasets against four baselines
  (Velocyto, scVelo dynamical, VeloVI, TFvelo) with 8 metrics.

## Key results (cross-dataset means, 5 datasets — Table IV)

Higher is better for every metric; best value per column in **bold**; "–" marks
undefined or non-applicable entries.

| Method | ICCoH ↑ | CBDir ↑ | VelConf ↑ | PT-Spear ↑ | PT-DistCorr ↑ | PT-Cons ↑ | RootAcc ↑ | Gene-R²ₛₚₗ ↑ |
|--------|:------:|:------:|:--------:|:---------:|:------------:|:--------:|:--------:|:-----------:|
| **TARVI** | 0.776 | **0.933** | **0.927** | **0.747** | **0.771** | **0.986** | 0.671 | **0.477** |
| VeloVI | 0.787 | 0.931 | 0.920 | 0.331 | 0.341 | 0.904 | 0.723 | 0.443 |
| TFvelo | **0.933** | 0.569 | 0.915 | 0.397 | 0.412 | 0.794 | 0.655 | – |
| scVelo dynamical | – | – | – | 0.604 | 0.628 | 0.852 | **0.814** | – |
| Velocyto | 0.813 | 0.505 | 0.755 | 0.700 | 0.684 | 0.873 | 0.745 | – |

## Related material

| Item | Description |
|------|-------------|
| [`../supplementary/supplementary_material.pdf`](../supplementary/supplementary_material.pdf) | Full component ablation (TF / VTC / VPT blending across all 5 datasets), background theory, reproducibility & scope notes — with LaTeX source |
| [`../notes/TARVI_methodology_notes.pdf`](../notes/TARVI_methodology_notes.pdf) | Extended methodology notes (LaTeX source alongside) |
| [`../slides/TARVI.pptx.pdf`](../slides/TARVI.pptx.pdf) | Presentation slides |
| [`../docs/assets/mercon_certificate.jpg`](../docs/assets/mercon_certificate.jpg) | MERCon 2026 Best Paper Award certificate |
| [repository root](../) | Code, installation, and reproduction instructions |

## Citation

```bibtex
@inproceedings{adhikari2026tarvi,
  title     = {TARVI: Transcription-Factor Aided RNA Velocity Inference with
               Supervised Latent Time and Velocity-Pseudotime Blending},
  author    = {Adhikari, Chandula and Dassanayake, Sandeep and Herath, Damayanthi},
  booktitle = {2026 Moratuwa Engineering Research Conference (MERCon)},
  year      = {2026},
  publisher = {IEEE},
  note      = {Best Paper Award, Biomedical Engineering and Instrumentation track}
}
```

See [`../CITATION.cff`](../CITATION.cff) for machine-readable metadata.
