# XRF and ER-FTIR Analysis of Oil Paintings

Notebooks and supporting code for the non-invasive analysis of multilayer oil paintings using X-ray fluorescence (XRF) and external reflection Fourier-transform infrared spectroscopy (ER-FTIR).

The two workflows are independent. The XRF branch covers physics-based spectral simulation, augmentation, neural-network training and inference; the ER-FTIR branch covers feature extraction, PCA/HCA, kernel density estimation and projection of new spectra.

> **Repository note:** The folder names below describe the workflows documented in this project. Verify notebook locations and input paths against the current repository version before running them. Large datasets and trained checkpoints may not be included.

## Organization

```text
XRF/
├── XRF_simulation/
│   ├── Notebook/           # Simulation and augmentation notebooks
│   └── Utility_files/      # Json examples and supporting files for the notebooks
└── XRF_NN_model/           # Pre-training, fine-tuning and deterministic/MC-dropout inference notebooks
ER-FTIR/                    # ER-FTIR PCA and KDE notebook
```

## XRF notebooks

### `Main_sim.ipynb` — Synthetic spectrum generation

**Purpose:** Simulates layered configurations, synthetic XRF spectra and ground-truth labels for multilayer paintings, including sampled pigment compositions and layer properties.

**Inputs:** `xrf_model.py` (xrf layered model executable), `Density.json` (phase density file), `Pigment_range.json` (range of phases in the nominal pigment), `exspe40.ini` (X-ray tube spectrum), `xrf_common_dict.json` (detector efficiency).

**Output:** Simulated XRF spectra and associated stratigraphic labels, stored in a JSON files.

### `Generate_Augmented_Dataset.ipynb` — Background augmentation

**Purpose:** Inject experimental background into simulated spectra using background templates from, intensity scaling and Poisson noise.

**Inputs:** Synthetic XRF JSON files containing counts and labels; `smooth_background_templates.npz` containing the experimental background templates.

**Output:** Augmented JSON spectra retaining the original stratigraphic targets and source file.


### `Pretraining.ipynb` — Model pre-training

**Purpose:** Train the T10 multilayer XRF neural network on simulated spectra. The model predicts second-layer presence, log areal density and elemental composition. Its three original objectives are binary cross-entropy with logits, masked thickness MSE and masked composition soft-label cross-entropy, combined with moving-average normalization and thickness weighting.

**Inputs:** Synthetic training/validation JSON spectra and labels (the original configuration used a directory such as `cartella_air14cm_CuReplaced`).

**Outputs:** Pre-trained model checkpoints and weights, training and validation losses logs.

### `FineTuning.ipynb` — Model fine-tuning

**Purpose:** Adapt the pre-trained network to augmented spectra using the configured layer-freezing strategy, training and validation loops.

**Inputs:** Augmented training/validation JSON spectra; pre-trained model checkpoint (`best_checkpoint_pretrain_T10_FULL_all.pth`).

**Outputs:** Fine-tuned model checkpoint and weights, training and validation losses logs.

### `Inference.ipynb` — Deterministic and MC-dropout inference

**Purpose:** MC-dropout or deterministic inference on experimental XRF spectra. Set `INFERENCE_MODE` to `deterministic` for a single forward pass or `bayesian` for Monte Carlo dropout (40 stochastic passes in the example).

**Inputs:** Experimental spectra as JSON files,Fine-tuned checkpoint (`best_model_ft_T10_FULL.pth`).

**Outputs:** Predictions for layer presence, composition and log areal density; `.npz` inference caches.

## ER-FTIR notebook

### `ER_FTIR_Features_PCA_KDE_Projection.ipynb` — Feature-based analysis

**Purpose:** Demonstrate the complete analysis from spectra to PCA score plots and KDE, including out-of-sample projection.

**Steps:** Load `.SPA` spectra; apply Savitzky–Golay smoothing, SNV and first derivative; construct spectral-window features (`width`, `signed_p2p`, `energy`); standardize features; compute PCA and HCA; optimize KDE bandwidth by leave-one-out log-likelihood; visualize the 2D KDE; project new spectra using the **existing** feature definitions, scaler and PCA model without refitting.

**Inputs:** `.SPA` spectra in the example folders `data/Pellet/`, `data/Oil_Pigments/`, `data/Layer/`, `data/Layer_T/`; optional `.SPA` spectra in `data/New_Samples/` for projection.

**Outputs:** Feature vectors, PCA score plot, HCA groups, optimized KDE plots, and PC1/PC2 coordinates of newly projected spectra.

**Interpretation:** Out-of-sample PCA projection illustrates where new measurements lie relative to the reference samples; it is not, on its own, a classification accuracy test.

## Workflow overview

```text
XRF simulation → synthetic JSON → augmentation → pre-training
                                            ↓
                                      fine-tuning → Inference → .npz cache

ER-FTIR .SPA → preprocessing → feature extraction → PCA/HCA
                                             ↓
                                  LOO bandwidth → KDE
                                             ↓
                                  new-sample PCA projection
```

## Running the examples

1. Install the Python packages imported by the relevant notebook.
2. Supply its input files, spectra, background bank and/or model checkpoint.
3. Adjust the paths and analysis parameters in its configuration cells.
4. Run the notebook in order from a fresh kernel.

The repository documents the methodological workflow. Full numerical reproduction requires the original datasets and compatible trained checkpoints, which may not be distributed with the code.
