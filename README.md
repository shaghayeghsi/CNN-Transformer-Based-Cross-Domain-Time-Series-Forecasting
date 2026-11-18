# CNN–Transformer for Multi-Domain Time-Series Forecasting

A deep-learning project exploring multi-step time-series forecasting across electricity demand, photovoltaic generation, and cloud-computing workload data. The project combines signal decomposition, feature engineering, clustering, a CNN–Transformer encoder, and a two-stage training strategy with encoder reuse/fine-tuning.

> **Project status: Experimental / Work in Progress**
>
> The metrics below are the results currently recorded in the notebook. They are preliminary and should not be interpreted as final benchmark claims. The preprocessing and evaluation pipeline is being reviewed for temporal data leakage, consistent target scaling, and reproducibility. Updated results will be published after that validation.

## Overview

The goal is to investigate whether a shared sequence-encoding approach can learn useful temporal representations for different forecasting problems. The current implementation uses a 72-step historical input window to predict the next 12 time steps.

The pipeline currently includes:

- **Signal decomposition:** Empirical Mode Decomposition (EMD) and Discrete Wavelet Transform (DWT).
- **Feature engineering:** Lagged observations, rolling statistics, and calendar/time features where available.
- **Clustering:** PCA-based representation followed by Particle Swarm Optimization (PSO) to select a K-Means cluster count.
- **Sequence modeling:** Causal 1D CNN layers followed by Transformer encoder blocks with multi-head self-attention.
- **Transfer-learning workflow:** A reusable encoder and two-stage training (train the prediction head with the encoder frozen, then fine-tune the encoder).
- **Optimization:** Adam, Huber loss, gradient clipping, learning-rate scheduling, and early stopping.
- **Evaluation:** MAE, MSE, RMSE, and coefficient of determination (R²), including horizon-wise error analysis in the electricity-demand experiment.

## Datasets and Current Experimental Results

The notebook contains experiments on the following datasets:

| Dataset | Forecast target in the notebook | Input history | Forecast horizon | Current recorded R² | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| Ontario electricity demand | `hourly_demand` | 72 time steps | 12 time steps | **0.9800** | 0.010233 | 0.013348 |
| Renewables.ninja PV profiles | `DE_pv_national_current` | 72 hours | 12 hours | **0.9404** | 0.039850 | 0.060359 |
| Google Borg traces | CPU usage parsed from `random_sample_usage` (`cpus`) | 72 time steps | 12 time steps | **0.3297** | 0.007988 | 0.022411 |

**Important interpretation notes**

1. The notebook applies target scaling and inverse scaling, but the full preprocessing pipeline has not yet been audited end-to-end. Confirm that each reported metric is expressed in the original target units before interpreting MAE/RMSE across datasets.
2. The three targets have different meanings and scales. Their MAE and RMSE values must not be compared directly across datasets.
3. R² is scale-independent, but it is still affected by the test split, target definition, preprocessing, and possible leakage. It does not by itself establish that one dataset/model experiment is better than another.
4. The current Borg result is materially weaker in R² than the other two experiments. This is retained transparently rather than omitted.
5. These are notebook-recorded results, not yet results from a leakage-audited, repeated, or statistically controlled benchmark.

## Model Architecture

The current forecasting model is organized into two components:

### 1. CNN–Transformer encoder

- Causal 1D convolution layers capture local temporal patterns.
- A projection layer maps the convolutional representation to the Transformer embedding dimension.
- Positional embeddings provide sequence-order information.
- Transformer encoder blocks use multi-head self-attention, residual connections, layer normalization, and feed-forward layers.
- Global average pooling and a dense projection produce a compact sequence representation.

### 2. Forecasting head

A dense prediction head maps the encoder representation to a vector containing the next 12 forecast values.

### Two-stage optimization

1. **Stage A – Head training:** Freeze the encoder and train the forecasting head.
2. **Stage B – Fine-tuning:** Unfreeze the encoder and fine-tune the complete model using a lower learning rate.

The notebook also supports loading encoder weights when a compatible pretrained checkpoint is available. For reproducibility, the origin of the checkpoint and the exact source/target pretraining protocol should be documented for each experiment.

## Main Configuration

| Setting | Current value |
|---|---:|
| Historical input length | 72 time steps |
| Forecast horizon | 12 time steps |
| Clustering window | 24 time steps |
| CNN filters | 64, then 128 |
| Transformer blocks | 2 |
| Attention heads | 2 |
| Attention key dimension | 32 |
| Feed-forward dimension | 128 |
| Encoder representation | 256 dimensions |
| Batch size | 8 |
| Stage A epochs | 3 |
| Stage B maximum epochs | 15 |
| Stage A learning rate | 5e-4 |
| Fine-tuning learning rate | 2.5e-4 |
| Optimizer | Adam |
| Loss | Huber |
| Random seed | 42 |

Some settings and preprocessing details differ between dataset-specific notebook sections. Consult the corresponding notebook cells before reproducing an individual experiment.

## Evaluation Metrics

- **MAE (Mean Absolute Error):** Average absolute prediction error.
- **MSE (Mean Squared Error):** Average squared prediction error.
- **RMSE (Root Mean Squared Error):** Square root of MSE, expressed in the target's scale.
- **R² (Coefficient of Determination):** Measures the proportion of target variance explained by the predictions relative to a mean-prediction baseline.

For multi-step forecasting, horizon-wise MAE/RMSE are also calculated in the Ontario experiment to examine how error changes with forecast distance.

## Repository Structure

```text
.
├── README.md
├── notebooks/
│   └── forecasting_experiments.ipynb
├── src/
│   ├── preprocessing.py
│   ├── signal_decomposition.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── model.py
│   └── evaluation.py
├── results/
│   ├── metrics/
│   └── figures/
├── requirements.txt
└── LICENSE
```

The structure above is the recommended organization for the repository. The current implementation is provided as a notebook; the modular `src/` files and results folders can be added as the project is refactored.

## Getting Started

### Requirements

The notebook uses Python and the following main libraries:

- NumPy
- pandas
- scikit-learn
- TensorFlow / Keras
- PyWavelets
- EMD-signal (PyEMD)
- pyswarms
- Matplotlib

Install the dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow PyWavelets EMD-signal pyswarms matplotlib
```

### Data setup

The datasets are not bundled with this repository. Obtain them from their respective providers, review their usage terms, and place them in a local data directory. Update the dataset paths and target-column settings in the notebook before running an experiment.

Dataset references:

- **Ontario electricity demand:** use the source and version corresponding to the `ontario_electricity_demand.csv` file used in the experiment.
- **Renewables.ninja:** https://www.renewables.ninja/
- **Google Cluster Data / Borg traces:** https://github.com/google/cluster-data

The exact data versions, download dates, licenses, filtering rules, and target definitions should be recorded before the experiments are treated as reproducible benchmarks.

### Run

1. Clone the repository.
2. Install the dependencies.
3. Download the datasets from their original sources.
4. Configure the file paths and target columns.
5. Run the notebook from top to bottom.

The original notebook was developed in a Google Colab environment and currently contains Google Drive path configuration. Those paths must be changed for a local or other hosted environment.

## Current Limitations and Next Steps

The following work is planned before presenting the results as publication-grade experiments:

- **Temporal leakage audit:** fit scalers, PCA, clustering, and any data-dependent preprocessing only on training data; ensure decomposition and feature construction do not use future observations.
- **Chronological evaluation:** define train/validation/test boundaries by timestamp and preserve them consistently across all models.
- **Scaling audit:** verify target transformations and inverse transformations; report errors in the original physical units where applicable.
- **Baseline comparison:** evaluate persistence/seasonal-naive, linear models, XGBoost or LightGBM, LSTM, GRU, CNN-only, and Transformer-only baselines under the same data split and forecast horizon.
- **Ablation study:** measure the individual contribution of EMD, DWT, PSO-based clustering, CNN layers, Transformer blocks, and encoder transfer.
- **Robustness:** repeat experiments across multiple random seeds and report mean and standard deviation.
- **Reproducibility:** pin dependency versions, document dataset versions, save split definitions and configuration files, and provide scripts for generating all tables and plots.
- **Forecast visualization:** include actual-versus-predicted plots and horizon-wise error curves for each dataset.

## Research and Practical Relevance

The project explores a reusable forecasting pipeline for several time-dependent systems:

- Electricity demand forecasting
- Photovoltaic generation forecasting
- Cloud workload / CPU utilization forecasting

Potential applications include energy planning, renewable integration, resource allocation, and data-center capacity management. These are application areas motivating the experiments; the current notebook does not constitute a deployed production system.

## Citation

If you use or build on this repository, please cite the repository URL and the relevant original dataset publications/providers. A formal citation for the proposed method will be added if the work is developed into a publication.

## License

Choose and add an appropriate open-source license before publishing. Also verify that the licenses and terms of use of the source datasets permit your intended redistribution and use.
