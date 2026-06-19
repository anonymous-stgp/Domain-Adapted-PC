# Code for Domain Adapted Power Curve for Across Wind Farm Applications

## Datasets

The experiments in this paper use wind turbine SCADA datasets collected from a utility-scale wind farm. The dataset contains measurements from **66 turbines** recorded at **10-minute intervals**.

Each turbine dataset is stored as a CSV file and contains measurements for multiple operational variables. The datasets are organized by turbine ID and year.


### Turbine Dataset

The turbine datasets consist of SCADA measurements from **2017 and 2018** for all turbines. Each CSV file corresponds to a single turbine and includes measurements such as:

- wind_speed
- temperature
- wind_direction
- turbulence_intensity
- power

Each turbine file contains approximately **40,000–50,000 observations**.


### Terrain Data

A separate CSV file contains terrain features for all **66 turbines**. The terrain variables include:

- slope
- rix
- ridge

These terrain features are used in the second stage of matching in the terrain-aware transfer learning framework.


## Code

The implementation of the proposed method and the benchmark methods is written in **R** and **Python**. Three categories of methods are implemented:

- Transfer learning methods (WD-based, our proposed approach)
- Geographic-neighbor transfer methods
- Pooled data modeling methods

Our transfer learning approach selects source data based on a supervised Weighted Dissimilarity (WD) metric. Geographic matching selects turbines that are geographically closest to the target turbine, while the pooled strategy uses data from all training turbines.

Transfer learning (WD-based, K=7):
- thinnedSV (R)
- thinned twinGP (R) — computed as part of `Table1.R` / `Table1_dfp.R` (weighted_ks metric)
- ANN (Python)
- XGBoost (Python)

Geographic-neighbor transfer (K=7):
- ANN (Python)
- thinned twinGP (R)
- Random Forest (R)
- XGBoost (R)
- SVR (R)

Pooled training (all turbines):
- ANN (Python)
- thinned twinGP (R)
- XGBoost (R)
- GNN (Python)
- Binning (R)

The **distant farm planning (DFP)** experiment uses only the **Transfer learning** and **Geographic-neighbor transfer** groups; the Pooled group is not part of the DFP evaluation.


### Note on STGP and BHM

We do not provide code for **STGP** or the **BHM (Bayesian Hierarchical Model)** method in this repository:

- STGP results are taken from a separate paper/repository and are included in `compile_results.py` as hardcoded reference values for comparison only.
- BHM is implemented in a separate repository: https://github.com/TAMU-AML/BHM-Terrain-Paper. BHM is also significantly slower than the other methods considered in this paper, so it is not reproduced here.

Both STGP and BHM values are hardcoded as constants at the top of `compile_results.py` (see `HARDCODED_VALUES`). Edit those constants directly if the source numbers are updated.


## Repository Structure

Each method is labeled with one of the prefixes (TF, G, P):

- TF: Transfer learning (WD-based)
- G: Geographic-neighbor based methods
- P: Pooled methods

For example:

- `Table2_P_binning.R`: Binning applied to pooled data
- `Table2_G_SVR.R`: SVR applied to geographically closest turbines
- `Table2_TF_thinned_SV.R`: thinned SV applied within the WD transfer learning framework

The repository is organized as follows:

```
STGP-Terrain-Aware-Power-Curve/
│
├── data/
│   ├── Turbine_i_2017.csv
│   ├── Turbine_i_2018.csv
│   ├── terrain_features.csv
│   ├── location.csv
│   └── processed_data/
│
├── run_all_loto.bat
│
├── run_all_dfp.bat
│
├── compile_results.py
│
├── code/
│   │
│   ├── matching.py
│   ├── matching_dfp.py
│   │
│   ├── Table1.R                       # LOTO metric comparison (WD, Sinkhorn, Mean, Marginal Energy)
│   ├── Table1_dfp.R                   # DFP metric comparison / DFP thinned twinGP (TF group)
│   ├── thinnedsv_source.R             # helper sourced by Table2_TF_thinned_SV(.R/_dfp.R)
│   │
│   ├── Table2_TF_thinned_SV.R
│   ├── Table2_TF_thinned_SV_dfp.R
│   ├── Table2_TF_ANN.py
│   ├── Table2_TF_ANN_dfp.py
│   ├── Table2_TF_XGBoost.py
│   ├── Table2_TF_XGBoost_dfp.py
│   │
│   ├── G_thinned_twinGP.R
│   ├── G_thinned_twinGP_dfp.R
│   ├── Table2_G_ANN.py
│   ├── Table2_G_ANN_dfp.py
│   ├── Table2_G_rf.R
│   ├── Table2_G_rf_dfp.R
│   ├── Table2_G_SVR.R
│   ├── Table2_G_SVR_dfp.R
│   ├── Table2_G_XGBoost.R             # LOTO geographic-neighbor XGBoost
│   ├── Table2_G_XGBoost_dfp.R
│   │
│   ├── P_thinned_twinGP.R
│   ├── Table2_P_ANN.py
│   ├── Table2_P_XGBoost.R
│   ├── Table2_P_XGBoost_dfp.R
│   ├── Table2_P_GNN.py
│   ├── Table2_P_GNN_dfp.py
│   ├── Table2_P_binning.R
│   ├── Table2_P_binning_dfp.R
│   │
│   ├── Figure5_twingp.R               # K=2..10 sweep, thinned twinGP, ensemble vs. concat
│   └── Figure5_ann.py                 # K=2..10 sweep, ANN, ensemble vs. concat
│
├── results/
│   ├── final_results.csv
│   ├── final_results_Table2_Metrics_LOTO.csv
│   ├── final_results_Table3_LOTO.csv
│   ├── final_results_Table4_DFP.csv
│   └── intermediate/
│
├── LICENSE
└── README.md
```

> **Note:** `Table2_P_ANN.py`, `Table2_G_ANN.py`/`_dfp.py`, `Table2_TF_ANN.py`/`_dfp.py`, and `Figure5_ann.py` use a Keras/TensorFlow feedforward network (8-16-8, ReLU, Adam). All other Python scripts that fit standard models (XGBoost, GNN) use scikit-learn / LightGBM / PyTorch as appropriate; see each script's imports.

**data/**
Contains turbine SCADA datasets, terrain features, turbine location data, and processed datasets. The processed data includes turbine selection based on different matching metrics.

**data/processed_data/**
Contains processed datasets generated from turbine matching procedures (`matching.py` / `matching_dfp.py`).

**code/**
Contains implementations of the main and benchmark methods.

**results/final_results.csv**
Stores the aggregated results presented in the paper (Tables 2, 3, and 4). Running `compile_results.py` regenerates this file along with one clean per-table CSV (`final_results_Table2_Metrics_LOTO.csv`, `final_results_Table3_LOTO.csv`, `final_results_Table4_DFP.csv`).

**results/intermediate/**
Contains intermediate outputs such as runtime logs and turbine-level prediction errors, written by each individual script.

## Instructions to Reproduce Results

The process to reproduce the tables in the paper is straightforward. Follow the steps below:

1. Download the GitHub repository as a ZIP file and extract it to any directory on your computer.

2. Navigate to the `Domain-Adapted-PC` folder.

3. Inside this folder, hold **Shift**, right-click, and select **Open PowerShell window here**.

4. To reproduce **Table 2** and **Table 3** (LOTO experiments), run:
   ```
   .\run_all_loto.bat
   ```

5. To reproduce **Table 4** (distant farm planning / DFP experiment), run:
   ```
   .\run_all_dfp.bat
   ```
   This only runs the Transfer learning (TF) and Geographic-neighbor (G) groups; the Pooled (P) group is skipped, since it is not part of the DFP table.

6. After each script finishes running, its results are written to `results/intermediate/`.

7. Once you have finished running the scripts you need, run:
   ```
   python compile_results.py
   ```
   to produce the summarized, paper-formatted tables. The combined output is stored in `results/final_results.csv`, and one clean CSV per table is also written (`final_results_Table2_Metrics_LOTO.csv`, `final_results_Table3_LOTO.csv`, `final_results_Table4_DFP.csv`).

If you only want to run a subset of the methods, you can comment out the corresponding method blocks in `run_all_loto.bat` / `run_all_dfp.bat`. `compile_results.py` will simply leave the corresponding RMSE/gap cells blank (NaN) for any method that hasn't been run yet.

## Runtime Summary for the Tables

**Note:** Several scripts have been substantially rewritten since these runtimes were last measured (e.g., the ANN scripts now use Keras/TensorFlow instead of scikit-learn, and the twinGP scripts were restructured into standalone `G_thinned_twinGP.R` / `P_thinned_twinGP.R` files). Runtimes for those updated scripts are left blank below until re-measured; all other entries reflect the original measurements.

Matching takes 40 minutes. The runtime for **Table 2** is approximately **49 hours**.

### Table 3

Method | Runtime
------ | -------
TF_thinned_SV | 3 hours
TF_thinned_twinGP | already computed in Table 2
TF_ANN |
G_SVR | 1 hour
G_XGBoost | 1 minute
G_random_forest | 4 minutes
P_GNN | 1.5 hours
P_XGBoost | 1 minute
P_thinned_twinGP |
P_Binning | 1 minute
**Total** |

### Table 4

Method | Runtime
------ | -------
TF_thinned_SV | 29 hours
TF_thinned_twinGP | already computed in Table 2
TF_ANN |
G_SVR | 14 hours
G_XGBoost | 2 minutes
G_random_forest | 15 minutes
**Total** |


### Total Runtime

The total runtime for reproducing **Table 3** and **Table 4** is approximately **TBD** (see note above).

If you only want to run a subset of the methods, you can comment out the corresponding method blocks in the files:

- `run_all_loto.bat`
- `run_all_dfp.bat`

### Note 1 on Runtime Performance

All experiments are implemented and executed on the Georgia Tech Partnership for an Advanced Computing Environment (PACE) high-performance computing cluster. Jobs are submitted through the Inferno service using CPU-only Intel processor nodes, with each job allocated one node and eight cores. The Inferno partition provides access to shared research computing resources with standard SLURM-based job scheduling, enabling reproducible and parallelized execution of the computationally intensive model fitting and evaluation pipelines described in this work.

### Note 2 on Runtime Performance

We observed that the runtime of several methods is significantly faster when **OpenBLAS** is used for linear algebra operations.

If OpenBLAS is not configured in your R installation, we recommend installing it using the instructions provided here:

https://github.com/david-cortes/R-openblas-in-windows

Without OpenBLAS, the runtime of some methods may be **up to two times slower**.
