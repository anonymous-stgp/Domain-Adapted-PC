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

- Transfer learning methods
- Geographic-neighbor transfer methods
- Pooled data modeling methods

Our transfer learning approach selects source data based on a supervised weighted Kolmogorov–Smirnov metric. Geographic matching selects turbines that are geographically closest to the target turbine, while the pooled strategy uses data from all training turbines.

Transfer learning:
- thinnedSV (R)
- Multi-layer NN (Python)
- thinned twinGP (R)

Geographic-neighbor transfer:
- Random Forest (R)
- XGBoost (Python)
- SVR (R)

Pooled training:
- XGBoost (Python)
- Graph NN (Python)
- BHM (R)
- twinGP (R)
- Binning (R)


### Note on BHM Implementation

We do not provide code for the **BHM (Bayesian Hierarchical Model)** method in this repository. The same leave-one-out experiment is already implemented in:

https://github.com/TAMU-AML/BHM-Terrain-Paper

In addition, the BHM method is significantly slower than the other methods considered in this paper. Including it in the reproduction scripts would substantially increase the time required to reproduce the results.



## Repository Structure

Each method is labeled with one of the prefixes (TF, G, P):

- TF: Transfer learning
- G: Geographic-based methods
- P: Pooled methods

For example:

- P_binning: Binning applied to pooled data
- G_SVR: SVR applied to geographically closest turbines
- TF_thinned_SV: thinned SV applied within the transfer learning framework

The repository is organized as follows:

```
STGP-Terrain-Aware-Power-Curve/
│
├── data/
│   ├── Turbine_i_2017.csv
│   ├── Turbine_i_2018.csv
│   ├── terrain_features.csv
│   ├── turbine_locations.csv
│   └── processed_data
│
├── run_all_loto.bat
│
├── run_all_dfp.bat
│
├── compile_results.py
│
│
├── code/
│
│   ├── matching.py
│   ├── Table1.R
│   ├── thinnedsv_source.R
│
│   ├── Table2_TF_thinned_SV.R
│   ├── Table2_TF_thinned_twinGP.R
│   ├── Table2_TF_ANN.py
│
│   ├── Table2_G_SVR.R
│   ├── Table2_G_XGBoost.py
│   ├── Table2_G_random_forest.R
│
│   ├── Table2_P_GNN.py
│   ├── Table2_P_XGBoost.py
│   ├── Table2_P_twinGP.R
│   ├── Table2_P_Binning.R
│
│   ├── matching_dfp.py
│   ├── Table1_dfp.R
│
│   ├── Table2_TF_thinned_SV_dfp.R
│   ├── Table2_TF_thinned_twinGP_dfp.R
│   ├── Table2_TF_ANN_dfp.py
│
│   ├── Table2_G_SVR_dfp.R
│   ├── Table2_G_XGBoost_dfp.py
│   ├── Table2_G_random_forest_dfp.R
│
│   ├── Table2_P_GNN_dfp.py
│   ├── Table2_P_XGBoost_dfp.py
│   ├── Table2_P_twinGP_dfp.R
│   └── Table2_P_Binning_dfp.R
│
├── results/
│   ├── final_results.csv
│   └── intermediate/
│
├──LICENSE
└── README.md
```

**data/**  
Contains turbine SCADA datasets, terrain features, turbine location data, and processed datasets. The processed data includes turbine selection based on different matching metrics. 
 
**data/processed_data/**  
Contains processed datasets generated from turbine matching procedures.

**code/**  
Contains implementations of main and benchmark methods.

**results/final_results.csv**  
Stores the aggregated results presented in the paper (Tables 1, 2, and 3). Running each method updates the corresponding row in this file.

**results/intermediate/**  
Contains intermediate outputs such as runtime logs and turbine-level prediction errors.

## Instructions to Reproduce Results

The process to reproduce the tables in the paper is straightforward. Follow the steps below:

1. Download the GitHub repository as a ZIP file and extract it to any directory on your computer.

2. Navigate to the `Domain-Adapted-PC` folder.

3. Inside this folder, hold **Shift**, right-click, and select **Open PowerShell window here**.

4. To reproduce **Table 1**, run the following command:
.\run_all_loto.bat

5. To reproduce **Table 2**, run the following command:
.\run_all_dfp.bat

6. After each method finishes running, the corresponding results will be written to results/intermediate.

7. Once you finished with computation, to get the summarized clean tables like the tables in the paper, run:
   python compile_results.py
   the tables will be stored in:
   results/final_results.csv

## Runtime Summary for the Tables

Matching takes 40 minutes. The runtime for **Table 1** is approximately **49 hours**.


### Tables 2

Method | Runtime
------ | -------
TF_thinned_SV | 3 hours
TF_thinned_twinGP | already computed in Table 1
TF_ANN | 1 hour
G_SVR | 1 hour
G_XGBoost | 1 minute
G_random_forest | 4 minutes
P_GNN | 1.5 hours
P_XGBoost | 1 minute
P_twinGP | 1 hour
P_Binning | 1 minute
**Total** | **~8 hours**

### Tables 3

Method | Runtime
------ | -------
TF_thinned_SV | 29 hours
TF_thinned_twinGP | already computed in Table 1
TF_ANN | 4 hours
G_SVR | 14 hours
G_XGBoost | 2 minutes
G_random_forest | 15 minutes
P_GNN | 19 hours
P_XGBoost | 24 minutes
P_twinGP | 5 hours
P_Binning | 2 minutes
**Total** | **~71 hours**


### Total Runtime

The total runtime for reproducing **Tables 2** and **Tables 3** is approximately **8 and 71 hours**.

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

