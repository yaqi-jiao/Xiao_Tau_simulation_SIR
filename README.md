# SIR Model – Project Fork
This repository is a fork of the original SIR model repository developed in DeMON lab.

## Modifications in this fork

This fork extends the original ROI-matching procedure to support cases where the observed tau data and the connectivity matrix are defined in different ROI spaces.

Main changes include:

- Support for both **observed tau ROIs > connectivity ROIs** and **connectivity ROIs > observed tau ROIs**.

- Simulations can run on the **full connectivity space**, while predictions are mapped or sliced back to the observed tau ROIs for evaluation.

- Added support for **higher-resolution connectivity parcellations**, where multiple fine-resolution connectivity ROIs can be mapped to a common coarse observed tau ROI. 

These modifications were introduced for the current project and are mainly contained in the scripts of this fork.

## Test data

The `data/` directory contains small synthetic datasets for validating and debugging the newly added ROI-mapping functionality. These datasets are designed specifically for testing the mapping workflow and are not intended for scientific analysis.

Two complementary test cases are provided:

#### 1. Tau ROI space larger than the connectivity ROI space

The alternative connectivity contains a 6-ROI subset of the 10 observed tau ROIs. This test is used to confirm that the original one-directional ROI matching remains compatible with the extended implementation.

A simple test run can be launched with:

```python 
python run.py \
    --model_name toy_restore_altSC \
    --input_data_name Input_SIR_toy_tau10.pkl \
    --connectivity_file Connectomes_toy_alt6.pkl \
    --SC toy_alt_sc \
    --epicenter_list ctx_lh_entorhinal \
    --simulated_protein tau \
    --protein_type Load \
    --T_total 100
```

For this case, the expected behavior is that the observed tau data are matched to the 6 ROIs available in the alternative connectivity before the corresponding simulation output is evaluated.

#### 2. Higher-resolution connectivity mapped to a coarse tau atlas

The observed tau input contains 8 coarse ROIs, while the alternative connectivity contains 16 ROIs. Entorhinal and precuneus ROIs correspond directly between the two spaces, whereas the hippocampus and amygdala are represented by multiple finer subregions in the connectivity atlas.

This test therefore verifies that the simulation space and evaluation space can be different: the SIR model runs on the full higher-resolution connectome, while the simulated output is mapped back to the coarse observed tau atlas before model evaluation.

A simple test run can be launched with:

```python
python run.py \
    --model_name toy_hipamy_mapping \
    --input_data_name Input_SIR_toy_coarse_HIPAMY.pkl \
    --connectivity_file Connectomes_toy_highres_HIPAMY.pkl \
    --SC SC \
    --epicenter_list ctx_lh_entorhinal \
    --simulated_protein tau \
    --protein_type Load \
    --T_total 100
```

The full-resolution prediction should remain in the 16-ROI space during simulation and only be transformed to the 8-ROI observed space when required for evaluation.

**Note:** T_total=100 is used here only for a quick functional test. It is not intended as a meaningful simulation setting.

# Original README from the upstream repository

The documentation below is retained from the original repository and describes the base SIR model, installation, data structure, and general usage.

# Simulation tau propatation in the brain using the SIR model

**Preprint:** Available on [arXiv](https://www.biorxiv.org/content/10.1101/2025.04.17.648358).

This repository contains Python code for an agent-based Susceptible-Infectious-Recovered (SIR) model designed to simulate tau propagation in Alzheimer’s disease.

Our model builds upon the original [SIR simulator](https://github.com/yingqiuz/SIR_simulator) by <u>Ying-Qiu Zheng</u>. This mechanistic framework simulates based on both **connectome-based spread** and **regional vulnerability** across brain regions, enabling in silico testing of distinct pathophysiological mechanisms via tunable regional parameters affecting tau *synthesis* and *clearance*.

## Key Enhancements
We extend the original SIR framework to allow **regional vulnerability** not only modify tau synthesis and clearance, but also tau *spread* and *misfolding* rates. This refined model allows for more realistic simulations of disease progression.

## Data Sources
We gratefully acknowledge:
- <u>Justine Hansen</u> for providing human brain connectivity ([link](https://github.com/netneurolab/hansen_many_networks/tree/v1.0.0)) and regional neurotransmitter receptor data ([link](https://github.com/netneurolab/hansen_receptors))
- [Allen Human Brain Atlas](https://human.brain-map.org/) for gene expression data (e.g., MAPT, APOE)
- [abagen](https://abagen.readthedocs.io/en/stable/) for processing transcriptomic data

The framework supports hyperparameter tuning, null model simulations, and automated summarization of simulation outputs.


# Directory Structure

- **Script Directory:**  
  All scirpts used are in `scripts/`.

- **Input Directory:**  
  All input files must be stored in the `data/` directory under the root path.

- **Output Directory:**  
  Results will be saved to the `results/` directory under the root path.


```bash
project/
├── README.md   # This file
├── data/                                    # Input and processed data files
│   ├── Input_SIR_example.pkl                # Main simulation data
│   ├── regional_vulnerability_example.csv   # Regional vulnerability data
│   ├── Connectomes_all.pkl                  # All connectivity matrices
│   ├── Null_connectomes_all.pkl             # All null connectivity matrices
│   ├── Tau_subtypes_example.csv             # Subtype data
├── scripts/
│   ├── requirements.txt  # Python packages needed for the model
│   ├── run.py            # Run with default hyperparameters
│   ├── run_hypertune.py  # Run with hyperparameter tuning
│   ├── run_null_model.py # Run null model simulation
│   ├── User_input_settings.txt # Essential simulation settings
│   └── summary/
│   │   ├── summary.py               # Functions to extract and summarize simulation metrics
│   │   ├── run_summary_one_conn.py  # Run simulation results summarization for a single connectivity
│   │   └── run_summary_all_conns.py # Run simulation results summarization for all connecvities tested
│   └── src/
│       ├── evaluation.py        # Evaluation functions and metrics computation
│       ├── init_run.py          # Initialization and data loading routines
│       ├── log_redirector.py    # Custom logging setup
│       ├── params.py            # Command-line argument parsing and settings
│       ├── results_traker.py    # Tracking and saving simulation results
│       ├── simulated_atrophy.py # Core simulation functions for atrophy and protein spread
│       ├── summary.py           # Functions to extract and summarize simulation metrics
│       └── utils.py             # Utility functions for plotting and other tasks
└── results/
```

# Installation

## Requirements

This project requires Python 3.7 or later along with the following packages:

- numpy ≥ 1.19.0
- pandas ≥ 1.1.0
- matplotlib ≥ 3.2.0
- seaborn ≥ 0.11.0
- scipy ≥ 1.5.0
- scikit-learn ≥ 0.23.0
- joblib ≥ 1.0.0

It is recommended to create a virtual environment and install the packages using pip:

```bash
python -m venv simulation_env
source simulation_env/bin/activate
pip install -r requirements.txt
```

# Usage

## 1. Configuration
Simulation parameters and settings are defined in `User_input_settings.txt`. Modify this file to set parameters such as:
```text
model_name = "Your_model_name"            # Model name - use in output folder name
input_data_name = "Input_SIR_example.pkl" # Input data file name
epicenter_list = "ctx_lh_entorhinal"      # List of epicenter names
simulated_protein = "tau"                 # Name of simulated protein, e.g., tau, amyloid, etc.
protein_type = "Presence"                 # Simulated protein type, dataframe column name of the observed tau in `input_data_name` file, e.g., Presence, Load, Subtype1.
```

Detailed parameters and hyperparameters are listed in `src/params.py`.
Adjust these values according to your experimental setup.

## 2. Running the SIR model
### A. Group-avegraged data
#### Single Run
Run a single SIR model, execute:
```bash
python run.py
```
or you could also change simulation parameters here:
```bash
python run.py --model_name SC-only --epicenter_list ctx_lh_entorhinal
```
#### Hyperparameter Tuning
Run the SIR model with hyperparameters tunning, execute:
```bash
python run_hypertune.py --synthesis_var MAPT
```
Or add the parameter in `User_input_settings.txt`: synthesis_var = "MAPT"  

***Note:*** `synthesis_var`,`spread_var`, `misfold_var` or `clearance_var` (>=1 of the four parameters) are required to be set in `User_input_settings.txt` for the hyperparameter tuning. 

Hyperparameters tuned are listed in the `run_hypertune.py` file.

This script conducts a grid search over hyperparameters (e.g., `p_stay`, `trans_rate`, `v`), runs simulations for each combination, and records the best-performing configuration.

#### Null Brain Connectivity with Tuning
Run the SIR model for null brain connecvity with hyperparameters tunning, execute:
```bash
python run_null_model.py --model_name MySIRmodel_null_model0-1 --SC neurotransmission_similarity
```
***Note:*** 
* `null_model<i>-<j>` is essential to add in `model_name` in order to successfully run the null model simulation. 
* `--SC <connectivity_name>` is also required to be set, orelse the script will use the default connecvity in `Input_SIR_example.pkl` file to simulate, which, in most cases, is not the desired null connectivity.

This script iterates over all null brain connectivities, treating each as a separate null model iteration, performs hyperparameter tuning for each iteration, and consolidates the simulation results.

### B. Individual-level data
#### running with pre-defined best hyperparmeters set

Run the SIR model using **individualized (subject-level) data** for all participants, execute:
```bash
python run_individualized.py --individual_data_file data_individual.pkl
```
Or add the parameter in `User_input_settings.txt`: individual_data_file = "data_individual.pkl"

This will run the SIR model separately for each subject, using subject-specific inputs (e.g., tau, connectivity, ROI size, and regional vulnerability variables) provided in the individual data file.

***Note:*** 
* The hyperparameter must be pre-defined here, or else will use the default hyperparameters. Please define the hyperparaemters set that fits your group of data the best. You can run `A. Group-avegraged data` - `Hyperparameter Tuning` on your group-averaged data first to get the best hyperparameters set.

* The list of subjects to be simulated must be provided in a text file named: `Subject_IDs.txt`.

  **Format requirements:**
  - One subject ID per line
  - Subject IDs must match the column names / dictionary keys used in data_individual.pkl

  Example:
  ```bash
  subj_001
  subj_002
  subj_003
  ```

### 3. Result Summarization

***Note:***  
* Need to reconfigure in each run file, namely `summary/run_summary_one_conn.py` and `summary/run_summary_all_conns.py`.
* `--result_path` could be the <u>*extact path*</u> for one run of the model, or a <u>*parent folder*</u> that contains folders of multiple runs that you would like to summarize together

#### Summarize for Single Connectivity
To summarize results for a single brain connectivity and regional vulnerabilities (e.g., as shown in **Fig.2**), run the following command to extract and aggregate simulation evaluation metrics:
```bash
python run_summary_one_conn.py \
  --result_path /fullpath/to/results \
  --connectivity SC \
  --epicenter lh_ctx_entorhinal \
  --input_data_name Input_SIR_example.pkl
```

#### Summarize for All Connectivities Tested
To summarize results across all brain connectivities and regional vulnerabilities (e.g., as shown in **Fig.3c-f**), run the following command to extract and aggregate simulation evaluation metrics:
```bash
python run_summary_all_conns.py \
  --result_path /path/to/results \
  --index_conn 2 \
  --epicenter ctx_lh_entorhinal \
  --input_data_name Input_SIR_example.pkl \
  --connectivity_file Connectomes_all.pkl
```
***Note:*** Before running the script, please review and update the **connectivity measure list** (`conn_list`) and the **regional factor dictionary** (`factors_dict`) in the source code if necessary, to ensure they align with your specific simulation model settings.

## Contributing
Contributions and suggestions are welcome! Please open an issue or submit a pull request if you encounter any bugs or have suggestions for improvements.

## Acknowledgments
* The original SIR model by Ying-Qiu Zheng provided the foundation for this work.
* The adapted Python version by Vincent Bazinet, we adapted the `simulated_atrophy.py` file from his work.

## References
* [Zheng, Y. Q. et al., (2019).](https://academic.oup.com/brain/article/146/1/321/6533638) Local vulnerability and global connectivity jointly shape neurodegenerative disease propagation. PLoS Biology.
* [Shafiei, G. et al., (2023).](https://academic.oup.com/brain/article/146/1/321/6533638) Network structure and transcriptomic vulnerability shape atrophy in frontotemporal dementia. Brain.

## Contact
For any questions or inquiries, please contact [@xiaoyucaly](https://github.com/xiaoyucaly): caly.xiaoyu@gmail.com

## License
You can use, share, and adapt the code as long as you give proper credit and notify the author.
