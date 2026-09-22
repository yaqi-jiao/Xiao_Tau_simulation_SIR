# Toy Data for Modified SIR model

This folder contains small synthetic datasets used to test the ROI-mapping extensions implemented in this modified SIR model.

The toy datasets are intended **only for code testing, validation, and debugging**. The connectivity matrices, ROI sizes, and tau values are synthetic and should not be interpreted as biologically meaningful data.

## Relationship to the project workflow repository

The toy input datasets used here are prepared as part of the project-level workflow maintained in a separate repository:

[Subcortex-SIR](<https://github.com/yaqi-jiao/Subcortex_SIR>)

The data-generation and preprocessing scripts themselves are maintained in the project repository rather than duplicated here.

## General Data Structure

The toy datasets follow the same input conventions as the main SIR workflow.

Two types of files are used:
    
### 1. Main SIR input files

   Contain observed tau data together with a default connectivity representation.

   Main input file follow the structure:


```text
    {
        "tau": pandas.DataFrame,
        "conn": {
            "conn": numpy.ndarray,
            "SC_len": numpy.ndarray or None,
            "name": list,
            "ROI_size": dict
        }
    }
```

`tau`

- **Type:** `pandas.DataFrame`

- **Shape:** `(n_observed_ROIs, n_tau_types)`

- **Description:** Observed or synthetic tau values used as the reference data for simulation evaluation.
  
`conn['conn']`

- **Type:** `numpy.ndarray`

- **Shape:** `(n_observed_ROIs, n_observed_ROIs)`

- **Description:** Default connectivity matrix associated with the observed ROI space
  
`conn['SC_len']`

- **Type:** `numpy.ndarray` or None
  
- **Shape:** `(n_observed_ROIs, n_observed_ROIs)` when available

- **Description:** Structural connection-length matrix aligned with the default connectivity.

`conn["name"]`

- **Type:** `list`

- **Length:** `n_observed_ROIs`

- **Description:** ROI labels defining the observed tau space.

`conn["ROI_size"]`

- **Type:** `dict`

- **Description:** ROI sizes keyed by the ROI labels in `conn["name"]`.
    

### 2. Alternative connectivity files

    Contain a replacement connectivity matrix and the ROI information associated with that connectivity space.
 
    Alternative connectivity files follow the structure:

```text
{
    "<connectivity_name>": numpy.ndarray,
    "labels": list,
    "ROI_size": dict,
    "SC_len": numpy.ndarray or None
}

```

`connectivity_name`

- **Type:** `numpy.ndarray`

- **Shape:** `(n_connectivity_ROIs, n_connectivity_ROIs)`

- **Description:** Alternative connectivity matrix used to replace the default connectivity during simulation.

`labels`

- **Type:** `list`

- **Length:** `n_connectivity_ROIs`

- **Description:** ROI labels corresponding to the rows and columns of the alternative connectivity matrix.

`ROI_size`

- **Type:** `dict`

- **Description:** ROI sizes aligned with the alternative connectivity labels.

`SC_len`

- **Type:** `numpy.ndarray` or `None`

- **Shape:** `(n_connectivity_ROIs, n_connectivity_ROIs)` when available

- **Description:** Structural connection-length matrix aligned with the alternative connectivity.


## Directory Structure

```text
data/
├── README.md  # This file
│
├── toy_for_restore_altSC/  # 10 observed ROIs → 6 connectivity ROIs
│   ├── Input_SIR_toy_tau10.pkl
│   └── Connectomes_toy_alt6.pkl
│
├── toy_for_altSC/  # 
│   ├── 
│   └── 
│
└── toy_hipamy_mapping/  # 8 coarse observed ROIs ← 16 connectivity ROIs 
    ├── Input_SIR_toy_coarse_HIPAMY.pkl
    └── Connectomes_toy_highres_HIPAMY.pkl

```
Each folder contains one main SIR input file and one alternative connectivity file.

## Test Scenarios

### 1. toy_for_restore_altSC

This dataset tests the original ROI-matching case where the observed tau data contain more ROIs than the alternative connectivity.

#### Files

```text
Input_SIR_toy_tau10.pkl
    Observed tau space: 10 ROIs
    Tau columns: Load, Presence
    Default connectivity: 10 × 10

Connectomes_toy_alt6.pkl
    Alternative connectivity space: 6 ROIs
    Connectivity key: toy_alt_sc
    Alternative connectivity: 6 × 6
    SC_len: 6 × 6
```

The six alternative-connectivity ROIs are a subset of the ten observed tau ROIs.

#### Mapping tested

```text
Observed tau: 10 ROIs
        ↓
Match ROI labels
        ↓
Alternative connectivity: 6 ROIs
        ↓
Select overlapping tau ROIs
        ↓
Simulation and evaluation in the matched 6-ROI space
```

### 2. toy_hipamy_mapping

This dataset tests the higher-resolution connectivity mapping workflow.

The observed tau data are defined in a coarse 8-ROI space, while the alternative connectivity contains 16 ROIs.

#### Files

```text
Input_SIR_toy_coarse_HIPAMY.pkl
    Observed tau space: 8 coarse ROIs
    Tau column: Load
    Default connectivity: 8 × 8

Connectomes_toy_highres_HIPAMY.pkl
    Alternative connectivity space: 16 ROIs
    Connectivity key: SC
    Alternative connectivity: 16 × 16
    SC_len: None
```

#### Mapping tested

```text
Observed tau
8 coarse ROIs
        ↓
Higher-resolution connectivity
16 ROIs
        ↓
Run SIR on the full 16-ROI connectome
        ↓
Fine-to-coarse HIP/AMY mapping
        ↓
Prediction mapped back to 8 observed ROIs
        ↓
Evaluation
```

## Reproducing the Toy Inputs

The toy input files stored here are generated or prepared in the project-level workflow repository:

[Subcortex_SIR](<https://github.com/yaqi-jiao/Subcortex_SIR>)

The intended workflow is:

```text
Project / workflow repository
        │
        ├── generate / prepare toy data
        ├── construct SIR-compatible input
        └── construct alternative connectivity
                │
                ▼
        copy test inputs to
        SIR model repository/data/
                │
                ▼
        run model-level ROI-mapping tests
```

This separation keeps project-specific data preparation outside the core SIR model repository while retaining the minimal test datasets required to validate model functionality.

## Notes

- All values are synthetic.

- These datasets are intentionally small so that ROI ordering and mapping behavior can be inspected manually.

- They should not be used for biological interpretation or scientific analysis.

- For the higher-resolution HIP/AMY example, the data files contain the coarse and fine ROI definitions; the fine-to-coarse mapping behavior is handled by the simulation/evaluation code rather than being stored as an additional mapping object in the connectivity file.