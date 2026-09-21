# Toy Data for ROI-Mapping Tests

This folder contains small synthetic datasets used to test the ROI-mapping extensions added to the SIR simulation pipeline.

The toy datasets are intended **only for code testing, validation, and debugging**. The connectivity matrices, ROI sizes, and tau values are synthetic and should not be interpreted as biologically meaningful data.

Two mapping scenarios are included:

1. **`toy_for_restore_altSC`** — tests the original case where the observed tau data contain more ROIs than the alternative connectivity.
2. **`toy_hipamy_mapping`** — tests higher-resolution connectivity, where several fine-resolution hippocampal/amygdala ROIs correspond to one coarse observed tau ROI.

---

# 1. `toy_for_restore_altSC`

## Purpose

This dataset tests ROI matching when:

- the observed tau data are defined on **10 ROIs**;
- the alternative connectivity is defined on a **6-ROI subset** of those ROIs.

This represents the case:

```text
Observed tau ROIs > Connectivity ROIs
```

The observed tau data are matched to the ROIs available in the alternative connectivity so that simulation and evaluation use the common ROI subset.

## 1.1 Main simulation data

**Filename:** `Input_SIR_toy_tau10.pkl`  
**File format:** Pickle (`.pkl`)  
**Top-level type:** Python `dict`

### Expected structure

```python
{
    "tau": pandas.DataFrame,
    "conn": {
        "conn": numpy.ndarray,
        "SC_len": numpy.ndarray,
        "name": list,
        "ROI_size": dict
    }
}
```

### `tau`

- **Type:** `pandas.DataFrame`
- **Shape:** `(10, 2)`
- **Columns:** `Load`, `Presence`
- **Description:** Synthetic observed tau values for 10 ROIs.

ROI order:

```text
ctx_lh_entorhinal
ctx_rh_entorhinal
ctx_lh_precuneus
ctx_rh_precuneus
ctx_lh_middletemporal
ctx_rh_middletemporal
ctx_lh_hippocampus
ctx_rh_hippocampus
ctx_lh_thalamus
ctx_rh_thalamus
```

### `conn["conn"]`

- **Type:** `numpy.ndarray`
- **Shape:** `(10, 10)`
- **Description:** Default toy connectivity matrix corresponding to the 10-ROI input space.

### `conn["SC_len"]`

- **Type:** `numpy.ndarray`
- **Shape:** `(10, 10)`
- **Description:** Toy structural connection-length matrix corresponding to the default 10-ROI connectivity.

### `conn["name"]`

- **Type:** `list`
- **Length:** `10`
- **Description:** ROI labels corresponding to the rows and columns of `conn["conn"]`.

### `conn["ROI_size"]`

- **Type:** `dict`
- **Number of entries:** `10`
- **Description:** Synthetic ROI sizes keyed by the 10 input ROI names.

---

## 1.2 Alternative connectivity

**Filename:** `Connectomes_toy_alt6.pkl`  
**File format:** Pickle (`.pkl`)  
**Top-level type:** Python `dict`

### Expected structure

```python
{
    "toy_alt_sc": numpy.ndarray,
    "labels": list,
    "ROI_size": dict,
    "SC_len": numpy.ndarray
}
```

### `toy_alt_sc`

- **Type:** `numpy.ndarray`
- **Shape:** `(6, 6)`
- **Description:** Alternative toy connectivity matrix used to replace the default connectivity.

### `labels`

- **Type:** `list`
- **Length:** `6`
- **Description:** ROI labels corresponding to the alternative connectivity.

ROI order:

```text
ctx_lh_entorhinal
ctx_rh_entorhinal
ctx_lh_middletemporal
ctx_rh_middletemporal
ctx_lh_thalamus
ctx_rh_thalamus
```

These 6 ROIs are a subset of the 10 observed tau ROIs.

The following observed ROIs are therefore not included in this alternative connectivity:

```text
ctx_lh_precuneus
ctx_rh_precuneus
ctx_lh_hippocampus
ctx_rh_hippocampus
```

### `ROI_size`

- **Type:** `dict`
- **Number of entries:** `6`
- **Description:** ROI sizes aligned with the 6 alternative-connectivity ROIs.

### `SC_len`

- **Type:** `numpy.ndarray`
- **Shape:** `(6, 6)`
- **Description:** Structural connection-length matrix aligned with `toy_alt_sc`.

---

## 1.3 Mapping tested

Conceptually:

```text
Observed tau: 10 ROIs
        |
        | match ROI labels
        v
Alternative connectivity: 6 ROIs
        |
        v
Select the 6 overlapping tau ROIs
        |
        v
SIR simulation and evaluation in the matched 6-ROI space
```

This toy dataset is used to confirm backward compatibility with the original one-directional ROI matching procedure.

---

# 2. `toy_hipamy_mapping`

## Purpose

This dataset tests a different situation in which the alternative connectivity has a **higher atlas resolution** than the observed tau data.

The observed tau input contains **8 coarse ROIs**, whereas the alternative connectivity contains **16 ROIs**. Entorhinal and precuneus ROIs are shared directly, while coarse hippocampus and amygdala ROIs are represented by multiple fine-resolution subregions in the alternative connectivity.

This represents the case:

```text
Higher-resolution connectivity ROIs > Observed tau ROIs
```

The intended workflow is to run the SIR model on the full higher-resolution connectivity and map the simulated prediction back to the coarse observed tau space for evaluation.

---

## 2.1 Coarse observed-tau input

**Generated filename:** `Input_SIR_toy_coarse_HIPAMY.pkl`  
**File format:** Pickle (`.pkl`)  
**Top-level type:** Python `dict`

### Expected structure

```python
{
    "tau": pandas.DataFrame,
    "conn": {
        "conn": numpy.ndarray,
        "SC_len": numpy.ndarray,
        "name": list,
        "ROI_size": dict
    }
}
```

### `tau`

- **Type:** `pandas.DataFrame`
- **Shape:** `(8, 1)`
- **Column:** `Load`
- **Description:** Synthetic observed tau values defined in the coarse evaluation space.

ROI order:

```text
ctx_lh_entorhinal
ctx_rh_entorhinal
ctx_lh_precuneus
ctx_rh_precuneus
Left_Hippocampus
Right_Hippocampus
Left_Amygdala
Right_Amygdala
```

### `conn["conn"]`

- **Type:** `numpy.ndarray`
- **Shape:** `(8, 8)`
- **Description:** Synthetic coarse connectivity matrix associated with the observed tau input.

### `conn["SC_len"]`

- **Type:** `numpy.ndarray`
- **Shape:** `(8, 8)`
- **Description:** Synthetic connection-length matrix for the coarse input connectivity.

### `conn["name"]`

- **Type:** `list`
- **Length:** `8`
- **Description:** Coarse ROI labels defining the observed/evaluation space.

### `conn["ROI_size"]`

- **Type:** `dict`
- **Number of entries:** `8`
- **Description:** Synthetic ROI sizes for the coarse atlas.

---

## 2.2 Higher-resolution alternative connectivity

**Generated filename:** `Connectomes_toy_highres_HIPAMY.pkl`  
**File format:** Pickle (`.pkl`)  
**Top-level type:** Python `dict`

### Expected structure

```python
{
    "SC": numpy.ndarray,
    "labels": list,
    "ROI_size": dict,
    "SC_len": None
}
```

### `SC`

- **Type:** `numpy.ndarray`
- **Shape:** `(16, 16)`
- **Description:** Synthetic higher-resolution alternative connectivity matrix.

### `labels`

- **Type:** `list`
- **Length:** `16`
- **Description:** ROI labels corresponding to the higher-resolution connectivity.

ROI order:

```text
ctx_lh_entorhinal
ctx_rh_entorhinal
ctx_lh_precuneus
ctx_rh_precuneus
Hippocampus_head_medial_division-lh
Hippocampus_head_lateral_division-lh
Hippocampus_body-lh
Hippocampus_tail-lh
Hippocampus_head_medial_division-rh
Hippocampus_head_lateral_division-rh
Hippocampus_body-rh
Hippocampus_tail-rh
Lateral_amygdala-lh
Medial_amygdala-lh
Lateral_amygdala-rh
Medial_amygdala-rh
```

### `ROI_size`

- **Type:** `dict`
- **Number of entries:** `16`
- **Description:** Synthetic ROI sizes aligned with the higher-resolution connectivity labels.

The finer hippocampal ROI sizes sum to the corresponding coarse hippocampal ROI size:

```text
Left hippocampus:
900 + 850 + 1200 + 650 = 3600

Right hippocampus:
900 + 850 + 1200 + 650 = 3600
```

The finer amygdala ROI sizes likewise sum to the corresponding coarse amygdala ROI size:

```text
Left amygdala:
950 + 650 = 1600

Right amygdala:
950 + 650 = 1600
```

### `SC_len`

- **Value:** `None`
- **Description:** No structural connection-length matrix is provided for this higher-resolution toy connectivity.

---

## 2.3 Intended fine-to-coarse mapping

The four coarse hippocampus/amygdala ROIs correspond to the following higher-resolution ROIs:

### Left hippocampus

```text
Left_Hippocampus
    <- Hippocampus_head_medial_division-lh
    <- Hippocampus_head_lateral_division-lh
    <- Hippocampus_body-lh
    <- Hippocampus_tail-lh
```

### Right hippocampus

```text
Right_Hippocampus
    <- Hippocampus_head_medial_division-rh
    <- Hippocampus_head_lateral_division-rh
    <- Hippocampus_body-rh
    <- Hippocampus_tail-rh
```

### Left amygdala

```text
Left_Amygdala
    <- Lateral_amygdala-lh
    <- Medial_amygdala-lh
```

### Right amygdala

```text
Right_Amygdala
    <- Lateral_amygdala-rh
    <- Medial_amygdala-rh
```

The following ROIs are shared directly between the coarse and high-resolution spaces:

```text
ctx_lh_entorhinal
ctx_rh_entorhinal
ctx_lh_precuneus
ctx_rh_precuneus
```

---

## 2.4 Mapping tested

Conceptually:

```text
Observed tau
8 coarse ROIs
        |
        | define fine-to-coarse correspondence
        v
Higher-resolution connectivity
16 ROIs
        |
        v
Run SIR on the full 16-ROI connectome
        |
        v
Map / aggregate higher-resolution predictions
back to the 8 coarse observed ROIs
        |
        v
Evaluate in the observed tau space
```

This toy dataset tests separation between:

- **simulation space:** higher-resolution connectivity atlas;
- **evaluation space:** coarse observed tau atlas.

---

# Notes

- All values are synthetic.
- The datasets are intentionally small so that ROI ordering and mapping behavior can be inspected manually.
- These files are for unit/integration testing and debugging of the ROI-mapping workflow.
- They should not be used for biological interpretation or scientific analysis.
- For the higher-resolution HIP/AMY example, the data files contain the coarse and fine ROI definitions; the fine-to-coarse mapping behavior is handled by the simulation/evaluation code rather than being encoded as an additional mapping object in the generated connectivity file.
