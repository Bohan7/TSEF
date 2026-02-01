# TSEF Attack Scripts

This folder contains shell scripts for reproducing TSEF attack experiments on different datasets.

## Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `run_lowvar.sh` | LowVarDetect | Synthetic dataset |
| `run_seqcombmv.sh` | SeqCombMV | Multivariate synthetic dataset |
| `run_seqcombsingle.sh` | SeqCombSingle | Univariate synthetic dataset |
| `run_mitecg.sh` | MITECG | Real-world ECG dataset |
| `run_pam.sh` | PAM | Real-world activity recognition dataset |
| `run_epilepsy.sh` | Epilepsy | Real-world EEG dataset |
| `run_all.sh` | All | Run all experiments sequentially |

## Usage

### Before Running

1. **Update model paths** in each script:
   - `MODEL_PATH`: Path to your trained explainer model
   - `PREDICTOR_PATH`: Path to your trained classifier model

2. **Update data paths** if your datasets are in a different location:
   - `DATA_PATH`: Root directory containing all datasets

### Run Single Dataset

```bash
cd scripts
bash run_lowvar.sh
```

### Run All Datasets

```bash
cd scripts
bash run_all.sh
```

## Parameters

Key parameters used in the scripts:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--exp_method` | ours / ig_batch | Explainer: ours (TimeX/TimeX++), ig_batch (Integrated Gradients) |
| `--attack` | 0.1 | Attack budget (10% of data range) |
| `--tsef_attack_iters` | 100 | Outer iterations |
| `--tsef_attack_k1` | 10 | Inner iterations for Mt optimization |
| `--tsef_attack_k2` | 20 | Inner iterations for Mf optimization |
| `--tsef_attack_lr_t` | 1.0 | Learning rate for temporal mask |
| `--tsef_attack_lr_f` | 1.0 | Learning rate for frequency mask |
| `--tsef_attack_r_mt` | 0.3 | Temporal mask sparsity ratio |
| `--tsef_attack_beta_t` | 1.0 | Weight for sparsity loss |
| `--tsef_attack_gamma_t` | 1.0 | Weight for connectivity loss |
| `--tsef_attack_lambda_cls` | varies | Weight for classification loss (dataset-specific) |
| `--tsef_attack_lambda_exp` | 1.0 | Weight for explanation loss |
| `--split_no` | -1 | Run on all 5 splits |
| `--is_timex` | flag | Use TimeX model (vs TimeX++) |

## Output

Results will be saved to:
```
./results/
```
