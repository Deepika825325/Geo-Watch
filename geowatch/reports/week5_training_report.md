# GeoWatch Week 5 Training Report

## 1. Executive Summary

Week 5 completed the GeoWatch model-training, convergence, probability-threshold selection and controlled loss-ablation workflow on the OSCD benchmark.

The final Dice+Focal model achieved:

- Validation precision: 0.425278
- Validation recall: 0.421346
- Validation F1: 0.423303
- Validation IoU: 0.268475
- Selected probability threshold: 0.76
- Best checkpoint epoch: 24

The model exceeded the frozen Otsu baseline by 0.234245 F1 and 0.164077 IoU.

A controlled plain-BCE ablation achieved a diagnostic validation F1 of 0.323598 and IoU of 0.193031 after validation-only threshold tuning. Dice+Focal therefore improved F1 by 0.099705 and IoU by 0.075443 over the tuned BCE diagnostic result.

Official OSCD test regions and test labels remained sealed throughout Week 5.

## 2. Evaluation Protocol

| Field | Value |
|---|---|
| Benchmark | OSCD |
| Training regions | 11 |
| Validation regions | Hong Kong, Mumbai and Paris |
| Training patches | 125 |
| Validation patches | 25 |
| Patch size | 256 × 256 |
| Input bands | B02, B03, B04 and B08 |
| Positive class | Change |
| Primary metrics | Change-class F1 and IoU |
| Secondary metrics | Precision, recall and accuracy |
| Validation pixels | 1,638,400 |
| Positive validation pixels | 53,761 |
| Positive-pixel prevalence | 3.281311% |
| Official test regions accessed | False |
| Official test labels accessed | False |

Training, checkpoint selection, threshold selection and ablation analysis used only the frozen training and validation regions.

No official test image, test region or test label was used to select a checkpoint, threshold, loss or hyperparameter.

## 3. Main Dice+Focal Experiment

### 3.1 Configuration

| Field | Value |
|---|---|
| Experiment | `week5_full_dice_focal` |
| Architecture | Weight-shared Siamese ResNet-18 U-Net |
| Temporal fusion | Absolute feature difference |
| Trainable parameters | 13,819,681 |
| Loss | Dice + Focal |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Scheduler | Cosine annealing |
| Batch size | 16 |
| Maximum epochs | 50 |
| Early-stopping monitor | Validation F1 |
| Early-stopping patience | 10 |
| Mixed precision | CUDA AMP |
| Experiment tracking | W&B offline mode |

### 3.2 Convergence

Training proceeded through checkpoint-safe resume stages and stopped at epoch 34 after early stopping reached 10 consecutive non-improving epochs.

| Field | Value |
|---|---:|
| Final training epoch | 34 |
| Best checkpoint epoch | 24 |
| F1 at best epoch using threshold 0.50 | 0.405815 |
| IoU at best epoch using threshold 0.50 | 0.254559 |
| Final-epoch validation F1 | 0.394062 |
| Final-epoch validation IoU | 0.245378 |
| Peak observed GPU memory | 4,827 MB |
| Peak observed GPU-memory utilization | 19.65% |

The best checkpoint was frozen as:

`experiments/run_full/checkpoints/best_model_epoch24.pt`

Frozen checkpoint SHA-256:

`61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94`

## 4. Main Threshold Search

Threshold selection was performed only on Hong Kong, Mumbai and Paris using the frozen epoch-24 checkpoint.

The search evaluated 91 thresholds from 0.05 to 0.95 with a step size of 0.01. The selection objective was validation change-class F1.

| Metric | Threshold 0.50 | Selected threshold 0.76 | Difference |
|---|---:|---:|---:|
| Precision | 0.336001 | 0.425278 | +0.089277 |
| Recall | 0.512249 | 0.421346 | -0.090902 |
| F1 | 0.405815 | 0.423303 | +0.017488 |
| IoU | 0.254559 | 0.268475 | +0.013915 |
| Accuracy | 0.950779 | 0.962328 | +0.011550 |

The higher threshold reduced false positives and produced a nearly balanced precision-recall operating point.

The selected threshold for the main model is therefore frozen at:

`0.76`

Threshold-search artifacts:

- `experiments/run_full/threshold_search/threshold_metrics.csv`
- `experiments/run_full/threshold_search/threshold_search_summary.json`

## 5. Frozen Otsu Baseline Comparison

| Method | Threshold | F1 | IoU |
|---|---:|---:|---:|
| Frozen Otsu baseline | Data-driven Otsu | 0.189058 | 0.104398 |
| Dice+Focal Siamese U-Net | 0.76 | 0.423303 | 0.268475 |

The Dice+Focal model improved over the frozen Otsu baseline by:

- Absolute F1 improvement: 0.234245
- Absolute IoU improvement: 0.164077
- F1 multiplier: 2.239×
- IoU multiplier: 2.572×

## 6. Controlled Plain-BCE Ablation

### 6.1 Fairness Contract

The BCE experiment retained the following main-experiment settings:

- Same architecture
- Same pretrained encoder initialization procedure
- Same seed: 42
- Same 11 training regions
- Same 3 validation regions
- Same patch construction
- Same augmentations
- Same batch size: 16
- Same optimizer
- Same learning rate
- Same scheduler
- Same maximum epoch limit
- Same early-stopping rule
- Same training metric threshold of 0.50

The only controlled objective change was:

`Dice+Focal → unweighted BCEWithLogitsLoss`

The contract is stored at:

`experiments/run_ablation_bce/ablation_contract.json`

### 6.2 BCE Training Behaviour

Plain BCE stopped at epoch 11 because validation F1 remained zero at threshold 0.50 for 10 consecutive non-improving epochs.

Validation accuracy remained 0.967187 because the model predicted every validation pixel as unchanged at threshold 0.50. This accuracy is misleading because unchanged pixels constitute approximately 96.72% of the validation data.

The BCE loss nevertheless decreased:

| Field | Value |
|---|---:|
| Epoch-1 validation BCE | 0.326475 |
| Epoch-11 validation BCE | 0.220111 |
| F1 at threshold 0.50 | 0.000000 |
| IoU at threshold 0.50 | 0.000000 |
| Final epoch | 11 |
| Early stopping triggered | True |
| Peak observed GPU memory | 4,742 MB |

This indicates severe probability miscalibration and majority-background dominance rather than complete absence of learned ranking information.

### 6.3 BCE Checkpoint Selection Disclosure

The checkpoint selected by the frozen training protocol was epoch 1 because all epochs tied at F1 0.0 using threshold 0.50.

Its validation-only threshold search produced:

| Field | Value |
|---|---:|
| Checkpoint epoch | 1 |
| Selected threshold | 0.20 |
| Precision | 0.032813 |
| Recall | 1.000000 |
| F1 | 0.063542 |
| IoU | 0.032813 |

Epoch 11 was analyzed separately as a diagnostic checkpoint because it had the lowest observed validation BCE and represented the final learned model before early stopping.

Its validation-only threshold search produced:

| Field | Value |
|---|---:|
| Diagnostic checkpoint epoch | 11 |
| Selected threshold | 0.24 |
| Precision | 0.261632 |
| Recall | 0.424025 |
| F1 | 0.323598 |
| IoU | 0.193031 |

The epoch-11 result is reported as a diagnostic comparison, not as a silent replacement for the protocol-selected epoch-1 checkpoint.

The selected diagnostic checkpoint is frozen locally as:

`experiments/run_ablation_bce/checkpoints/selected_model_epoch11.pt`

Frozen checkpoint SHA-256:

`6a98a20e9d30510d7451aa6130dab79a4ba13d4ce364c69e19d49828b01c5aaa`

## 7. Final Ablation Comparison

| Experiment | Loss | Epoch | Threshold | Precision | Recall | F1 | IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| Frozen Otsu baseline | Otsu | — | — | — | — | 0.189058 | 0.104398 |
| Main model | Dice+Focal | 24 | 0.76 | 0.425278 | 0.421346 | 0.423303 | 0.268475 |
| BCE protocol checkpoint | Plain BCE | 1 | 0.20 | 0.032813 | 1.000000 | 0.063542 | 0.032813 |
| BCE diagnostic checkpoint | Plain BCE | 11 | 0.24 | 0.261632 | 0.424025 | 0.323598 | 0.193031 |

Compared with the tuned BCE diagnostic checkpoint, Dice+Focal achieved:

| Metric | Dice+Focal minus BCE |
|---|---:|
| Precision | +0.163646 |
| Recall | -0.002679 |
| F1 | +0.099705 |
| IoU | +0.075443 |

Dice+Focal delivered 1.308× the BCE diagnostic F1.

Recall was nearly identical between the two models. The primary improvement came from substantially higher precision, demonstrating that Dice+Focal controlled false positives more effectively under severe foreground-background imbalance.

## 8. Engineering Outcomes

Week 5 also validated the following production-style training capabilities:

- CUDA mixed-precision training
- Deterministic experiment configuration
- Full optimizer and scheduler checkpointing
- CUDA GradScaler checkpointing
- CPU and CUDA RNG-state restoration
- CUDA-safe checkpoint resume
- Early stopping
- Best and last checkpoint management
- Validation prediction logging
- Offline W&B experiment tracking
- GPU utilization monitoring
- Validation-only threshold optimization
- Machine-readable ablation contracts
- Machine-readable comparison artifacts
- Unit and regression tests

W&B was intentionally run in offline mode. Local run directories are retained under the corresponding experiment log directories. No cloud report link is claimed in this report.

## 9. Limitations

- Only three geographically separated validation regions were available.
- Validation threshold tuning may not transfer perfectly to unseen official test regions.
- The official test split remains unmeasured until Week 6.
- The training set contains only 125 patches.
- Changed pixels represent approximately 3.28% of validation pixels.
- The architecture uses four Sentinel-2 bands and absolute-difference fusion.
- BCE checkpoint selection exposed a mismatch between the fixed 0.50 monitoring threshold and the model's probability calibration.
- The BCE epoch-11 result is diagnostic because it was selected using validation BCE after the frozen F1-based early-stopping process had selected epoch 1.

## 10. Week 6 Handoff

Week 6 should:

1. Freeze the Dice+Focal epoch-24 checkpoint and threshold 0.76.
2. Keep all model and threshold decisions unchanged.
3. Open the official test split only once for final evaluation.
4. Report positive-class precision, recall, F1 and IoU.
5. Compare the final neural model with the frozen Otsu baseline.
6. Generate per-region and qualitative change-map results.
7. Document failure cases without modifying Week 5 hyperparameters.

## 11. Artifact Index

### Main experiment

- `experiments/run_full/train_config.yaml`
- `experiments/run_full/threshold_search/threshold_metrics.csv`
- `experiments/run_full/threshold_search/threshold_search_summary.json`
- `experiments/run_full/checkpoints/best_model_epoch24.pt`

### BCE ablation

- `experiments/run_ablation_bce/train_config.yaml`
- `experiments/run_ablation_bce/ablation_contract.json`
- `experiments/run_ablation_bce/threshold_search_epoch1/`
- `experiments/run_ablation_bce/threshold_search_epoch11/`
- `experiments/run_ablation_bce/threshold_search/`
- `experiments/run_ablation_bce/checkpoints/selected_model_epoch11.pt`

### Final comparison

- `experiments/week5_comparison/ablation_comparison.json`
- `experiments/week5_comparison/ablation_comparison.csv`

## 12. Completion Statement

The Week 5 technical scope is complete:

- Main model trained to convergence
- Best checkpoint frozen
- Validation threshold selected
- Plain-BCE ablation completed
- BCE calibration behaviour diagnosed
- Ablation comparison produced
- Official test regions and labels kept sealed
- Final report populated

The repository release commit and replacement Week 5 completion tag will be created only after the final acceptance audit passes. The existing `week5-complete` tag will not be moved.
