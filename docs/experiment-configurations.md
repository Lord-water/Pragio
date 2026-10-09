# Recorded training configurations

[The run registry](../protocols/training_runs.json) exposes 40 launchable configurations from the existing study. It contains runner paths and arguments, not selected checkpoints, scores or fitted calibrators. Configuration filenames retain their original run identifiers to preserve traceability.

| Run family | Purpose / implementation |
| :--- | :--- |
| `original_*_s42`, `A_repeat_*` | Separate heads, coupled heads and AN-LS; seeds 42, 2026, 3407 |
| `B_main_*_human_matched_*` | Human-only BCE, same human sampling and 6,006 optimizer updates |
| `B_main_*_epr_adapted_*` | EPR natural-label objective plus the same human BCE |
| `B_main_*_crisp_adapted_*`, `T_tune_bert_crisp_shift1_s42` | CRISP adaptation; BERT shift=1 effective configuration |
| `C_backbone_roberta_wwm_*` | Pinned Chinese RoBERTa backbone comparisons |
| `B_main_*_single_*` | Observed-choice-only objective |
| `D_weight_*` | Human-loss weight sensitivity |
| `E_diagnostic_*` | No preference, frozen encoder or no history controls |
| `F_targets_*` | Alternative AN-LS smoothing |

These include retrospective supplemental experiments. Publishing their settings or repeating them now does not turn them into new prospective confirmations. The early 240-update human-only diagnostic is not the matched control and is not included in this launcher.

## Protocol details

Configurations pin the backbone revision, optimizer settings, seeds, batch sizes and relevant method settings. Natural examples supply the corpus-choice objective; independently sampled human examples supply masked BCE. U judgments are masked. All data paths target reconstructed local inputs and the released training annotations.

The matched Human-only runner intentionally checks the frozen study dimensions (48 contexts, 549 known judgments) and checkpoint steps 2,002/4,004/6,006. It matches optimizer updates and human exposures, not FLOPs. It is not a generic arbitrary-size human-data trainer.

EPR recomputes its expected positive count using only training judgments: sum over candidates of `(A + 1) / (A + N + 2)`. This packaging change replaces a precomputed value with the same calculation; no test labels are consulted. Its objective uses the original mean-positive-loss / candidate-count reduction and unit expected-count penalty, plus human BCE.

CRISP estimates priors from the training population after warm-up and at subsequent epoch boundaries. It draws an additional positive example per candidate after warm-up; those added examples do not enter the population mean. Priors are created locally, not shipped. These are task adaptations to sentence-final acceptability, not reproductions of the original methods' image-classification experiments.

## Development selection

Use `scripts/select_and_calibrate.py` with raw predictions for candidate checkpoints, supplied in chronological checkpoint order. It selects the highest macro within-label AUROC on the 50 human development contexts, breaks ties with lower macro Brier, then with supplied order. It rejects predictions from outside the corpus development partition.

The fixed five context folds are in [development_folds.json](../protocols/development_folds.json). Shared calibration compares raw, temperature and affine-sigmoid mappings using pooled out-of-fold macro Brier. Candidate calibration compares the selected shared mapping with shrinkage strengths 0.1 and 1.0. The chosen mappings are refitted on all 50 development contexts. Hold both the checkpoint and calibration fixed before test scoring.

This release includes neither the originally selected checkpoints nor their fitted parameters or scores. Hardware and library differences can affect reruns. CPU checks verify implementation behavior; reproducing empirical performance still requires full training, source data access and appropriate evaluation.
