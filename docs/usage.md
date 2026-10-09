# Training, inference and calibration

Run commands from the repository root using Python 3.10 or newer. Recorded configurations and final human labels are included; raw source dialogue, checkpoints and fitted parameters are not. Obtain and reconstruct the source inputs before training. `configs/example.json` remains a separate, generic interface example.

## 1. Install

```bash
python -m pip install -r requirements-training.txt
python src/check_method_core.py
python src/check_training_interface.py
python src/experiments/check_epr.py
python src/experiments/check_crisp.py
```

The checks use synthetic tensors and a tiny randomly initialized BERT. They need no pretrained download and establish implementation properties, not model performance. For calibration and scoring only, use `requirements-analysis.txt` instead.

## 2. Prepare local inputs

For the released study, reconstruct the fixed inputs as described in [data access](data-access.md):

```bash
python scripts/restore_lccc.py --source /path/to/LCCC-large.zip
python scripts/validate_release.py
```

For custom data instead, follow [the input format](data-format.md) and edit a copy of `configs/example.json`. The mixed-batch runner requires some known human training judgments. Its `single` comparison ignores the human loss but still uses that schedule; it is not a standalone no-annotation trainer.

Supply a local BERT-compatible encoder and tokenizer, or download the example backbone:

```bash
python scripts/download_backbone.py --config configs/original_independent_s42.json
```

If using a different backbone, change its repository, pinned revision and local path together. The downloader and model loading use the original model provider's terms.

## 3. Train

```bash
python scripts/train.py --run original_independent_s42 --dry-run
python scripts/train.py --run original_independent_s42
```

Full training requires CUDA with bfloat16 support in this runner. Set memory limits and batch sizes for the available GPU. Each run directory must be new. A positive `--max-steps` enables a short CPU/GPU debugging run; it is not a full training run.

The `independent` mode is the separate-head method. Other modes (`coupled`, `no_preference`, `an_ls`, `single`) are comparison implementations. Select checkpoints with independent development judgments. The runner's corpus-choice accuracy and human **training** fit are diagnostics, not evidence for human acceptability generalization. Freeze selection and calibration before test evaluation.

EPR, CRISP and the matched Human-only control have dedicated runners exposed through the same launcher. Run `python scripts/train.py --help` for available IDs and read [configuration notes](experiment-configurations.md) for the comparison families. For a generic custom configuration, invoke `src/train_context_model.py` directly with `--config`, `--mode` and `--run-dir`.

## 4. Predict

Use an actual saved epoch checkpoint and supply the same local backbone path recorded during training:

```bash
python scripts/predict.py --checkpoint outputs/runs/original_independent_s42/epoch1.pt --inputs data/local/dev_ready.jsonl --output outputs/dev_raw.json
```

The command performs CPU float32 inference and refuses to overwrite its output. `epoch1.pt` is an interface example, not a recommended selected epoch. Only load trusted local checkpoints. Raw sigmoid scores are not automatically calibrated.

## 5. Fit and apply calibration

For the released protocol, the training run writes raw development predictions for each completed epoch. Supply them in checkpoint order:

```bash
python scripts/select_and_calibrate.py --predictions outputs/runs/original_independent_s42/epoch1_dev_predictions.json outputs/runs/original_independent_s42/epoch2_dev_predictions.json outputs/runs/original_independent_s42/epoch3_dev_predictions.json --output outputs/selection.json
```

Read `selected_checkpoint` in `selection.json`: it identifies the selected checkpoint's **prediction file**. Use the corresponding `.pt` checkpoint for inference. For example, `epoch2_dev_predictions.json` corresponds to `epoch2.pt`; that name is an illustration, not a preselected result. Generate raw predictions on `data/local/heldout_ready.jsonl`, save them as `outputs/heldout_raw.json`, and apply the frozen bundle:

```bash
python scripts/calibrate.py apply --predictions outputs/heldout_raw.json --parameters outputs/selection.json --family shared --output outputs/heldout_calibrated.json
python scripts/evaluate.py --predictions outputs/heldout_calibrated.json --annotations data/labels/confirmation.json --output outputs/confirmation_metrics.json
```

Use `--family candidate` for the separately selected candidate-calibration comparison, and `data/labels/first_test.json` for the other evaluation batch. The application command does not enforce checkpoint identity: supply predictions from the checkpoint identified by selection. Do not select between calibration families on test performance.

For a custom development set and a preselected calibration family:

```bash
python scripts/calibrate.py fit --dev-predictions outputs/dev_raw.json --dev-annotations data/labels/dev.json --dev-records data/local/dev_ready.jsonl --kind affine_sigmoid --output outputs/custom_calibration.json
python scripts/calibrate.py apply --predictions outputs/dev_raw.json --parameters outputs/custom_calibration.json --output outputs/custom_dev_calibrated.json
```

Available families are `raw`, `temperature`, `affine_sigmoid` and `candidate_shrinkage`. Candidate shrinkage additionally accepts a positive `--lambda`; tune it on development data only. It uses candidate offsets only when both positive and negative development counts reach five, with zero offsets otherwise. Never fit on test judgments. The split check verifies supplied metadata; the user remains responsible for truthful partitioning and source separation.

After fixing all development choices, obtain scores for held-out inputs and apply the same saved calibrator. Do not refit it. Development fit scores are not held-out estimates.

## 6. Evaluate supplied labels

```bash
python scripts/evaluate.py --predictions outputs/dev_calibrated.json --annotations data/labels/dev.json --output outputs/dev_metrics.json
```

The evaluator masks `U` consistently and reports candidate-wise Brier/BCE/AUROC and ranking/coverage summaries. These scripts generate new outputs locally; no experimental outputs are distributed in this repository.

## Repository map

| Path | Purpose |
| :--- | :--- |
| `src/` | Model interface, objectives, training, metrics and calibration routines |
| `scripts/` | Backbone download, prediction, calibration and evaluation entry points |
| `configs/example.json` | Editable local-input configuration |
| `configs/`, `protocols/training_runs.json` | Recorded run settings and launcher registry |
| `data/labels/`, `data/index/` | Final judgments and text-free fixed corpus indices |
| `docs/` | Input schemas and usage |
| `assets/` | Original README banner and method overview |

`data/local/`, `models/`, `outputs/`, `results/`, `predictions/` and checkpoints are ignored by Git. Only the explicit label/index release and source manifest are included under `data/`. Keep private material outside the release tree; `.gitignore` does not erase files already committed.
