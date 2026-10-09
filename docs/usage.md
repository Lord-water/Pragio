# Training, inference and calibration

Run commands from the repository root using Python 3.10 or newer. The example configuration illustrates the interface; it is not a released experiment configuration or a universal setting. This release does not include data, checkpoints or fitted parameters and cannot reproduce the paper's scores by itself.

## 1. Install

```bash
python -m pip install -r requirements-training.txt
python src/check_method_core.py
python src/check_training_interface.py
```

The checks use synthetic tensors and a tiny randomly initialized BERT. They need no pretrained download and establish implementation properties, not model performance. For calibration and scoring only, use `requirements-analysis.txt` instead.

## 2. Prepare local inputs

Create train/development records and training judgments using [the input format](data-format.md). Copy or edit `configs/example.json` to point to them. The runner requires some known human training judgments for its mixed-batch schedule. Its `single` comparison ignores the human loss but still uses that schedule; it is not a standalone no-annotation trainer.

Supply a local BERT-compatible encoder and tokenizer, or download the example backbone:

```bash
python scripts/download_backbone.py --config configs/example.json
```

If using a different backbone, change its repository, pinned revision and local path together. The downloader and model loading use the original model provider's terms.

## 3. Train

```bash
python src/train_context_model.py --config configs/example.json --mode independent --run-dir outputs/independent
```

Full training requires CUDA with bfloat16 support in this runner. Set memory limits and batch sizes for the available GPU. Each run directory must be new. A positive `--max-steps` enables a short CPU/GPU debugging run; it is not a full training run.

The `independent` mode is the separate-head method. Other modes (`coupled`, `no_preference`, `an_ls`, `single`) are comparison implementations. Select checkpoints with independent development judgments. The runner's corpus-choice accuracy and human **training** fit are diagnostics, not evidence for human acceptability generalization. Freeze selection and calibration before test evaluation.

## 4. Predict

Use an actual saved epoch checkpoint and supply the same local backbone path recorded during training:

```bash
python scripts/predict.py --checkpoint outputs/independent/epoch1.pt --inputs data/dev.jsonl --output outputs/dev_raw.json
```

The command performs CPU float32 inference and refuses to overwrite its output. `epoch1.pt` is an interface example, not a recommended selected epoch. Only load trusted local checkpoints. Raw sigmoid scores are not automatically calibrated.

## 5. Fit and apply calibration

```bash
python scripts/calibrate.py fit --dev-predictions outputs/dev_raw.json --dev-annotations data/dev_annotations.json --dev-records data/dev.jsonl --kind affine_sigmoid --output outputs/calibration.json
python scripts/calibrate.py apply --predictions outputs/dev_raw.json --parameters outputs/calibration.json --output outputs/dev_calibrated.json
```

Available families are `raw`, `temperature`, `affine_sigmoid` and `candidate_shrinkage`. Candidate shrinkage additionally accepts a positive `--lambda`; tune it on development data only. It uses candidate offsets only when both positive and negative development counts reach five, with zero offsets otherwise. Never fit on test judgments. The split check verifies supplied metadata; the user remains responsible for truthful partitioning and source separation.

After fixing all development choices, obtain scores for held-out inputs and apply the same saved calibrator. Do not refit it. Development fit scores are not held-out estimates.

## 6. Evaluate supplied labels

```bash
python scripts/evaluate.py --predictions outputs/dev_calibrated.json --annotations data/dev_annotations.json --output outputs/dev_metrics.json
```

The evaluator masks `U` consistently and reports candidate-wise Brier/BCE/AUROC and ranking/coverage summaries. These scripts generate new outputs locally; no experimental outputs are distributed in this repository.

## Repository map

| Path | Purpose |
| :--- | :--- |
| `src/` | Model interface, objectives, training, metrics and calibration routines |
| `scripts/` | Backbone download, prediction, calibration and evaluation entry points |
| `configs/example.json` | Editable local-input configuration |
| `docs/` | Input schemas and usage |
| `assets/` | Original README banner and method overview |

`data/`, `models/`, `outputs/`, `results/`, `predictions/` and checkpoints are ignored by Git. Keep private material outside the release tree; `.gitignore` does not erase files already committed.
