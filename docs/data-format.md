# Input format

Supply UTF-8 JSONL records and JSON annotations under the ignored `data/` directory. The examples below are invented schema illustrations, not study samples or human judgments.

## Dialogue records

```json
{"record_id":"example-train-01","split":"train","group_id":"dialogue-01","body_key":"unique-target-01","history":["现在出发？"],"target_body":"我们一起去","target_punctuation":"。","observed_particle":"吧"}
```

- `history`: a list of earlier utterances.
- `target_body`: the target with the candidate sentence-final particle removed.
- `target_punctuation`: trailing punctuation, separately stored.
- `observed_particle`: one of the twelve supported particles below.
- `split`: `train` or `dev` for the training runner.
- `record_id`: unique within and across partitions.
- `group_id` and `body_key`: stable dialogue and deduplication keys. Training and development sets must not overlap on either key; assign these based on actual source grouping, not arbitrary split-specific names.

The BERT-style tokenizer receives history as the first sequence and target + mask + punctuation as the second. Only the history may be truncated. A target too long to fit must be filtered or otherwise handled before training.

## Candidate order

All score vectors and label tensors use exactly:

```text
吗, 呢, 吧, 嘛, 呗, 罢了, 啊, 哦, 呀, 啦, 么, 咯
```

## Judgment mappings

Training annotations map record IDs to candidate labels. Omitted candidates are masked, as are explicit `U` labels. For development calibration and evaluation, provide all twelve candidates explicitly as `A`, `N` or `U`.

```json
{"example-train-01":{"吧":"A","吗":"N","呢":"U"}}
```

`A` means acceptable, `N` unacceptable, and `U` uncertain/unrated. Resolve independent annotator disagreements using a defined annotation protocol before providing this mapping. A reviewed `N` judgment for the observed particle disables the conflicting corpus-choice target for that training example.

## Prediction mappings

`scripts/predict.py` writes a JSON object from record ID to twelve probabilities in the order above. Calibration accepts this mapping or the training runner's list of `{record_id, acceptability, choice}` objects. Outputs from untrained models are not meaningful acceptability estimates.
