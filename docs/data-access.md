# Data access and release scope

Pragio releases **498 final human-annotated contexts**, represented by text-free record IDs and twelve A/N/U judgments per context. Source dialogue text must be obtained separately. The labels are research annotations, not model predictions or experimental scores.

| Annotation partition | Contexts | Corpus partition | Intended use |
| :--- | ---: | :--- | :--- |
| `train.json` | 48 | train | Human supervision |
| `dev.json` | 50 | dev | Checkpoint and calibration selection |
| `first_test.json` | 200 | heldout | Initial evaluation |
| `confirmation.json` | 200 | heldout | Later evaluation with clarified annotation guidance |

Each context has twelve explicit judgments, including uncertainty (`U`): **5,976 judgments** in total. The four ID sets are disjoint. The two evaluation batches have distinct histories and annotation guidance; do not pool them as interchangeable independent replications. See [annotation guidance](annotation-guidelines.md).

## Reconstruct the dialogue inputs

Obtain **LCCC-large** through the [original CDial-GPT project](https://github.com/thu-coai/CDial-GPT), subject to the source's applicable terms. Use the `LCCD.json` member of that archive. [The source manifest](../data/source_manifest.json) records the archive SHA-256, source version and retained counts.

```bash
python scripts/restore_lccc.py --source /path/to/LCCC-large.zip
```

An extracted `LCCD.json` is also accepted. The script reads the archive incrementally, restores only the indexed rows, verifies each row's text-field SHA-256, and writes:

```text
data/local/train_ready.jsonl    32,021 rows
data/local/dev_ready.jsonl       4,026 rows
data/local/heldout_ready.jsonl   3,916 rows
```

The released [index](../data/index) fixes the original 39,963 retained records and their order. It records source member, dialogue/turn offsets, grouping keys, observed particle and a text-field hash. It does not contain dialogue strings. Do not resample or reassign partitions when reproducing this setup. Reconstruction rejects text mismatches and existing output files. A matching ZIP filename alone does not establish a matching source version.

Annotation files map `record_id` to judgments. Join them to restored records by this ID. The 498 annotated contexts are a subset of the corpus partitions, not 498 additional natural training examples. Held-out text and labels must not enter training, tuning, or calibration.

## Third-party scope

The LCCC repository's code license does not, by itself, settle redistribution rights for all incorporated third-party dialogue sources. This release therefore distributes our annotations and fixed indices, with reconstruction from a separately acquired source, rather than repackaging the raw corpus. Local reconstructed text remains ignored by Git.

NaturalConv-based external-context and output-evaluation materials are **not included**: no X100/Y100 dialogue text, record-level annotations or continuations. Consult the [author-hosted NaturalConv dataset](https://huggingface.co/datasets/xywang1/NaturalConv) and its [license](https://huggingface.co/datasets/xywang1/NaturalConv/blob/main/LICENSE) for permitted access and use. This release does not grant permission to redistribute those materials.

Paper files, empirical model scores, per-record predictions, model checkpoints and fitted calibration parameters are also excluded. Releasing annotations and configurations does not make this a complete archive of every experiment. See [NOTICE](../NOTICE.md) for this project's research-use terms.
