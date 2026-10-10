# Reanalysis and Retraining

[Français](reproduction.md)

## Statistical reanalysis without a GPU

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python scripts/analyser_extension.py results/evidence analysis-output
python scripts/analyser_routeur.py
python scripts/figures_english.py analysis-output
```

The extension analysis recomputes both orders, means and sample standard deviations across three seeds, eight paired comparisons with 2,000 term-group bootstrap resamples, and the validation curves. Expected observation counts are 90 final evaluations, 945 curve points, 90 cost records, 15 tuning runs, and 30 reloads. Hashed group labels preserve grouping and record order. The bootstrap is descriptive and conditional on the observed core and split.

The router analysis recomputes accuracy, macro-F1, confusion matrix, grouped intervals, and the means of predicted-route losses, verifying them against recorded observations. Numerical CSV fields match the archived tables within absolute tolerance 1e-10; router metrics and intervals match within 1e-12. `figures_english.py` draws English figures from the same CSV and validation observations.

The [Colab notebook](https://colab.research.google.com/github/mandi-model/mandia-mce/blob/main/notebooks/reanalysis.ipynb) performs this reanalysis using the public repository. It requires no Drive account, corpus, checkpoint, or GPU.

## CPU model check

```bash
python -m pip install -r requirements.txt
python scripts/smoke_model.py
```

An actual update to a new expert on a miniature randomly initialized model checks absent gradients on the core and previous bank, unchanged frozen weights and previous-route logits, and checkpoint reconstruction. Both residual MCE and independent LoRA are covered. This does not evaluate legal capabilities or rerun the GPU study.

## Historical GPU protocol

The experiment used PyTorch 2.11.0+cu130, a Tesla T4, bfloat16 autocast, and float32 weights. Native hardware BF16 was not reported as supported. The router used scikit-learn 1.6.1. Requirements specify working dependencies, not a bitwise-reproduction guarantee for the Colab environment.

The scripts under `experiments/` preserve the experimental logic while replacing cloud mounting and private paths with environment variables. Their syntax was checked; no new GPU training was performed for this release.

Three directories, **not included in the release**, are required:

- `MCE_INPUT_DIR`: `manifest.json`; `config.json` with `general_validation_starts`; `mandi_model.py`; `mandi_mce_v1.py`; `tokenizer.json`; `documents.jsonl`; `general_validation.bin`; and `civil_{train,validation,test}.bin` / `penal_{train,validation,test}.bin`.
- `MCE_CORE_DIR`: `latest.pt`, containing the pretrained core's `model` and `config` keys.
- `MCE_CORPUS_DIR`: general-language `train.jsonl` for the overlap audit, `test.bin`, and `manifest.json`.

Token streams are uint16 little-endian. Legal document records contain `hash`, `group`, `domain`, `split`, and `text`, where text is `term\ndefinition`. General-language training records contain `hash` and `text`. Input manifest entries contain `sha256` and `bytes`. The tokenizer recognizes `<eos>`.

The required core SHA-256 is `38233f63ee915a882ee49f0bbf79b280389101ab46f2d09d27ad2f08559e43ab`; the tokenizer SHA-256 is `07fd856fa6e63c8e8594fda5b5d4317a11e906801c049066cd851eda328fb77d`. The runner rejects a different core. A new-core experiment requires explicit documentation that its results differ from those reported here.

```bash
export MCE_INPUT_DIR=/path/to/prepared-legal-inputs
export MCE_CORE_DIR=/path/to/archived-core
export MCE_CORPUS_DIR=/path/to/prepared-general-corpus
export MCE_OUTPUT_DIR=/path/to/new-runs
python experiments/run_all.py
```

Each run creates new output and archive directories without overwriting existing runs. The script checks input hashes, audits lexical overlap, selects learning rates on validation, then evaluates all final models. Router selection uses validation; predicted-route evaluation follows. The main extension is limited to 3,600 seconds, with routing additional.

The dataset URL alone does not reconstruct the original core or split. Missing inputs must not be replaced with simulated observations to obtain apparently successful runs. Reanalysis of recorded numerical results is distinct from independent experimental replication.

## Inference with separately supplied checkpoints

`charger_etude_mce.charger_pilote` checks fingerprints and keys, reconstructs the model, freezes weights, and sets evaluation mode. Its inherited function name supports all five extension methods. `prediction_pilote` accepts `civil`, `penal`, or `None` for the general route.

`RouteurTermesMCE` reconstructs TF-IDF and logistic decisions from JSON and accepts one supplied legal term per line. Router weights and vocabulary are not distributed. It selects civil/penal only, without general or out-of-distribution detection.
