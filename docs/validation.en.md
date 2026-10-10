# Release Validation

[Français](validation.md)

The public observations were reanalyzed without a GPU. All numerical fields in the ten-row final-results table and eight paired comparisons match the archived CSV values within absolute tolerance **1e-10**. Router accuracy, macro-F1, confusion matrix, and clustered intervals match within **1e-12**. All 48 predicted-route evaluations have their NLL sums and target counts checked.

The CPU model check ran with Python 3.12 and PyTorch **2.14.1**. Two MCE bottlenecks and an independent LoRA bank undergo actual updates. New-expert parameters change, while previous weights and logits remain identical; gradients remain absent for frozen core and previous bank, and checkpoint reconstruction preserves logits. This miniature random-model check is distinct from the historical legal GPU experiment under PyTorch 2.11.0+cu130.

`mandi_model.py` and `mandi_mce_v1.py` are byte-identical to the sources executed in the archived experiment. Python scripts and notebook cells are syntax-checked. French and English manuscript tables carry identical numerical cells. PDF pages are rendered for visual review; author metadata are “mandia”. Private cloud links and local account paths are removed. Public observations contain no document-text, term, vocabulary, or email fields. Relative Markdown links and file fingerprints are checked.

The portable GPU runner is syntax-checked and derived from the executed source. It has not been rerun on a GPU for this release. The public Colab notebook has not been executed remotely for this release. Corpus inputs, tokenizer, and checkpoints are required for exact retraining and are not included.

Integrity checks and reanalysis establish consistency of the release, not external certification of the data or peer review of the manuscripts.
