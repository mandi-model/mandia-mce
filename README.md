# Mandia MCE

Experimental study of modular adaptation in a small French language model.

**English** | [Français](README.fr.md)

Author: **mandia**. Study completed on 10 October 2026. Manuscript version 1.0. Not peer reviewed.

| Manuscript | Source | PDF |
|---|---|---|
| English | [Full article](paper/article.en.md) | [Download](paper/article.en.pdf) |
| Français | [Article complet](paper/article.fr.md) | [Télécharger](paper/article.fr.pdf) |

## Research question

We examine whether independent domain modules can adapt a small language model while preserving previously frozen computational routes. The study compares residual Modular Continual Expansion (MCE) adapters, separate LoRA banks, and full fine-tuning on civil and penal definitions from one French legal glossary.

The shared Transformer has **17,129,280 parameters** and was pretrained from random initialization on **14,753,792 observed training tokens**. No pretrained IvoireSLM weights were used. Five methods are evaluated with the same three-candidate learning-rate search, followed by three adaptation seeds and both domain orders: **15 tuning runs and 30 final models**. Each domain receives 300 updates, batch size 4, and context length 256.

## Main results

Held-out test cross-entropy in nats per token; lower is better. Means across three adaptation seeds, civil-then-penal order, with the domain route supplied for each independent bank:

| Method | Trainable parameters per domain | General | Civil | Penal |
|---|---:|---:|---:|---:|
| Full fine-tuning | 17,129,280 | 5.772 | 3.920 | 3.442 |
| MCE 256 | 1,969,920 | 4.414 | 3.685 | 3.716 |
| MCE 64 | 495,360 | 4.414 | 3.780 | 3.710 |
| LoRA r8 | 516,096 | 4.414 | 3.782 | 3.736 |
| LoRA r32 | 2,064,384 | 4.414 | 3.649 | 3.788 |

![Held-out test losses in both domain orders](results/figures/test_results.en.png)

MCE and independent LoRA banks preserve their frozen routes exactly in this experiment. At nearby capacities, LoRA r32 has lower civil loss than MCE 256, while MCE 256 has lower penal loss. Full fine-tuning adapts strongly to the final domain but increases general-language loss. These results do not establish a universally superior method. The full manuscript reports both orders, standard deviations, paired descriptive intervals, training costs, and preservation checks.

A closed-set term router achieves **83.2% accuracy** and **0.791 macro-F1** on 155 legal terms, compared with **67.7%** majority-class accuracy. Penal recall is **60%**. Predicted-route definition losses are evaluated separately. The router is not evaluated on free-form questions or out-of-distribution detection.

## Reanalysis

The release contains numerical observations without corpus text, term vocabulary, tokenizer, or model weights. Term-group labels are replaced by stable hashes. The statistical calculations can be reproduced without a GPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python scripts/analyser_extension.py results/evidence analysis-output
python scripts/analyser_routeur.py
python scripts/figures_english.py analysis-output
```

[Colab reanalysis notebook](https://colab.research.google.com/github/mandi-model/mandia-mce/blob/main/notebooks/reanalysis.ipynb)

The tables and paired intervals were independently recomputed from the release at absolute tolerance 1e-10. Router metrics and group intervals were also reproduced. A CPU check performs actual updates on miniature random MCE/LoRA models and verifies frozen weights, gradients, previous-route logits, and checkpoint reconstruction:

```bash
python -m pip install -r requirements.txt
python scripts/smoke_model.py
```

Exact retraining requires archived corpus inputs and the pretrained core, which are not distributed. The portable GPU runner is syntax-checked and derived from the executed historical runner; it was not rerun for this release. See [reproduction instructions](docs/reproduction.en.md) and the [validation report](docs/validation.en.md).

## Scope and limitations

This is an exploratory study with one weakly pretrained core, one split, one glossary, two related domains, and three adaptation seeds. Dropout differs between full and frozen-core methods. Parameter counts are close; FLOPs are not equalized. The clustered bootstrap is conditional on this core and split. Definition likelihood does not establish legal reasoning or factual accuracy. Preserving a fixed route does not guarantee correct expert selection.

Frozen-path preservation has precedents in [progressive networks](https://arxiv.org/abs/1606.04671), [NLP adapters](https://proceedings.mlr.press/v97/houlsby19a.html), and [LoRA](https://arxiv.org/abs/2106.09685). Architectural novelty is not claimed. Expert cooperation, fusion, broader multi-corpus evaluation, replay baselines, and open routing remain future experiments. [Architecture](docs/architecture.en.md).

## Repository contents

| Location | Contents |
|---|---|
| Root Python modules | Causal Transformer, residual adapters, separate LoRA banks, checkpoint loader, router inference |
| `experiments/` | Portable GPU extension and routing evaluation |
| `scripts/` | Statistical reanalysis, English figures, integrity checks, CPU verification, PDF rendering |
| `results/evidence/` | Numerical observations, controls, and provenance |
| `results/figures/` | Figures in French and English |
| `paper/` | Complete manuscripts in French and English, Markdown and PDF |
| `docs/` | Architecture, protocols, reproduction, validation, data provenance, and arXiv preparation |

## Provenance and citation

Legal data originate from [bernard-ng/french-legal-glossary](https://huggingface.co/datasets/bernard-ng/french-legal-glossary), revision `f8c850c424af7a60ee55cf9eaf050cf05b59aba7`. The dataset card declares **CC BY-NC-SA 4.0**; underlying-text rights and legal accuracy were not independently verified. Corpus texts and weights are excluded. Public repository access does not grant a general redistribution or commercial-use license. [Data and rights](docs/data.md).

Suggested citation: **mandia (2026). Modular Expansion of a Small French Language Model: Legal Adaptation, Domain Order, and Term Routing. Version 1.0.** The French and English manuscripts are language versions of the same study. No DOI or arXiv identifier has been assigned. [arXiv preparation](docs/arxiv.en.md).

OpenAI Codex assisted implementation, experiment execution, analysis, and manuscript preparation. Scientific interpretation, review, and author responsibility remain human. This disclosure documents the workflow; it is not a claim of independent scientific validation.
