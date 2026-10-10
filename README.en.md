# Mandia — Modular Continual Expansion

An exploratory study of modular legal-domain adaptation in a **17.13M-parameter French language model**, pretrained from random initialization. Author: **mandia**. This manuscript has not been peer reviewed or submitted to arXiv.

[Full French documentation](README.md) · [Manuscript](paper/article.fr.md) · [Numerical observations](results/evidence/) · [Reanalysis](docs/reproduction.md)

Five methods (full fine-tuning, residual MCE adapters with bottlenecks 256/64, separate LoRA banks with ranks 8/32), three learning-rate candidates, three final adaptation seeds, and both civil/penal training orders were evaluated. This produced 15 tuning runs and 30 final models on one legal glossary.

Frozen routes are preserved exactly for **both MCE and separate LoRA banks**. MCE is not uniformly better than LoRA: MCE 256 has lower penal loss, whereas LoRA r32 has lower civil loss. Full fine-tuning improves adaptation to the most recently trained domain but damages general-language loss. The closed-set term router reaches 83.2% accuracy on 155 legal terms; it is not an open-question or out-of-distribution router.

The repository contains model code, portable GPU experiment scripts, fixed protocols, aggregate tables, per-document numerical observations with hashed group labels, and analysis scripts. Corpus texts, tokenizer, model weights, and router vocabulary are not distributed. Statistical reanalysis can be performed without these inputs; exact retraining cannot.

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python scripts/analyser_extension.py results/evidence analysis-output
```

This is one core, one split, two related domains, and three adaptation seeds. Definition likelihood is not legal reasoning or factual accuracy. Independent multi-domain replication, matched dropout, additional continual-learning baselines, and open routing remain future work.

The upstream dataset card declares CC BY-NC-SA 4.0. No general reuse license is granted for this initial repository version; public access alone does not grant redistribution or commercial-use rights. OpenAI Codex assisted implementation, execution, analysis and manuscript preparation; scientific responsibility remains with the author.
