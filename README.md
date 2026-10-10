# Mandia — Modular Continual Expansion (MCE)

**Une étude exploratoire sur l’adaptation modulaire d’un petit modèle français.**

Auteur : **mandia** · expérience du 10 octobre 2026 · manuscrit non évalué par les pairs.

[English overview](README.en.md) · [Article complet](paper/article.fr.md) · [Article PDF](paper/article.fr.pdf) · [Architecture](docs/architecture.md) · [Reproduction](docs/reproduction.md) · [Préparation arXiv](docs/arxiv.md) · [Validation](docs/validation.md)

## La question

Peut-on ajouter un domaine à un modèle de langage en entraînant des modules spécialisés, tout en conservant les chemins déjà appris ? MCE conserve un cœur partagé gelé et ajoute une banque de petits modules par domaine. L’expérience compare cette stratégie à un entraînement complet et à des banques LoRA indépendantes.

Le cœur de **17 129 280 paramètres** a été initialisé aléatoirement et préentraîné sur **14 753 792 tokens vus** de français général. Aucun poids préentraîné d’IvoireSLM n’a servi à cette expérience. Il reste un petit modèle faiblement préentraîné, sans démonstration de capacités de raisonnement.

## Ce qui a été exécuté

- Cinq méthodes : complet, MCE 256, MCE 64, LoRA r8, LoRA r32.
- Deux sous-domaines d’un même glossaire : droit civil et droit pénal.
- Trois taux d’apprentissage par méthode, sélectionnés sur validation avant test : **15 modèles de réglage**.
- Trois graines d’adaptation et deux ordres des domaines : **30 modèles finaux**.
- 300 pas par domaine, batch 4, contexte 256 : **307 200 tokens par domaine et modèle**.
- Vérification du gel des paramètres, des anciennes routes et du rechargement des checkpoints.
- Un routeur de termes TF-IDF + régression logistique, puis mesure du coût de ses erreurs.

Les observations numériques, les scripts et les protocoles sont inclus. Les données textuelles, le tokenizer et les poids ne sont pas distribués. La réanalyse est accessible immédiatement ; le réentraînement exact requiert les entrées archivées décrites dans [Reproduction](docs/reproduction.md).

## Résultats principaux

Loss test en nats par token, **plus faible = meilleur**. Moyenne sur trois graines, ordre **civil puis pénal**, route fournie pour chaque banque :

| Méthode | Paramètres entraînés / domaine | Général | Civil | Pénal |
|---|---:|---:|---:|---:|
| Complet | 17 129 280 | 5,772 | 3,920 | 3,442 |
| MCE 256 | 1 969 920 | 4,414 | 3,685 | 3,716 |
| MCE 64 | 495 360 | 4,414 | 3,780 | 3,710 |
| LoRA r8 | 516 096 | 4,414 | 3,782 | 3,736 |
| LoRA r32 | 2 064 384 | 4,414 | 3,649 | 3,788 |

![Comparaison des losses sur les deux ordres](results/figures/figure_test_extension.png)

Les écarts types et le second ordre figurent dans le [tableau complet](results/tableau_test_extension.csv). Les banques produisent les mêmes losses pour les deux ordres à graine identique : ce contrôle structurel ne constitue pas six répétitions indépendantes.

**Interprétation :** le cœur et les anciennes routes gelées se conservent exactement dans cette expérience. Cette propriété bénéficie aussi aux banques LoRA. À capacité voisine, LoRA r32 obtient une meilleure loss civile que MCE 256 ; MCE 256 obtient une meilleure loss pénale. L’entraînement complet adapte fortement le dernier domaine mais dégrade le général. Ces comparaisons ne démontrent aucune supériorité générale de MCE.

Le routeur reçoit uniquement un **terme juridique**, sans sa définition : accuracy **83,2 %**, macro-F1 **0,791**, contre **67,7 %** pour la classe majoritaire, sur 155 termes test. Son rappel pénal n’est que **60 %**. Il n’a pas été évalué sur des questions libres et ne détecte pas le hors distribution. Les erreurs de route augmentent la loss des définitions ; le manuscrit les mesure séparément du routage fourni.

[Ouvrir la réanalyse dans Colab](https://colab.research.google.com/github/mandi-model/mandia-mce/blob/main/notebooks/reanalysis.ipynb)

## Démarrer

Python 3.12 conseillé pour la réanalyse :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python scripts/analyser_extension.py results/evidence analysis-output
python scripts/analyser_routeur.py
```

Les tableaux, intervalles descriptifs et figures sont recalculés depuis les observations, sans GPU et sans accès au stockage de l’auteur. Les groupes de termes sont remplacés par des identifiants SHA-256 stables, ce qui préserve le bootstrap groupé.

Pour vérifier le gel et la conservation après une mise à jour réelle sur CPU :

```bash
python -m pip install -r requirements.txt
python scripts/smoke_model.py
```

Cette démonstration utilise un modèle miniature aléatoire ; elle ne réexécute pas l’étude juridique. Les scripts GPU dans `experiments/` sont une version portable du runner historique ; leurs chemins d’entrée sont configurables. Voir [les prérequis et commandes](docs/reproduction.md).

## Organisation

| Chemin | Contenu |
|---|---|
| `mandi_model.py` | Transformer causal, RMSNorm, RoPE, SwiGLU, embeddings liés |
| `mandi_mce_v1.py` | Modules résiduels MCE et banques LoRA par domaine |
| `charger_etude_mce.py` | Reconstruction vérifiée depuis des checkpoints fournis séparément |
| `routeur_termes_mce.py` | Inférence du routeur depuis son état JSON fourni séparément |
| `experiments/` | Extension GPU et évaluation du routage |
| `scripts/` | Réanalyse, contrôle des empreintes, vérification CPU |
| `results/evidence/` | Observations numériques, contrôles et provenance |
| `results/figures/` | Figures de l’expérience achevée |
| `docs/` | Architecture, protocoles, provenance, reproduction, limites, arXiv |
| `paper/` | Manuscrit français et PDF publics |

## Limites et suites

Un seul cœur, un seul split, un seul glossaire, deux sous-domaines proches et trois graines d’adaptation limitent la portée. Le dropout du complet diffère de celui des banques. Les budgets de paramètres sont voisins, les FLOPs ne sont pas égalisés. L’audit lexical ne vérifie pas la contamination sémantique. Les losses sur des définitions ne mesurent ni exactitude juridique ni raisonnement.

La conservation dépend du gel, de la route choisie et de l’inférence déterministe. Il n’y a ni fusion ni coopération entre experts. Le protocole droit/histoire/science/économie, plusieurs cœurs et splits, un contrôle du dropout, les baselines de replay et le routage ouvert restent **à exécuter**. Voir [la feuille de route](docs/limitations.md).

MCE s’inscrit dans des travaux antérieurs sur l’expansion progressive, les adapters et LoRA. La conservation par gel n’est pas une nouveauté revendiquée. Voir [les références et les différences](docs/architecture.md).

## Provenance, droits et citation

Le glossaire provient de [bernard-ng/french-legal-glossary](https://huggingface.co/datasets/bernard-ng/french-legal-glossary), révision `f8c850c424af7a60ee55cf9eaf050cf05b59aba7`, avec une licence déclarée **CC BY-NC-SA 4.0**. Les textes et poids ne sont pas inclus. Voir [la fiche de provenance et les droits](docs/data.md).

Le code et la documentation sont consultables publiquement. Aucune licence générale de réutilisation n’est accordée à cette première version ; ne pas interpréter l’accès public comme une autorisation de redistribution ou d’usage commercial. La licence du dataset reste distincte.

Citation proposée : **mandia (2026). Expansion modulaire d’un petit modèle français : adaptation juridique, ordre des domaines et routage des termes. Manuscrit exploratoire et dépôt logiciel Mandia MCE.** Aucun DOI ni identifiant arXiv n’est attribué à ce travail à cette date.

OpenAI Codex a assisté le code, l’exécution, l’analyse et la rédaction. Les résultats proviennent d’une expérience exécutée et de ses fichiers observés ; la relecture et la responsabilité scientifiques demeurent celles de l’auteur.
