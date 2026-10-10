# Données et provenance

## Cœur français général

Initialisation aléatoire ; préentraînement sur des sous-ensembles de Wikipédia français et FineWeb. Le cœur archivé est au pas 3632, après 14 753 792 tokens vus. Le train préparé comporte 27 587 documents et 20 001 329 tokens ; les tokens préparés ne sont pas tous des tokens vus par ce checkpoint. Le tokenizer BPE compte 8192 tokens. Aucun poids d’IvoireSLM n’a été utilisé.

Le corpus général, son tokenizer et le checkpoint ne sont pas distribués dans ce dépôt. Le présent kit ne contient pas la chaîne complète de reconstruction du corpus général depuis les sources ; il ne prétend pas rendre le préentraînement exact publiquement réplicable.

## Glossaire juridique

Source : [bernard-ng/french-legal-glossary](https://huggingface.co/datasets/bernard-ng/french-legal-glossary). Révision : `f8c850c424af7a60ee55cf9eaf050cf05b59aba7`. Empreinte du fichier source : `c26153be965d4ef30c6125ad4770827d0a91d5ce1851607092e33ab28ac5221d`. Licence **déclarée par la fiche : CC BY-NC-SA 4.0**. Les droits des textes sous-jacents, leur juridiction et leur actualité n’ont pas été vérifiés indépendamment.

Catégories retenues : Droit civil et Droit pénal sans catégorie multiple ambiguë. Déduplication globale des définitions et textes, filtres de longueur et termes sur une seule ligne. Split 80/10/10 par familles de termes fondées sur le premier mot normalisé, avec contrôle lexical Jaccard de 5-grammes au seuil 0,85. Il s’agit du split réalisé, pas d’une garantie d’absence de paraphrases.

| Domaine | Train : documents / tokens | Validation | Test |
|---|---:|---:|---:|
| Civil | 788 / 143 273 | 88 / 15 748 | 105 / 18 628 |
| Pénal | 207 / 37 216 | 28 / 5 337 | 50 / 7 574 |

Les tokens de corpus incluent les premiers tokens de documents ; le test principal compte 18 523 cibles civiles et 7 524 pénales. Un document pénal du **train** est signalé par l’audit lexical contre le train général ; aucun document du test n’atteint le seuil fixé. La contamination sémantique n’est pas évaluée.

## Diffusion des observations

Les fichiers publics contiennent des losses, comptes de tokens, coûts, seeds, contrôles, empreintes de documents et labels de groupes pseudonymisés par SHA-256. Ils ne contiennent pas les termes, définitions, vocabulaire du routeur ou poids. Cette transformation conserve les regroupements nécessaires au bootstrap. Les empreintes peuvent permettre un rapprochement avec une source publique : elles ne sont pas présentées comme une anonymisation garantie.

`results/evidence/provenance.json` distingue les empreintes des fichiers observés originaux de celles de leur version publique. `release_manifest.json` couvre les fichiers de la diffusion. Les empreintes sont des contrôles d’intégrité, sans certification externe des résultats.

La consultation publique du dépôt ne constitue pas une licence générale de réutilisation. Une éventuelle diffusion des données ou poids devra documenter ses propres droits et conditions, distincts de ceux du code et du manuscrit.
