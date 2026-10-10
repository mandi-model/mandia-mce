# Validation de la diffusion du 10 octobre 2026

Les contrôles suivants ont été exécutés sur les fichiers de la version publique.

## Observations et calculs

- Réanalyse principale exécutée sans GPU, à partir de `results/evidence/`.
- Les 10 lignes des tableaux finaux et les 8 comparaisons appariées reproduisent tous les champs numériques des CSV archivés à une tolérance absolue de **1e-10**.
- Accuracy, macro-F1, matrice de confusion et intervalles groupés du routeur recalculés : concordance à **1e-12**.
- Les sommes NLL et comptes de cibles des 48 résultats sous route prédite sont vérifiés ; leurs moyennes sont recalculées.
- Les identifiants des groupes sont hachés dans la diffusion ; les résultats numériques et regroupements sont conservés.

## Modèle miniature sur CPU

`python scripts/smoke_model.py` exécuté avec Python 3.12 et PyTorch **2.14.1** : deux bottlenecks MCE et une banque LoRA.

Pour chaque variante, le contrôle entraîne un premier expert puis ajoute et entraîne réellement un second expert. Il vérifie que des paramètres du second expert changent, que les anciens paramètres restent exactement identiques, que les gradients du cœur et du premier expert sont absents, que les logits du premier expert et de la route générale sont exactement conservés, et que la sérialisation/reconstruction conserve les logits.

Ce contrôle local porte sur un petit modèle aléatoire. Il ne remplace ni l’expérience juridique GPU, exécutée avec PyTorch 2.11.0+cu130, ni une réplication scientifique.

## Sources et article

- `mandi_model.py` et `mandi_mce_v1.py` sont identiques octet par octet aux sources exécutées dans l’expérience archivée.
- Tous les scripts Python et les cellules Python du notebook de réanalyse sont vérifiés en syntaxe.
- Le PDF public compte **10 pages**, auteur **mandia** ; ses 10 pages ont été rendues et examinées visuellement.
- Les liens cloud privés et chemins du compte local ont été retirés. Les observations diffusées ne comportent aucun champ de texte documentaire, terme, vocabulaire ou email.
- Les liens relatifs Markdown sont contrôlés et les fichiers sont couverts par `release_manifest.json`.

Le runner GPU portable a été vérifié en syntaxe et dérivé du runner exécuté. **Aucun nouvel entraînement GPU ni exécution Colab du notebook public n’a été effectué pour cette diffusion.** Les corpus, tokenizer et checkpoints restent nécessaires au réentraînement exact et ne sont pas inclus.

Les contrôles d’intégrité et la réanalyse attestent la cohérence de cette diffusion. Ils ne constituent ni une certification indépendante des données ni une évaluation du manuscrit par les pairs.
