# Réanalyse, vérification et réentraînement

## 1. Réanalyse des résultats exécutés — sans GPU

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python scripts/analyser_extension.py results/evidence analysis-output
python scripts/analyser_routeur.py
```

Le premier script recalcule les deux ordres, les moyennes/écarts types sur trois graines, huit comparaisons avec bootstrap apparié par groupe (2000 tirages), les courbes et figures. Les identifiants de groupes sont hachés ; les sommes NLL, nombres de cibles et regroupements sont préservés. Le comptage attendu est 90 évaluations finales, 945 points de courbe, 90 coûts, 15 réglages et 30 rechargements.

Le second script recalcule accuracy, macro-F1, matrice de confusion, intervalles du routeur et moyennes des losses sous route prédite, et les vérifie contre les observations archivées.

Le bootstrap est descriptif et conditionnel. L’ordre de parcours des groupes reste celui des observations, ce qui préserve les tirages du script. La dernière décimale peut varier avec l’environnement numérique ; le script de contrôle compare les tableaux à 1e-10.

## 2. Vérification fonctionnelle du code sur CPU

```bash
python -m pip install -r requirements.txt
python scripts/smoke_model.py
```

Une mise à jour réelle d’un nouvel expert sur un modèle miniature aléatoire vérifie l’absence de gradients sur le cœur et l’ancienne banque, la stabilité exacte de leurs poids/logits, et le rechargement. Les deux variantes MCE et les banques LoRA sont couvertes. Ce contrôle ne mesure aucune compétence juridique.

## 3. Réentraînement du protocole historique — entrées requises

Les scripts `experiments/` conservent la logique de l’expérience ; le montage Drive et les chemins privés ont été remplacés par des variables d’environnement. Cette adaptation portable est vérifiée en syntaxe ; aucun nouvel entraînement GPU n’a été effectué pour cette diffusion.

Le protocole principal a été exécuté avec PyTorch **2.11.0+cu130**, Tesla T4, autocast bfloat16 et poids FP32 ; le GPU ne déclare pas de support BF16 natif. Le routeur utilisait scikit-learn **1.6.1**. Les `requirements` indiquent les dépendances de travail, pas une garantie de reproduction bit à bit de l’environnement Colab.

Trois répertoires locaux, **non fournis**, sont nécessaires :

- `MCE_INPUT_DIR` : `manifest.json`, `config.json` avec `general_validation_starts`, `mandi_model.py`, `mandi_mce_v1.py`, `tokenizer.json`, `documents.jsonl`, `general_validation.bin`, et `civil_{train,validation,test}.bin` / `penal_{train,validation,test}.bin`.
- `MCE_CORE_DIR` : `latest.pt`, contenant les clés `model` et `config` du cœur préentraîné.
- `MCE_CORPUS_DIR` : `train.jsonl` pour l’audit lexical, `test.bin` général et `manifest.json`.

Les flux `.bin` sont des IDs de tokens uint16 little-endian. Les documents juridiques portent `hash`, `group`, `domain`, `split` et `text`, ce dernier sous forme `terme\ndefinition`. Le train général comporte `hash` et `text`. Le manifeste des entrées contient, pour chaque fichier, `sha256` et `bytes`. Le tokenizer connaît `<eos>`.

L’empreinte du cœur archivé est `38233f63ee915a882ee49f0bbf79b280389101ab46f2d09d27ad2f08559e43ab`. Celle du tokenizer est `07fd856fa6e63c8e8594fda5b5d4317a11e906801c049066cd851eda328fb77d`. Le runner refuse un autre cœur : changer cette attente pour une nouvelle expérience exige de documenter que les résultats ne sont pas ceux du cœur publié ici.

```bash
export MCE_INPUT_DIR=/path/to/prepared-legal-inputs
export MCE_CORE_DIR=/path/to/archived-core
export MCE_CORPUS_DIR=/path/to/prepared-general-corpus
export MCE_OUTPUT_DIR=/path/to/new-runs
python experiments/run_all.py
```

Le script crée deux nouveaux répertoires par run, vérifie les empreintes des entrées, effectue l’audit, sélectionne les LR sur validation puis évalue tous les modèles. Le routeur est sélectionné sur validation et le coût des routes prédites est évalué ensuite. Les répertoires existants ne sont pas écrasés. Le budget du principal est borné à 3600 secondes ; le routeur est supplémentaire.

Les traces de l’expérience achevée ne doivent jamais être remplacées par des données simulées pour faire fonctionner ces commandes. L’obtention ou la reconstruction des entrées requises reste une condition de réplication ; l’URL du dataset seule ne reconstruit ni le cœur ni le split exact.

## 4. Inférence depuis un checkpoint fourni séparément

`charger_etude_mce.charger_pilote` vérifie les empreintes et les clés avant chargement, puis gèle tous les poids en mode eval. `prediction_pilote` choisit explicitement `civil`, `penal` ou `None` pour la route générale. Le nom hérité « pilote » n’indique pas un ancien checkpoint : le chargeur gère les cinq méthodes de l’extension.

`RouteurTermesMCE` reconstruit le TF-IDF et la décision logistique depuis un fichier JSON. Les poids JSON du routeur ne sont pas distribués dans cette première version. Il ne reçoit qu’un terme sur une ligne et choisit civil/pénal.
