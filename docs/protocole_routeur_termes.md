# Annexe préspécifiée : routeur expérimental de termes juridiques

Le 10 octobre 2026, avant lecture des résultats de test de l'extension, ajout d'un routeur fermé civil/pénal. Entrée : uniquement le terme avant le premier saut de ligne, sans définition ni loss d'expert ni étiquette de test. TF-IDF de caractères word-boundary 2 à 5, 20 000 features maximum ; régression logistique balanced, C = 0,1 / 1 / 10, max_iter 1000, graine 20261010. Vocabulaire et IDF appris sur train uniquement. Sélection du C par macro F1 validation, égalité tranchée vers le C inférieur.

Une fois ce choix gelé, accuracy, macro F1, confusion, référence de classe majoritaire et bootstrap 2000 tirages par groupes de termes sur test. Une reconstruction du routeur depuis JSON doit reproduire toutes les décisions. Pas de pickle.

Pour les 24 banques finales (4 méthodes PEFT × 3 graines × 2 ordres), remplacer la sélection oracle d'expert par la prédiction du routeur. Garder les mêmes documents test, mais évaluer uniquement les tokens de définition et EOS, sans compter les tokens du terme fourni en entrée. Les routes oracle et prédite sont réévaluées sur ces mêmes cibles, séparément du tableau de loss intégrale. Publier la perte sous route prédite et le surcoût d'erreur de routage. Le terme est fourni en entrée avant génération : cette mesure évalue la complétion conditionnelle de définitions. Aucun réentraînement ou choix de modèle selon test.

Ce routeur est limité à deux catégories connues d'un glossaire ; il ne détecte pas les entrées générales ou hors distribution, ne route pas les questions et ne fusionne pas les experts. Le choix général demeure explicite. Cette annexe teste un premier composant de sélection autonome, pas la coopération conceptuelle de l'architecture initiale.
