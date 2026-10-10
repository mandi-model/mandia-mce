# Mandi — extension exploratoire préspécifiée du 10 octobre 2026

Nous conservons le même cœur (pas 3632), le même tokenizer et les mêmes splits du glossaire que le pilote du 9 octobre. Cette extension ne remplace pas le protocole confirmatoire à quatre domaines.

## Comparaisons et budgets fixés avant calcul

Cinq méthodes : entraînement complet, MCE bottleneck 256, MCE bottleneck 64 (ablation de capacité), LoRA r8, LoRA r32. Les tailles des petites et grandes variantes sont approximativement comparables, sans égalité exacte.

Recherche identique de trois LR maximaux par méthode : 0,0001 ; 0,0003 ; 0,001. Une graine de réglage séparée (20261012), ordre civil puis pénal, 300 pas par domaine. Sélection du LR par moyenne non pondérée des losses civiles et pénales finales sur validation, avec préférence au LR inférieur en cas d'égalité. Le général est mesuré mais exclu de ce critère ; le complet peut donc sacrifier le général. Aucune sélection de checkpoint : dernier pas fixé.

Après gel des cinq LR, trois nouvelles graines (20261013, 20261014, 20261015), deux ordres (civil/pénal et pénal/civil), 300 pas par domaine, batch 4, contexte 256, soit 307 200 tokens par domaine et modèle. AdamW betas 0,9/0,95, weight decay matriciel 0,1, clip 1, chauffe 30 pas puis cosinus, optimiseur redémarré par domaine. Les mêmes fenêtres et graines d'initialisation par domaine sont utilisées dans tous les modèles. Cœurs PEFT en eval et dropout adapters 0,1 ; complet en train : cette différence reste déclarée. L'ordre des méthodes est tournant pour les mesures de temps. Le chronométrage couvre les étapes d'entraînement synchronisées, hors validation et IO ; il n'est pas directement interchangeable avec celui du pilote initial.

15 modèles de réglage puis 30 modèles finaux : 27 648 000 tokens d'adaptation au total, en comptant toutes les méthodes, graines et recherches. Budget maximal de calcul de 60 minutes, avec sauvegarde des résultats intermédiaires. Aucun achat ni publication externe.

## Validation, test et contrôles

Courbes à 100, 200 et 300 pas. La validation conserve les concaténations et fenêtres du pilote afin d'observer les budgets, mais le test juridique est évalué par document, sans transition entre deux définitions. Chaque cible admissible est évaluée une fois ; le premier token de chaque document n'est pas prédit, l'EOS l'est. Le test général utilise 32 fenêtres fixes de 256 cibles : estimation partielle annoncée. Le test n'est accessible à l'évaluation du modèle qu'après sauvegarde du choix des LR. Les résultats de test ne modifient ni entraînement, ni LR, ni ordre, ni choix de modèle. Toutes les méthodes sont publiées dans les tableaux.

Audit lexical supplémentaire contre les documents du train général : signal si au moins 20 5-grammes uniques et au moins 50 % des 5-grammes d'une définition sont présents dans un même document général. Les splits ne sont pas modifiés post-hoc. Une sensibilité séparée exclut les documents test signalés ; ce seuil ne mesure pas le chevauchement sémantique.

Chaque banque vérifie les gradients absents du cœur et des anciens modules, les empreintes des poids gelés et l'égalité exacte des logits de l'ancien expert. Les 30 checkpoints finaux sont sauvegardés puis reconstruits et leurs trois losses de validation vérifiées (erreur < 1e-6). Le stockage du modèle complet préserve le partage embeddings/tête ; cela corrige un coût disque artificiel du premier pilote.

## Analyse de l'article prévue avant test

Présenter séparément les deux ordres, les trois graines, les variantes de capacité et les LR sélectionnés. Donner mean ± sd, pertes finales et oubli après le deuxième domaine. Comparaisons MCE vs LoRA de capacité voisine avec bootstrap apparié par groupes de termes sur les pertes documentaires moyennées sur les graines, 2000 tirages et intervalles descriptifs à 95 %. Ces intervalles conditionnels ne couvrent pas la variabilité du cœur, du split ou de l'optimisation. Aucun test de significativité confirmatoire.

Le test juridique évalue l'adaptation au glossaire, pas le raisonnement juridique. Un seul corpus, un seul cœur, une seule graine de recherche et une grille limitée empêchent toute revendication de supériorité générale. Routage explicite, pas de coopération ou fusion. Source déclarée CC BY-NC-SA 4.0 ; les corpus et checkpoints restent privés. La nouveauté architecturale reste à établir.
