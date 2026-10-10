# Ce qui est établi et ce qui reste à tester

**Mesuré :** adaptation textuelle de définitions civiles/pénales, conservation sous route gelée, budgets de paramètres, sensibilité à l’ordre, trois graines, rechargement, comparaison à LoRA, accuracy du routeur de termes et pénalité de ses erreurs.

**Non établi :** supériorité générale de MCE, nouveauté de la conservation par gel, raisonnement juridique, vérité des générations, compétences souveraines ou ivoiriennes, généralisation à plusieurs corpus, coopération entre experts, routage de questions libres, détection du hors distribution.

## Limites de l’étude achevée

- Un seul cœur faiblement préentraîné, un split et un glossaire ; deux domaines liés.
- Trois graines d’adaptation, une seule graine de recherche et LR choisi dans un seul ordre.
- Dropout du cœur complet différent de celui des méthodes à cœur gelé.
- Capacités voisines, sans égalisation des FLOPs ni recherche exhaustive.
- Baselines replay, EWC et module partagé réentraîné successivement absentes.
- Général évalué sur 32 fenêtres ; petit test pénal.
- Bootstrap descriptif groupé conditionnel au cœur et split, sans incertitude complète ni test confirmatoire.
- Anciennes routes et général préservés structurellement ; cela ne mesure pas une compétence nouvellement combinée.
- Routeur fermé : rappel pénal 60 %, sans détection générale ou OOD.

## Feuille de route scientifique

1. Préspécifier une étude droit/histoire/science/économie avant examen de ses tests.
2. Répéter sur plusieurs splits et plusieurs cœurs initialisés/préentraînés indépendamment.
3. Contrôler le dropout et les budgets de calcul ; élargir les baselines.
4. Évaluer questions factuelles et génération avec des critères et annotateurs définis.
5. Évaluer routage général/OOD, puis coopération ou fusion avec mesure de l’effet sur les anciennes routes.
6. Rendre disponibles, dans un cadre de droits clarifié, les entrées nécessaires à une réplication indépendante.

Ces étapes sont futures. Leur présence dans la feuille de route n’indique aucun résultat déjà obtenu.
