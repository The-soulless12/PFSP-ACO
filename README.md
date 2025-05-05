# PFSP-ACO
Implémentation en Python de l’algorithme d’optimisation par colonies de fourmis (ACO) pour le problème du flowshop de permutations (PFSP), visant à minimiser le makespan à l'aide de la phéromone.

# Fonctionnalités
- Résolution du PFSP à partir des instances de Taillard.
- Construction probabiliste de solutions basée sur la phéromone et l’heuristique.
- Arrêt de l'algorithme en cas de stagnation : pas d’amélioration pendant un certain nombre d’itérations.
- Optimisation automatique des hyperparamètres via Optuna (recherche bayésienne parallèle).

# Structure du projet
- aco.py : Contient l'implémentation principale de l’algorithme ACO, la logique de l’optimisation Optuna et la gestion des paramètres.
- fonctions.py : Regroupe les fonctions essentielles telles que la lecture et l'analyse des fichiers d’instances ainsi que l’évaluation du makespan pour une combinaison donnée.
- data/ : Répertoire contenant les fichiers d’instances du problème du flowshop de permutations.

# Prérequis 
- Python version 3.x
- Les blibliothèques optuna & numpy.

# Note
- Pour exécuter le projet, saisissez la commande `python aco.py nom_fichier.txt indice_instance` dans votre terminal.
- L’algorithme **Ant Colony Optimization (ACO) 🐜** s’inspire du comportement collectif des fourmis pour résoudre des problèmes d’optimisation. Chaque fourmi virtuelle explore différentes solutions en s’appuyant sur deux éléments clés : les **phéromones** qui servent de mémoire collective pour guider les futures explorations vers de bons chemins et une **fonction heuristique** qui prend en compte des critères locaux comme le temps de traitement, pour orienter les choix. À chaque itération, plusieurs solutions sont générées. Le meilleur makespan est identifié et les traces de phéromones sont ajustées en fonction de la qualité des solutions obtenues. Ce mécanisme permet de renforcer les bons chemins tout en laissant les moins performants s’estomper. Le processus se poursuit jusqu’à ce qu’aucune amélioration ne soit observée pendant 30 itérations consécutives.
- Pour optimiser les performances de l’algorithme, la bibliothèque **Optuna** a été utilisée. Elle permet d’ajuster automatiquement les paramètres de l’ACO (alpha, beta, taux d’évaporation, Q & le nombre de fourmis) afin d’obtenir une meilleure convergence. Optuna exécute plusieurs essais en parallèle, ce qui réduit le temps de recherche tout en augmentant les chances d’atteindre une solution optimale.
