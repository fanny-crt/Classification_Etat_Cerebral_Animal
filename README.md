# Classification de l'état célébrale d'un animal (Data Challenge)

L'objectif de ce projet est de classer l'état célébrale d'un animal. Pour cela, nous avons à notre disposition des séries temporelles représentant l’activité cérébrale d’un neurone. Dans ces fichiers, chaque individu correpond à un  ́echantillon de série et est constitué de 50 données prises dans le temps. L'étiquette de classification (”TARGET”) attribue la classe 0 pour l’état 1 et la classe 1 pour l’état 2. C’est donc un problème de classification binaire supervisée. Nous avons 16635 individus pour l’échantillon d’entrainement et 11969 individus dans l’échantillon test.

# Features Extraction

Dans un premier temps, nous avons extrait les features de chaque série temporelle de l’échantillon d’entrainement et de test avec la fonction extract_features de la librairie tsfresh de Python. Cependant, on obtient un jeu de données avec 464 variables. Pour réduire la dimension du jeu de données, nous avons ensuite sélectionné les variables avec la fonction select_features de tsfresh. On obtient un jeu de données avec 240 variables.

Dans un second temps, nous avons effectué une différenciation (∆Xt = Xt − Xt−1) des séries temporelles des  ́echantillons d’entrainement et de test. Nous avons ensuite appliqué une extraction et une sélection des features de la même manière que précédemment. Par la suite, les algorithmes de Machine Learning seront appliqués sur ces nouveaux jeux de données.

# Model Building

- Linear Discriminant Analysis (Analyse discriminante linéaire)

Application sur les features extraites des séries temporelles

![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/analyse_discriminante_lineaire.PNG)

Application sur les features extraites des séries temporelles différenciées

![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/analyse_discriminante_lineaire_diff.PNG)

- Logistic Regression (Régression logistique binaire)

Application sur les features extraites des séries temporelles

Application sur les features extraites des séries temporelles différenciées

- Random Forest (Forêts aléatoires)

Application sur les features extraites des séries temporelles

Application sur les features extraites des séries temporelles différenciées

# Model Performance

La particularité de ce data challenge est que les distributions des classes sont disproportionnées. En effet, dans l’échantillon d'entrainement, on a 13577 individus appartenant à la classe 0 et 3058 individus appartenant à la classe 1. C’est pour cela que nous avons choisi le Kappa de Cohen pour déterminer le meilleur modèle. Le calcul de ce score κ se fait de la manière suivante :

![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/kappa_cohen.PNG)

où Pr(a) est la probabilité que les prédictions concordent avec les valeurs réelles des classes 0 et 1 et Pr(h) est la somme de la probabilité que les prédictions concordent avec les valeurs réelles de la classe 0 par hasard et de la probabilité que les prédictions concordent avec les valeurs réelles de classe 1 par hasard. Le kappa de Cohen est donc une bonne métrique d’évaluation pour les données déséquilibrées, car il prend en compte que les bonnes estimations qui ne sont pas dû au hasard.


# Code 
- Python : scikit-learn, tsfresh
