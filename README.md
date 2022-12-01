# Classification de l'état célébrale d'un animal (Data Challenge)

L'objectif de ce data challenge est de classer l'état célébrale d'un animal. Pour cela, nous avons à notre disposition des séries temporelles représentant l’activité cérébrale d’un neurone. Dans ces fichiers, chaque individu correpond à un  ́echantillon de série et est constitué de 50 données prises dans le temps. L'étiquette de classification (”TARGET”) attribue la classe 0 pour l’état 1 et la classe 1 pour l’état 2. C’est donc un problème de classification binaire supervisée. Nous avons 16635 individus pour l’échantillon d’entrainement et 11969 individus dans l’échantillon test.

![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/xtrain.PNG)
![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/ytrain.PNG)

# Features Extraction

Dans un premier temps, nous avons extrait les features de chaque série temporelle de l’échantillon d’entrainement et de test avec la fonction extract_features de la librairie tsfresh de Python. Cependant, on obtient un jeu de données avec 464 variables. Pour réduire la dimension du jeu de données, nous avons ensuite sélectionné les variables avec la fonction select_features de tsfresh. On obtient un jeu de données avec 240 variables.

Dans un second temps, nous avons effectué une différenciation (∆Xt = Xt − Xt−1) des séries temporelles des  ́echantillons d’entrainement et de test. Nous avons ensuite appliqué une extraction et une sélection des features de la même manière que précédemment. Par la suite, les algorithmes de Machine Learning seront appliqués sur ces nouveaux jeux de données.

# Model Building

- Linear Discriminant Analysis (Analyse discriminante linéaire)

Application sur les features extraites des séries temporelles

![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/Analyse_discriminante_lineaire.PNG)

Application sur les features extraites des séries temporelles différenciées

![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/Analyse_discriminante_lineaire_diff.PNG)

- Logistic Regression (Régression logistique binaire) 

Les hyperparamètres de la régression logistique sont déterminés par une grille de recherche par validation croisée.

Application sur les features extraites des séries temporelles avec les hyperparamètres suivants : 
penalty : "none" 
solver : "newton-cg"

Nous avons également décidé d’appliquer une régression logistique pénalisée de type lasso en fixant les hyperparamètres penalty à ”l1”, C à 0.8 et solver à "liblinear" de la fonction LogisticRegression() de la librairie sklearn. 

Application sur les features extraites des séries temporelles différenciées
penalty : "none" 
solver : "newton-cg"

Comme pour les features extraites des séries temporelles, nous avons également décidé d’appliquer une régression logistique pénalisée de type lasso en fixant les hyperparamètres penalty à ”l1”, C à 0.8 et solver à "liblinear" de la fonction LogisticRegression() de la librairie sklearn. 

- Random Forest (Forêt aléatoire)

Les hyperparamètres de la forêt aléatoire sont déterminés par une grille de recherche par validation croisée.

Application sur les features extraites des séries temporelles
criterion: "entropy"
min samples leaf: 10
min samples split 10 
n estimators : 50

Application sur les features extraites des séries temporelles différenciées
criterion: "gini"
min samples leaf: 10
min samples split 10 
n estimators : 50

# Model Performance

La particularité de ce data challenge est que les distributions des classes sont disproportionnées. En effet, dans l’échantillon d'entrainement, on a 13577 individus appartenant à la classe 0 et 3058 individus appartenant à la classe 1. C’est pour cela que nous avons choisi le Kappa de Cohen pour déterminer le meilleur modèle. Le calcul de ce score κ se fait de la manière suivante :

![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/kappa_cohen.PNG)

où Pr(a) est la probabilité que les prédictions concordent avec les valeurs réelles des classes 0 et 1 et Pr(h) est la somme de la probabilité que les prédictions concordent avec les valeurs réelles de la classe 0 par hasard et de la probabilité que les prédictions concordent avec les valeurs réelles de classe 1 par hasard. Le kappa de Cohen est donc une bonne métrique d’évaluation pour les données déséquilibrées, car il prend en compte que les bonnes estimations qui ne sont pas dû au hasard.

![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/model_performance_neurones.PNG)


# Code 
- Python : scikit-learn, tsfresh
- Data Challenge : https://challengedata.ens.fr/challenges/14
