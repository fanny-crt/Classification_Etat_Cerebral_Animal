# Classification de l'état célébrale d'un animal (Data Challenge)

L'objectif de ce projet est de classer l'état célébrale d'un animal. Pour cela, nous avons à notre disposition des séries temporelles représentant l’activité cérébrale d’un neurone. Dans ces fichiers, chaque individu correpond à un  ́echantillon de série et est constitué de 50 données prises dans le temps. L'étiquette de classification (”TARGET”) attribue la classe 0 pour l’état 1 et la classe 1 pour l’état 2. C’est donc un problème de classification binaire supervisée. Nous avons 16635 individus pour l’échantillon d’entrainement et 11969 individus dans l’échantillon test.

# Features Extraction

# Model Building

# Model Performance

La particularité de ce data challenge est que les distributions des classes sont disproportionnées. En effet, dans l’échantillon d'entrainement, on a 13577 individus appartenant à la classe 0 et 3058 individus appartenant à la classe 1. C’est pour cela que nous avons choisi le Kappa de Cohen pour déterminer le meilleur modèle. Le calcul de ce score κ se fait de la manière suivante :

![alt text](https://github.com/fanny-crt/Classification_etat_celebrale_animal/blob/main/images/kappa_cohen.PNG)

où Pr(a) est la probabilité que les prédictions concordent avec les valeurs réelles des classes 0 et 1 et Pr(h) est la somme de la probabilité que les prédictions concordent avec les valeurs réelles de la classe 0 par hasard et de la probabilité que les prédictions concordent avec les valeurs réelles de classe 1 par hasard. Le kappa de Cohen est donc une bonne métrique d’évaluation pour les données déséquilibrées, car il prend en compte que les bonnes estimations qui ne sont pas dû au hasard.


# Code 
- Python
