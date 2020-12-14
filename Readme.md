# Décomposition en valeurs singulières SVD
Cet archive contient les fichiers suivants: 

- svd_lanlk.c : contient l'algorithme de bidiagonalisation et l'utilisation des routines lapack pour la décomposition spectrale de la matrice tridiagonale obtenu.

- svd_lanqr.c : contient l'algorithme de bidiagonalisation et l(utilisatin de la méthode QR shifté.

- svd_openmp.c : contient la version parallèle de la bidiagonalisation et utilisation des routines lapacke

- benchs.c : Pour les mesures de performances effectuées

- Makefile1.benchs: pour les mesures des perfs du programme SVD avec : bidigonalisation de Golub-Kahan-Lanczos et les routines de lapacke (fichier svd_lanlk.c)

- Makefile3.benchs : pour les mesures de perfs du programme SVD avec bidigonalisation et la méthode QR shifté (fichier svd_lanqr.c)

- Makefile2.benchs : pour mesurer les perfs de l'algorithme de bidigonalisation

- Le rapport du projet en format pdf
