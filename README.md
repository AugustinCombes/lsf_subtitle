# lsf_subtitle

Comment traduire la langue des signes? 
Ce projet cherche à améliorer la traduction de la langue des signes en utilisant notamment Mediapipe et des réseaux de neurone.

# 1ère partie

-Photos et Réseau de Neurone
Dans une première partie on utilise deux bases de données sur la langue des signes américaines, une première base de 87 000 photos et une deuxièmes bas de 870 photos.
Autour de la première base nous construisons un réseau de neuronne capable d'identifier un signe. On observe que le réseau n'est pas performant pour identifier les signes de la deuxième base de données. On arrive à la conclusion que des arrières plans de photos très différents sont la cause majeur de cette faible performance. 
 
-Filtre Médiapipe et Réseau de Neurone
En partant du constat de la première partie, nous utilisons Mediapipe pour ajouter plus de précision à notre reconnaissance des mains pour traiter les photos des deux bases de données. En faisant tourner le réseau sur ces nouvelles bases de photos(fond blanc avec les dessins de la main par Mediapipe) on voit que le problème d'arrière plan est résolut et on atteint une performance de 77% de réussite.

# 2ème partie

Forts de ce constat, nous créons un moyen pour récupérer des données vidéos sur la langue des signes françaises, données directement traduites en coordonées grace à Mediapipe, avec pour but la traduction de vidéos. Cette partie est la construction d'outils pour la traduction de vidéos en utilisant Mediapipe. 

#Conclusions et aller plus loin

Le problème se révèle plus complexe que prévu (manque de données sur la langue des signes française et les temps d'apprentissages très longs). Ces travaux peuvent être amenés plus loins en vue de faire de la traduction de vidéo en direct.
