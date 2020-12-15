# lsf_subtitle

Ce projet cherche à améliorer la traduction de la langue des signes en utilisant notamment Mediapipe.

# 1ère partie

Dans une première partie on utilise deux bases de données sur la langue des signes américaines, une première base de 87 000 photos et une deuxièmes bas de 870 photos.
Autour de la première base nous construisons un réseau de neuronne capable d'identifier un signe. On observe que le réseau n'est pas performant pour identifier les signes de la deuxième base de données car les arrières plans sont différents.

# 2ème partie 

En partant du constat de la première partie, nous utilisons Mediapipe pour traiter les photos des deux bases de données. En faisant tourner le réseau sur ces nouvelles bases de photos on voit que le problème d'arrière plan est résolut. 

# 3ème partie

Forts de ce constat, nous créons un moyen pour récupérer des données sur la langue des signes françaises, données directement traduites en coordonées grace à Mediapipe. 

