# lsf_subtitle

Comment sous-titrer efficacement la langue des signes? 
Ce projet cherche à sous-titrer la langue des signes (française, anglaise, en réalité le code ne dépend pas de la langue des signes utilisée) en utilisant notamment Mediapipe et des réseaux de neurone.

# 1ère partie

- Photos et Réseau de Neurone :
Dans une première partie on utilise deux bases de données sur la langue des signes américaines, une première base de 87 000 photos et une deuxièmes base de 870 photos.
Autour de la première base nous construisons un réseau de neurones capable d'identifier les signes. On observe que le réseau n'est pas très performant pour identifier les signes de la deuxième base de données: les arrières plans de photos très différents sont la cause majeure de cette performance (accuracy) décevante de 33%. 
 
- Filtre Médiapipe et Réseau de Neurone :
En partant du constat de la première partie, nous utilisons Mediapipe pour ajouter plus de précision à notre reconnaissance des signes et des positions des mains et traiter les photos des deux bases de données. En faisant tourner le réseau sur ces nouvelles bases d'images transformées (fond blanc avec les dessins du "squelette" de la main par Mediapipe) on voit que le problème d'arrière plan est résolu et on atteint une accuracy de 77% dans la reconnaissance de signes sur le même nombre d'epochs que dans la démarche précédente.

# 2ème partie

Forts de ce constat, nous créons un moyen pour récupérer des données vidéos sur la langue des signes françaises efficacement (enhanced_vocab_making.py): les données sont directement traduites en coordonnées numériques (landmarks) grâce à Mediapipe, avec pour but la traduction de vidéos ou d'input caméra en temps réel. On peut ensuite entrainer un réseau de neurone feed-forward pour tenter de reconnaître les signes effectués à la caméra (cam_recognition.py), sous réserve que ces derniers ont été intégrés à la base de données préalablement.

# Conclusions et aller plus loin

Le problème s'est révélé complexe (manque de données sur la langue des signes française et les temps d'apprentissages très longs). Une suite possible au projet serait de mettre en place une traduction de phrases, en faisant apprendre à un réseau de neurone comment réordonner les mots pour respecter la grammaire très particulière des différentes langues des signes, en particulier anglaise et française. Nous sommes cependant contents du travail effectué, le programme permettant de générer facilement et pratiquement des bases de données de signes.
