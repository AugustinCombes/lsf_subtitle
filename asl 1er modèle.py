import numpy as np
np.random.seed(6)
import tensorflow as tf
import keras
import pandas as pd
tf.random.set_seed(6)
import matplotlib.pyplot as plt
#matplotlib inline
import cv2
import os
from os.path import isdir, join

data_path = r"C:\Users\Louise\Desktop\asl_alphabet\asl_alphabet_train"#lien vers les images de train
data_path_bis = r"C:\Users\Louise\Desktop\asl_alphabet\asl_alphabet_bis"#lien vers les images avec un autre fond


##Data extraction

#Fonction aide pour télécharger les images en foncrion de leur lien

folders = sorted(os.listdir(data_path))
folders_bis = sorted(os.listdir(data_path_bis))

def images_extract(directory):
    images = []
    labels = []
    for idx, label in enumerate(folders):
        folder_path = join(directory, label)
        for path in os.listdir(folder_path):
            image_path = join(folder_path, path)
            image = cv2.resize(cv2.imread(image_path), (64, 64))
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return(images, labels)


images, labels = images_extract(directory = data_path)
print(images)
print(labels)

#On sépare la base de données

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, stratify = labels)

X_bis, y_bis =images_extract(directory = data_path_bis)

#Grandeurs utiles

n=len(folders)
n_bis = len(X_bis)
N = int(len(X_train) / len(folders))
N_bis = int(len(X_bis) / len(folders_bis))
train_n = len(X_train)
test_n = len(X_test)

print("Nombre de symboles: ", n)
print("Nombre d'images d'entrainement': " , train_n)
print("Nombre d'images de test': ", test_n)
print("Nombre d'images avec un autre arrière plan:", n_bis)

#On trie :

y_train_indice = y_train.argsort()
y_train = y_train[y_train_indice]
X_train = X_train[y_train_indice]

y_test_indice = y_test.argsort()
y_test = y_test[y_test_indice]
X_test = X_test[y_test_indice]

y_bis_indice = y_bis.argsort()
y_bis = y_bis[y_bis_indice]
X_bis = X_bis[y_bis_indice]

##Affichage des images

#Fonction pour afficher les images

def affiche_image (liste) :
    Norm= int(len(liste) / len(folders))
    fig = plt.figure(figsize=(29,11))
    for i in range(n):
        ax = fig.add_subplot(4, 8, i + 1)
        ax.imshow(np.squeeze(liste[Norm*i]))
        ax.set_title("{}".format(folders[i]))
    plt.show()


#On affiche les images d'entrainement
print("Images d'entrainement': ")
affiche_image(X_train)

#On affiche les images de test
print("Images de test: ")
affiche_image( X_test)

#On affiche les images de l'autre jeu de données
print("Images avec un autre arrière plan")
affiche_image( X_bis)


## On examine le dataset

def comptage(liste) :
    Compte =[]
    for i in range (n):
        Compte.append(sum(liste==i))
    return(Compte)

Compte_train = comptage(y_train)
Compte_test = comptage(y_test)
Compte_bis = comptage(y_bis)

print(Compte_train)
print(Compte_test)
print(Compte_bis)

# On affiche le comptage
print("Training set:")
print("\tA: {}, B: {}, C: {}, D: {}, E: {}, F: {}, G: {}, H: {}, I: {}, J: {}, K: {}, L: {}, M: {}, N: {}, O: {}, P: {}, Q: {}, R: {}, S: {}, T: {}, U: {}, V: {}, W: {}, X: {}, Y: {}, Z: {}, del: {}, nothing: {}, space: {}".format(2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700, 2700))


print("Test set:")
print("\tA: {}, B: {}, C: {}, D: {}, E: {}, F: {}, G: {}, H: {}, I: {}, J: {}, K: {}, L: {}, M: {}, N: {}, O: {}, P: {}, Q: {}, R: {}, S: {}, T: {}, U: {}, V: {},W: {}, X: {}, Y: {}, Z: {}, del: {}, nothing: {}, space: {}".format(300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300))

print("Autre set:")
print("\tA: {}, B: {}, C: {}, D: {}, E: {}, F: {}, G: {}, H: {}, I: {}, J: {}, K: {}, L: {}, M: {}, N: {}, O: {}, P: {}, Q: {}, R: {}, S: {}, T: {}, U: {}, V: {},W: {}, X: {}, Y: {}, Z: {}, del: {}, nothing: {}, space: {}".format(30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30))


## On normalise et catégorise

#A = 0 B= 1 C=3 ... 0=[1,0,0], 2=[0,0,1]...

y_train_OH = keras.utils.to_categorical(y_train)
y_test_OH = keras.utils.to_categorical(y_test)
y_bis_OH = keras.utils.to_categorical(y_bis)

#Normalisation RGB
X_train_Norm = X_train.astype('float32')/255.0
X_test_Norm = X_test.astype('float32')/255.0
X_bis_Norm = X_bis.astype('float32')/255.0


##Modèle

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.layers import Flatten, Dense
from keras.models import Sequential


model = Sequential()

model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu',input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(filters=128,kernel_size=5,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=256,kernel_size=5,padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(29, activation='softmax'))

model.summary()



model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train_Norm, y_train_OH, epochs = 4, batch_size = 64) #Mettre le nombre d'epoch souhaité



##Test modèle

score = model.evaluate(x = X_test_Norm, y = y_test_OH, verbose = 0)
score_bis = model.evaluate(x = X_bis_Norm, y = y_bis_OH, verbose = 0)
print('Précision pour les images test:', round(score[1]*100, 3), '%',)
print('Précision pour les images avec un autre arrière plan:', round(score_bis[1]*100, 3), '%')


#plus difficile pour lui de reconnaitre quand les images ont un fond différent


## Matrice de confusion
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sn

#Pour images test

y_prob = model.predict(X_test_Norm, batch_size = 64, verbose = 0)
y_pred = np.argmax(y_prob,axis=-1)

cm=confusion_matrix(y_test,y_pred)

df_cm = pd.DataFrame(cm, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'],index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'])


sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True,vmin=0,vmax=300,fmt='d',xticklabels= True,yticklabels= True,annot_kws={"size": 7},cmap="coolwarm")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#Pour images avec un autre fond

y_prob_bis = model.predict(X_bis_Norm, batch_size = 64, verbose = 0)
y_pred_bis = np.argmax(y_prob_bis,axis=-1)


cm_bis=confusion_matrix(y_bis,y_pred_bis)

df_cm_bis = pd.DataFrame(cm_bis, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'],index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'])

sn.set(font_scale=1)
sn.heatmap(df_cm_bis, annot=True,vmin=0,vmax=30,fmt='d',xticklabels= True,yticklabels= True,annot_kws={"size": 7},cmap="coolwarm")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()




