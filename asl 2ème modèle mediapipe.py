import numpy as np
np.random.seed(6)
import tensorflow as tf
import keras
import pandas as pd
tf.random.set_seed(6)
import matplotlib.pyplot as plt
import itertools
import seaborn as sn
import cv2
import os
from os.path import isdir, join
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.layers import Flatten, Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix

data_path = r"C:\Users\Louise\Desktop\asl_alphabet_mp\asl_alphabet_train_mp"#lien vers les images de train aprés mediapipe
data_path_bis = r"C:\Users\Louise\Desktop\asl_alphabet_mp\asl_alphabet_bis_mp"#lien vers les images avec un autre fond aprés mediapipe


folders = os.listdir(data_path)
folders_bis = os.listdir(data_path_bis)

##Data extraction
def images_extract2(directory):
    fold=os.listdir(directory)
    images = []
    labels = []
    for idx,label in enumerate(fold):
        folder_path = join(directory, label)
        image_path = join(folder_path, folder_path)
        image = cv2.resize(cv2.imread(image_path), (64, 64))
        images.append(image)
        labels.append(fold[idx][14])
    images = np.array(images)
    labels = np.array(labels)
    return(images, labels)


images, labels = images_extract2(directory = data_path)

X_bis, y_bis =images_extract2(directory = data_path_bis)


#Alphabet vers nombres
alphabet  = ['$','@','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

#maintenant del= 0, space= 1, A=2 ect
def conversion(liste):
    L=[]
    for elm in liste :
        for i in range(len(alphabet)):
            if elm == alphabet[i]:
                L.append(i)
    return(L)

y_bis=conversion(y_bis)
labels=conversion(labels)

#On sépare la base de données

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, stratify = labels)

X_train = np.array(X_train)
X_test = np.array(X_test)
X_bis = np.array(X_bis)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_bis = np.array(y_bis)



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

##Comptage

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

print("Training set:")
print("\tdel: {},space: {}, A: {}, B: {}, C: {}, D: {}, E: {}, F: {}, G: {}, H: {}, I: {}, J: {}, K: {}, L: {}, M: {}, N: {}, O: {}, P: {}, Q: {}, R: {}, S: {}, T: {}, U: {}, V: {}, W: {}, X: {}, Y: {}, Z: {}".format(1491, 1717, 1111, 1366, 1651, 1753, 359, 1494, 1580, 1705, 1805, 1712, 1749, 1924, 2062, 792, 737, 1327, 1125, 823, 1646, 1442, 510, 1527, 1533, 1506, 1602, 1317))


print("Test set:")
print("\tdel: {},space: {}, A: {}, B: {}, C: {}, D: {}, E: {}, F: {}, G: {}, H: {}, I: {}, J: {}, K: {}, L: {}, M: {}, N: {}, O: {}, P: {}, Q: {}, R: {}, S: {}, T: {}, U: {}, V: {},W: {}, X: {}, Y: {}, Z: {}".format(166, 191, 123, 152, 184, 195, 40, 166, 176, 190, 201, 190, 194, 214, 229, 88, 82, 147, 125, 91, 183, 160, 57, 170, 170, 167, 178, 146))

print("Autre set:")
print("\tdel: {},space: {}, A: {}, B: {}, C: {}, D: {}, E: {}, F: {}, G: {}, H: {}, I: {}, J: {}, K: {}, L: {}, M: {}, N: {}, O: {}, P: {}, Q: {}, R: {}, S: {}, T: {}, U: {}, V: {},W: {}, X: {}, Y: {}, Z: {}".format(30, 30, 24, 26, 26, 26, 27, 30, 30, 30, 29, 21, 30, 26, 29, 22, 23, 25, 29, 25, 30, 28, 17, 30, 27, 28, 30, 30))


## On normalise et catégorise


y_train_OH = keras.utils.to_categorical(y_train)
y_test_OH = keras.utils.to_categorical(y_test)
y_bis_OH = keras.utils.to_categorical(y_bis)

#Normalisation RGB
X_train_Norm = X_train.astype('float32')/255.0
X_test_Norm = X_test.astype('float32')/255.0
X_bis_Norm = X_bis.astype('float32')/255.0


##Modèle
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu',input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(filters=128,kernel_size=5,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=256,kernel_size=5,padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(28, activation='softmax'))

model.summary()



model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train_Norm, y_train_OH, epochs = 3, batch_size = 64) #Mettre le nombre d'epoch souhaité



##Test modèle

score = model.evaluate(x = X_test_Norm, y = y_test_OH, verbose = 0)
score_bis = model.evaluate(x = X_bis_Norm, y = y_bis_OH, verbose = 0)
print('Précision pour les images test:', round(score[1]*100, 3), '%',)
print('Précision pour les images avec un autre arrière plan:', round(score_bis[1]*100, 3), '%')


## Matrice de confusion

#Pour images test
y_prob = model.predict(X_test_Norm, batch_size = 64, verbose = 0)
y_pred = np.argmax(y_prob,axis=-1)

cm=confusion_matrix(y_test,y_pred)

df_cm = pd.DataFrame(cm, columns=['del', 'space','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],index=['del', 'space','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])


sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True,vmin=0,vmax=300,fmt='d',xticklabels= True,yticklabels= True,annot_kws={"size": 7},cmap="coolwarm")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#Pour images avec un autre fond
y_prob_bis = model.predict(X_bis_Norm, batch_size = 64, verbose = 0)
y_pred_bis = np.argmax(y_prob_bis,axis=-1)


cm_bis=confusion_matrix(y_bis,y_pred_bis)

df_cm_bis = pd.DataFrame(cm_bis, columns=['del', 'space','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],index=['del', 'space','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

sn.set(font_scale=1)
sn.heatmap(df_cm_bis, annot=True,vmin=0,vmax=30,fmt='d',xticklabels= True,yticklabels= True,annot_kws={"size": 7},cmap="coolwarm")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
