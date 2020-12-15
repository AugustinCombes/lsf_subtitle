#une fois que le csv est prêt de enhanced_vocab_making0.py

#idée : faire une colonne info dans le csv

import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import sys
import mediapipe as mp
import cv2
from collections import defaultdict
from operator import itemgetter

#Gathering informations from the csv file

labels = []
data = np.zeros((len(labels), number_imgs, 42*3))
count_labels, count_imgs = 0,0

with open('mp_data.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	read_l = list(reader)
	statup, number_imgs = read_l[-2], read_l[-1]
	for l in range(statup):
		labels.prepend(read_l[-1*l])

	limit,counter = statup-2-l,0
	for row in reader :
		if counter < limit :
			counter +=1
			if count_imgs==25:
				count_labels+=1
				count_imgs=0
			data[count_labels,count_imgs]=np.array(row)
			count_imgs+=1

data = data.reshape(len(labels)*number_imgs,-1)


#Setting up and training the FFNN

X = [list(data[i]) for i in range(len(data))]
y = []
for it in range(len(labels)):
	y+=[it]*(number_imgs)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) #we can test different solvers, differentes sizes

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=69)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
#print(np.array(X_train).shape) (40,126)
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

def preprocess(t):
	return scaler.transform([t])

def predict(a):
	return clf.predict([a])



#Automatic subtitle of camera input of words in vocabulary

def transform(img): #returns the mp landmarks data of 'img'
	mp_drawing = mp.solutions.drawing_utils
	mp_hands = mp.solutions.hands

	hands = mp_hands.Hands(
	    static_image_mode = True,
	    max_num_hands = 2,
	    min_detection_confidence = 0.7)

	d = np.zeros((42,3))

	res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	if res.multi_hand_landmarks is not None :
		list_landmarks = list(res.multi_hand_landmarks[0].landmark)
		
		if len(res.multi_handedness)==1: #only one hand detected on the img
			for ind in range(21):
				tmp = list_landmarks[ind]
				d[ind] = np.array([tmp.x,tmp.y,tmp.z])
				d[ind+21] = np.array([0,0,0])
		else: #two hands detected on the img
			list_landmarks_bis = list(res.multi_hand_landmarks[1].landmark)
			for ind in range(21):
				tmp1,tmp2 = list_landmarks[ind],list_landmarks_bis[ind]
				d[ind] = np.array([tmp1.x,tmp1.y,tmp1.z])
				d[ind+21] = np.array([tmp2.x,tmp2.y,tmp2.z])
		d = d.reshape(-1)
		return d
	#else :
		#print('Cannot interpret the image') (not useful on preselectionned images)

cap = cv2.VideoCapture(0)

count = 0
list_words = []

import time

while(True):
    ret, frame = cap.read()
    data_2 = transform(frame)
    if data_2 is not None:
    	if count < 10 :
    		count +=1
    		list_words.append(labels[predict(data_2)[0]])
    	else :
    		count = 0
    		d = defaultdict(int)
    		for w in list_words:
    			d[w]+=1
    		x = max(d.items(),key=itemgetter(1))[0]
    		print(x)


    		list_words = []
    	#data_2 = preprocess(data_2)[0] #reshape impossible erreur de type
    		#print(labels[predict(data_2)[0]])
    	#cv2.imshow('frame',data_2)
    else :
    	#cv2.imshow('frame',frame)
    	if cv2.waitKey(1) & 0xFF == ord('q'):
    		break

cap.release()
cv2.destroyAllWindows()
