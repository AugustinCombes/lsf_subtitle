import cv2
import os
import time
import uuid
import csv
import mediapipe as mp
import sys
import numpy as np

path = '/home/gus/Documents/lsf/collected_img' #where our recorded signs will be saved

labels = ['fermee', 'ouverte'] #our vocabulary

number_imgs = 25 #number of images to collect for each word

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 2,
    min_detection_confidence = 0.7)

data = np.zeros((len(labels), number_imgs, 42, 3)) #array to stock mediapipe landmarks

for l in range(len(labels)) :
	label = labels[l]
	#!mkdir {'home\gus_Documents\lsf\collected_img\\'+label}
	cap=cv2.VideoCapture(0) #can be a different int depending on the used laptop
	print('Collecting images for {}'.format(label))
	time.sleep(5) #time between 2 different words
	valid_img=0 #number of images good enough for mediapipe recognition
	while valid_img < number_imgs :
		ret, frame = cap.read()
		im_name = os.path.join(path, label, '.'+'{}.jpg'.format(str(uuid.uuid1())))
		res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		if res.multi_hand_landmarks is not None :
			list_landmarks = list(res.multi_hand_landmarks[0].landmark)
			for hand_landmarks in res.multi_hand_landmarks :
				mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
			
			if len(res.multi_handedness)==1: #only one hand detected on the img
				for ind in range(21):
					tmp = list_landmarks[ind]
					data[l,valid_img,ind] = np.array([tmp.x,tmp.y,tmp.z])
					data[l,valid_img,ind+21] = np.array([0,0,0])
			else: #two hands detected on the img
				list_landmarks_bis = list(res.multi_hand_landmarks[1].landmark)
				for ind in range(21):
					tmp1,tmp2 = list_landmarks[ind],list_landmarks_bis[ind]
					data[l,valid_img,ind] = np.array([tmp1.x,tmp1.y,tmp1.z])
					data[l,valid_img,ind+21] = np.array([tmp2.x,tmp2.y,tmp2.z])
			valid_img += 1

			cv2.imshow('frame', frame)
			time.sleep(2)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	cap.release()

#data[l,v,ind,coo] contains the coo (x,y,z) of the ind-th landmark of the v-th
#valid image collected for the l-th word of our vocabulary


#We then save the mediapipe data array into a csv file

with open('mp_data.csv', 'w', newline='') as csvfile :
	writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for l in range(len(labels)):
		for v in range(number_imgs):
			writer.writerow(data[l,v].reshape(-1))