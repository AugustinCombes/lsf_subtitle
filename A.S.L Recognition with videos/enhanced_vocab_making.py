import cv2
import time
import uuid
import csv
import mediapipe as mp
import sys
import numpy as np

print('Where do you want to store your images ? (ex:/home/gus/collected_img or C:/Windows/dekstop/collected_img)')
path = input() #where our recorded signs will be saved

print('Type each word you want to add to vocabulary')
print('Press ENTER after each word, if you are done, press ENTER without any word written')
label = input()
labels = []

while label != '' :
	labels.append(label)
	label = input()
print('Vocabulary :', labels)
print('Number of images you want to make for each word ?')
print('(Higher this number, higher the accuracy, standard value around 30 images)')
number_imgs = input()
number_imgs = int(number_imgs)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 2,
    min_detection_confidence = 0.7)

data = np.zeros((len(labels), number_imgs, 42, 3)) #array to stock mediapipe landmarks

for l in range(len(labels)) :
	label = labels[l]
	cap=cv2.VideoCapture(0) #can be a different int depending on the used laptop
	print('Collecting images for {}'.format(label))
	time.sleep(0.5) #time between 2 different words
	valid_img=0 #number of images good enough for mediapipe recognition
	while valid_img < number_imgs :
		ret, frame = cap.read()
		res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		if res.multi_hand_landmarks is None :
			cv2.imshow('frame', frame)
		else :
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
	for label in labels:
		writer.writerow(label)
	writer.writerow(str(len(labels)))
	writer.writerow(str(number_imgs))
