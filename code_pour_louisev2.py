import numpy as np
import mediapipe as mp
import csv
import cv2
import os

labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3']
number_imgs = 3000

data = np.zeros((len(labels),number_imgs,42,3))

print('Met ton path ici louise genre c:// ou je sais pas quoi sur windows')
print('Et met toutes les images (ordonnées, les 3000 a puis les 3000 b etc) dans ce path')
path = input()
images = os.listdir(path)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 2,
    min_detection_confidence = 0.7)

c=0
for l in range(len(labels)):
	for valid_img in range(number_imgs):
		i = cv2.imread(path+images[c])
		c+=1
		
		res = hands.process(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
		if res.multi_hand_landmarks is None :
			print('une image non acceptée de plus ^^')
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

c=0
for l in range(len(labels)):
	for valid_img in range(number_imgs):
		res = hands.process(cv2.cvtColor(cv2.imread(path+images[c]), cv2.COLOR_BGR2RGB))
		if res.multi_hand_landmarks is None :
			print('une image n\'a pas été acceptée')
		else:
			blank = 255*np.ones(cv2.imread(path+images[c]).shape)
			for hand_landmarks in res.multi_hand_landmarks:
				mp_drawing.draw_landmarks(blank, hand_landmarks, mp_hands.HAND_CONNECTIONS)
			cv2.imwrite('annotated_img_'+labels[l]+'png', blank)
		c+=1


with open('mp_data.csv', 'w', newline='') as csvfile :
	writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for l in range(len(labels)):
		for v in range(number_imgs):
			writer.writerow(data[l,v].reshape(-1))
	for label in labels:
		writer.writerow(label)
	writer.writerow(str(len(labels)))
	writer.writerow(str(number_imgs))