#Automatic subtitle of camera input of words in vocabulary

import sys
from fast_forward_neural_network import predict,preprocess
import mediapipe as mp
import numpy as np
from enhanced_vocab_making.py import infos
import cv2

labels, number_imgs = infos()

#test = cv2.imread('tr.jpg')

def mp_transform(img):
	mp_drawing = mp.solutions.drawing_utils
	mp_hands = mp.solutions.hands

	hands = mp_hands.Hands(
	    static_image_mode = True,
	    max_num_hands = 2,
	    min_detection_confidence = 0.7)

	data = np.zeros((42,3))

	res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	if res.multi_hand_landmarks is not None :
		list_landmarks = list(res.multi_hand_landmarks[0].landmark)
		
		if len(res.multi_handedness)==1: #only one hand detected on the img
			for ind in range(21):
				tmp = list_landmarks[ind]
				data[ind] = np.array([tmp.x,tmp.y,tmp.z])
				data[ind+21] = np.array([0,0,0])
		else: #two hands detected on the img
			list_landmarks_bis = list(res.multi_hand_landmarks[1].landmark)
			for ind in range(21):
				tmp1,tmp2 = list_landmarks[ind],list_landmarks_bis[ind]
				data[ind] = np.array([tmp1.x,tmp1.y,tmp1.z])
				data[ind+21] = np.array([tmp2.x,tmp2.y,tmp2.z])
		data = data.reshape(-1)
		return data
	#else :
		#print('Cannot interpret the image')

cap=cv2.VideoCapture(0)
ret, frame = cap.read()
data = mp_transform(frame)
data = preprocess(data)[0]
print(labels[predict(data)[0]])

#print(predict(data))