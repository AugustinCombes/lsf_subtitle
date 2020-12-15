import numpy as np
import mediapipe as mp
import csv
import cv2
import os
from os.path import join

n= #3000 si on traite la base de train et 30 si on traite la base bis

labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','$','@']

number_imgs = n

data = np.zeros((len(labels),number_imgs,42,3))

print('Path=')
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
        res = hands.process(cv2.cvtColor(cv2.imread(join(path,images[c])), cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks is None :
            print('Non reconnu')
        else:
            blank = 255*np.ones(cv2.imread(join(path,images[c])).shape)
            for hand_landmarks in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(blank, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imwrite(r'C:\Users\Louise\Desktop\test_mp_images\annotated_img_'+labels[l]+str(valid_img)+'.png', blank) #Modifier le lien 
        c+=1


