import cv2
import os
from keras.models import load_model
import numpy as np
# from pygame import mixer
import time

'''mixer.init()
sound = mixer.Sound('alarm.wav')'''

# face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')

'''cap = cv2.VideoCapture(0) #to access the camera 

count=0
score=0
thicc=2
rpred=[99]
lpred=[99]'''

'''With a webcam, we will take images as input. So to access the webcam, we made an infinite loop that will capture each frame.'''


def get_drowsy(model, img, gray1, faces):
    res = 1
    count = 0
    rpred = [99]
    lpred = [99]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    leye = cv2.CascadeClassifier('face_det\haar_files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('face_det\haar_files\haarcascade_righteye_2splits.xml')

    #lbl = ['Close', 'Open']
    #path = os.getcwd()
    '''ret, frame = cap.read()   #will read each frame and we store the image in a frame variable.
    height,width = frame.shape[:2]'''

    '''gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #OpenCV algorithm for object detection takes gray images in the input.

    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)) #Face detection'''

    # cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    #x,y,w,h=faces
    for (a,b,c,d) in faces:
        gray = gray1[b:b + d, a:a + c]

        left_eye = leye.detectMultiScale(gray)  # Left eye detection
        right_eye = reye.detectMultiScale(gray)

        '''r_eye = frame[y:y + h, x:x + w]
        count = count + 1'''
        if (len(right_eye) > 0):
            x, y, w, h = right_eye[0]
            r_eye = gray[y:y + h, x:x + w]
            # r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))  # My model is traing on 24*24 images
            r_eye = r_eye / 255  # Normalization so the model can works efficiently
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = model.predict_classes(r_eye)
            '''if (rpred[0] == 1):
                lbl = 'Open'
            if (rpred[0] == 0):
                lbl = 'Closed'''
        if (len(left_eye) > 0):
            x, y, w, h = left_eye[0]
            l_eye = gray[y:y + h, x:x + w]
            '''l_eye = frame[y:y + h, x:x + w]
            count = count + 1'''
            # l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = model.predict_classes(l_eye)
            '''if (lpred[0] == 1):
                lbl = 'Open'
            if (lpred[0] == 0):
                lbl = 'Closed'''

        if (rpred[0] == 0 and lpred[0] == 0):
            # score = score + 1
            #cv2.putText(img, "Closed", (a-10, b-10), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            res=0

            # if(rpred[0]==1 or lpred[0]==1):
        else:
            # score = score - 1
            #cv2.putText(img, "Open", (a-10, b-10), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            res=1

    return res


