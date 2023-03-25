import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
cam = cv2.VideoCapture(0)
while (True):
    ret, img = cam.read()
    #img = cv2.imread('test.jpg')
    #img=img[0:int(20+img.shape[0]/2),0:img.shape[1]]
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #start=time.time()
    #to draw faces on image
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    cv2.imshow("frame", cv2.resize(img, (850, 700)))
    #end=time.time()
    #print(end-start)
#plt.imshow(img)
#plt.show()
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break