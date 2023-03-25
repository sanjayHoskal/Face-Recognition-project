import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import time
detector = MTCNN()
cam = cv2.VideoCapture('C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\VID_20210107_102356.mp4')
while (True):
    start=time.time()
    ret, img = cam.read()
    #img = cv2.imread('test.jpg')
    faces = detector.detect_faces(img)# result
#to draw faces on image
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.imshow("frame", cv2.resize(img, (850, 700)))
    end=time.time()
    print(end-start)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break