import cv2
import time
import matplotlib.pyplot as plt
sampleNum=0
framenum=0
classifier = cv2.CascadeClassifier('haar_files\\haarcascade_frontalface2.xml')
eclassifier = cv2.CascadeClassifier('haar_files\\haarcascade_righteye_2splits.xml')
#img = cv2.imread('test.jpg')
cam = cv2.VideoCapture('C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\VID_20210107_102356.mp4')
while (True):
    ret, img = cam.read()
    faces = classifier.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))# result
#to draw faces on image
    #start=time.time()
    print(len(faces))
    for result in faces:
        x, y, w, h = result
        eyes = eclassifier.detectMultiScale(img[y:y+h,x:x+w])
        print("hi"+str(len(eyes)))
        if(len(eyes)>0):
            a,b,c,d=eyes[0]
            #cv2.rectangle(img, (x+a, y+b), (x+a + c, y+b + d), (0, 255, 0), 2)
        #x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #cv2.imwrite("C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\dataset\\imagesnew\\" + 'name' + "." + 'Id' + '.' + str(sampleNum) + ".jpg", img[y:y + h, x:x + w])
        #sampleNum+=1;
    cv2.imshow("frameHAAR", cv2.resize(img, (850, 700)))
    #end = time.time()
    #print(end-start)
#plt.imshow(img)
#plt.show()
    #cv2.waitKey(0)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break