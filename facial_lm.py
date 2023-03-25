import dlib
import cv2
import matplotlib.pyplot as plt
detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture('C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\VID_20210107_102356.mp4')
while (True):
    ret, img = cam.read()
    #img = cv2.imread('test.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1) # result
#to draw faces on image
    for result in faces:
        print(result)
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    cv2.imshow("frameDLIB", cv2.resize(img, (850, 700)))
#plt.imshow(img)
#plt.show()
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break