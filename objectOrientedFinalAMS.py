import tkinter as tk
from tkinter import Message ,Text
import cv2
import os

import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
from keras.models import load_model
import time
import tkinter.ttk as ttk
import tkinter.font as font
from tkinter import filedialog
import joblib
import head_pose_from_image
import sklearn
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tensorflow.keras.models import Sequential #CNN
from tensorflow.keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dropout,Dense
import faceEmoModel
import drowsiness_detection

primary = tk.Tk()
primary.title("AMS")
primary.geometry("1000x500")
message = tk.Label(primary, text="Emotion Recognition and Attendance Management" ,bg="lightgreen"  ,fg="white"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline'))
message.place(x=200, y=20)

lbl = tk.Label(primary, text="Enter ID",width=20  ,height=2  ,fg="black"  ,bg="white" ,font=('times', 15, ' bold ') )
lbl.place(x=400, y=200)

txt = tk.Entry(primary,width=20  ,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(primary, text="Enter Name",width=20  ,fg="black"  ,bg="white"    ,height=2 ,font=('times', 15, ' bold '))
lbl2.place(x=400, y=300)

txt2 = tk.Entry(primary,width=20  ,bg="white"  ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=315)

model_dd = load_model('models/cnncat2.h5')
model=faceEmoModel.makeModel()
class newAttendance:
    def TakeImages(self):
        harcascadePath = "C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\face_det\\haar_files\\haarcascade_frontalface_default.xml"
        Id=(txt.get())
        name=(txt2.get())
        if(Id.isnumeric() and name.isalpha()):
            cam = cv2.VideoCapture(0)

            detector=cv2.CascadeClassifier(harcascadePath)
            sampleNum=0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    #incrementing sample number
                    sampleNum=sampleNum+1
                    #saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\dataset\\images\\"+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                    #display the frame
                    cv2.imshow('frame',img)
                #wait for 100 miliseconds
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum>200:
                    break
            cam.release()
            cv2.destroyAllWindows()
            res = "Images Saved for ID : " + Id +" Name : "+ name
            row = [Id , name]
            with open('C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\details\\student_details.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            message.configure(text= res)
        else:
            if(Id.isnumeric()):
                res = "Enter Alphabetical Name"
                message.configure(text= res)
            if(name.isalpha()):
                res = "Enter Numeric Id"
                message.configure(text= res)



    def getImagesAndLabels(self,path):
        #get the path of all the files in the folder
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

        #create empth face list
        faces=[]
        #create empty ID list
        Ids=[]
        #now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:
            #loading the image and converting it to gray scale
            pilImage=Image.open(imagePath).convert('L')
            #Now we are converting the PIL image into numpy array
            imageNp=np.array(pilImage,'uint8')
            #getting the Id from the image
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(Id)
        return faces, Ids

    def TrainImages(self):
        recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
        faces,Id = self.getImagesAndLabels("C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\dataset\\images")
        recognizer.train(faces, np.array(Id))
        recognizer.save("C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\dataset\\Trainner.yml")
        res = "Image Trained"#+",".join(str(f) for f in Id)
        message.configure(text= res)

class get_attendance_emotion:

    classifer = joblib.load('C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\models\\cl_gabor_unnormalized_int_2_train.joblib')
    faceDet = cv2.CascadeClassifier("C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\face_det\\haar_files\\haarcascade_frontalface_default.xml")
    
    def __init__(self,filename):
        self.frameNo = 0
        self.attent= []
        self.att = []
        self.studentEmotion = {} # dictionary for storing emotions
        self.presentList = [] #list for storing students
        self.recognizer = cv2.face.LBPHFaceRecognizer_create() #cv2.createLBPHFaceRecognizer() #
        self.recognizer.read("C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\dataset\\Trainner.yml") #reading the trained face identification model
        self.df=pd.read_csv("C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\details\\student_details.csv")#to get id and name from the student list
        self.attend = []
        self.des={}
        self.desdd={}
        self.classifer = joblib.load('C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\models\\cl_gabor_unnormalized_int_2_train.joblib')
        self.faceDet = cv2.CascadeClassifier("C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\face_det\\haar_files\\haarcascade_frontalface_default.xml")
        self.getFrames(filename)


    #-------------------------------------Data structure for storing emotions-----------------------
    def storeEmotions(self,Id,emotion):
        if self.studentEmotion.get(Id,0)==0:
            self.studentEmotion[Id] = {}
            self.studentEmotion[Id][self.frameNo] = emotion
        else:
            self.studentEmotion[Id][self.frameNo] = emotion
    #------------------------------------End------------------------------------



    #-----------------------------------------start---------------------------------------------------
    '''
        * The below function accepts id, image as arguments and performs surf to extract emotions.
        * After extraction of emotions it will call storeEmotions where a nested dictionary will be created to
            store the emotions of each individual identified in the form
            {id:{timeInterval:'emotion,...}...}
    '''



    def build_filters(self):
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 2):
            kern = cv2.getGaborKernel((ksize, ksize), 8.0, theta, 25.0, 15.0, 0, ktype=cv2.CV_32F)

            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters

    def process(self,img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum


    def getEmotions(self,Id,gray): # Emotion detection block
        filters = self.build_filters()
        res = self.process(cv2.resize(gray,(64,64)), filters)
        localarray = np.matrix(np.array(res.ravel(),dtype=np.float32))
        emotionDisplayed = self.classifer.predict(localarray)
        self.storeEmotions(Id,emotionDisplayed[0])



    #------------------------this will recognize the person in the image and record attendence and call getEmotion Function---



    def get_attendance(self,img,gray,faces):
        try:
            img,re = head_pose_from_image.get_pose(model,img,gray,faces)
            c=0
            for i in re:
                if (i==1):
                    c+=1
            self.attent.append((c/len(re))*100)

            #img,res = drowsiness_detection.get_drowsy(model, img, gray, faces)
            #print (res)
            #faces = face
            for face in faces:
                x,y,w,h=face
                #cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
                #gray = gray[y:y+h,x:x+w]
                Id, conf = self.recognizer.predict(gray[y:y+h,x:x+w])
                if (conf < 70):
                    self.attend.append(self.df[self.df["Id"]==Id].values[0])
                    print(Id)
                    name = self.df[self.df["Id"] == Id]['Name'].values
                    faceid=str("ID: "+str(Id)+"\n"+"NAME: "+str(name)[2:-2])

                    self.getEmotions(Id,gray)
                    if Id in self.des.keys():
                        self.des[Id]+=100
                    else:
                        self.des[Id] = 100
                        self.desdd[Id]= 0
                    im_waste,resu=head_pose_from_image.get_pose(model,img,gray,[face])
                    dd = drowsiness_detection.get_drowsy(model_dd, img, gray, [face])
                    if(dd==0):
                        self.desdd[Id]+= 1
                    elif(self.desdd[Id]>1):
                        self.desdd[Id] -= 1
                    if(self.desdd[Id]>3):
                        resu[0]=0
                    print('yea',resu)
                    self.des[Id]+=resu[0]
                else:
                    faceid="unknown"
                cv2.putText(img,faceid, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 50, 255), 2)
                # return

        except:
            pass
        #get attendence, discuss with attendence team for further simplification
    #---------------------------End------------------------------



    '''
        *The below function takes video as input and extracts each frame and checks whether it matches the predefined frames
            as they indicate the time interval of 5 mins.
        *Once verifed it will call get_attendance by passing the matched frame as parameter.
        *timeInterval is used to provide the human readable form of time.
        * frame_no is incremented by 1 and timeInterval by 5min
    '''
        #Below codes are to write the attendence and emotion into respective files


    def write_attendance_emotion(self,studentEmotion,presentList):
        file_name = "emotion_"+str(datetime.date.today())+".csv"
        #specify you own path name, folder name and change the path format to primarys path names
        with open('C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\emotions\\'+file_name,'a') as file:
            row = csv.writer(file)
            for x,y in self.studentEmotion.items():
                row.writerow([x,y])

        attend = "attendence_"+str(datetime.date.today())+".csv"
        self.attend = pd.DataFrame(self.attend,columns = ['Id','Name'])
        self.attend=self.attend.drop_duplicates(subset=['Id'],keep='first')
        self.attend.to_csv("C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\details\\"+attend,index=False,index_label=False,columns=["Id","Name"])
    #----------------------------------End-----------------------------------------------------------


    def getFrames(self,filename):
        #frame_no = 0
        # getting single frame and identifying the face
        cap = cv2.VideoCapture(filename)
        numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        n=numFrames
        while numFrames!=0:
            try:
                ret, img = cap.read()
                #if frame_no in frame_sequence:
                if numFrames%10==0:
                    self.att.append(n-numFrames)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.faceDet.detectMultiScale(gray, 1.07, 4)
                    #self.get_attendance(img,frame_no)#read single frame and send to get_attendance function
                    self.get_attendance(img,gray, faces)#read single frame and send to get_attendance function

                self.frameNo+=1
                numFrames-=1
                print(numFrames)

            except:
                continue
            cv2.imshow("frame", cv2.resize(img, (850, 700)))
            # wait for 100 miliseconds
            if (cv2.waitKey(100) & 0xFF == ord('q')):
                break
            elif (numFrames==1):
                break
        print(self.att,self.attent)
        su=sum(self.attent)
        print(su/len(self.attent))
        print(self.des)
        print('*' * 80)
        print('*' * 80)
        print('*' * 80)
        t=PrettyTable(['ID','Name','Attentiveness','Drowsiness'])
        for x,y in self.des.items():
            a=((y%100)/int(y/100))*100
            if(self.desdd[x]>3):
                sleep="Sleepy"
            else:
                sleep="Awake"
            name=self.df[self.df["Id"]==x]['Name'].values
            t.add_row([x,str(name)[2:-2],a,sleep])
        print(t)
        print('*' * 80)
        print('*' * 80)
        print('*' * 80)
        plt.plot(self.att, self.attent, color='green', linestyle='dashed', linewidth=3,
                 marker='o', markerfacecolor='blue', markersize=12)

        # x-axis label
        plt.ylabel('attentiveness')
        # frequency label
        plt.xlabel('No.of frames')
        # plot title
        plt.title('Class Attention')

        # function to show the plot
        plt.show()
        self.write_attendance_emotion(self.storeEmotions,self.presentList)
        cv2.destroyAllWindows()
    #----------------------------------Global Initilizations ---------------------------------


    #-------------------------------End------------------------------------------------------


     #calling getFrames function execution starts from here


    #------------------------------------Storing the acqurired attendence and emotions-------------------------------


newAttend = newAttendance()

takeImg = tk.Button(primary, text="Take Images", command=newAttend.TakeImages  ,fg="black"  ,bg="white"  ,width=20  ,height=3, activebackground = "green" ,font=('times', 15, ' bold '))
takeImg.place(x=100, y=400)

trainImg = tk.Button(primary, text="Train Images", command=newAttend.TrainImages  ,fg="black"  ,bg="white"  ,width=20  ,height=3, activebackground = "green" ,font=('times', 15, ' bold '))
trainImg.place(x=400, y=400)

def getFileName():
    filename = filedialog.askopenfilename(initialdir="C:\\Users\\OM\\PycharmProjects\\ProjectDemo\\",title="Select the file") #file selection window
    get_attendance_emotion(filename)

select_file = tk.Button(primary, text="Get attendance", command=getFileName  ,fg="black"  ,bg="white"  ,width=20  ,height=3, activebackground = "green" ,font=('times', 15, ' bold '))
select_file.place(x=700, y=400)

'''
quitprimary = tk.Button(primary, text="Quit", command=function6  ,fg="black"  ,bg="white"  ,width=20  ,height=3, activebackground = "green" ,font=('times', 15, ' bold '))
quitprimary.place(x=1100, y=500)
'''
primary.mainloop()
