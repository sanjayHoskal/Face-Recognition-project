import cv2

from tensorflow.keras.models import Sequential #CNN
from tensorflow.keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dropout,Dense
import numpy

def makeModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(48, 48, 1)))  # input layer
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation="softmax"))  # output layer #sigmoid
    model.load_weights('models\\training.hdf5')
    return model



def emoPredict(model,image,gray,faces):
    expressions={0:'Boredom',1:'Confusion',2:'Engagement',3:'Frustration'}

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(200,100,80),2)
        face=gray[y:y+h,x:x+w]
        resize=cv2.resize(face,(48,48))
        expandedimg=numpy.expand_dims(numpy.expand_dims(resize,-1),0)
        predict=model.predict(expandedimg)
        maxexp=int(numpy.argmax(predict))
        finalexp=expressions[maxexp]
        if(maxexp==2):
            return 1,finalexp
        else:
            return 0,finalexp
