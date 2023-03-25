#!/usr/bin/env python3

import os
import cv2
import sys
import dlib
import argparse
import numpy as np

from drawFace import draw
import reference_world as world
import drowsiness_detection
import faceEmoModel

'''PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image",
                    type=str,
                    help="image location for pose estimation")

parser.add_argument("-f", "--focal",
                    type=float,
                    help="Callibrated Focal Length of the camera")

args = parser.parse_args()'''


def get_pose(model,im,gray,faces):
    res=[]
    #PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
    focal=1.7
    #detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    #im = cv2.imread(image)

    #faces = detector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 0)

    face3Dmodel = world.ref3DModel()

    for face in faces:
        x,y,w,h=face

        shape = predictor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), dlib.rectangle(x,y,x+w,y+h))

        #draw(im, shape)

        refImgPts = world.ref2dImagePoints(shape)

        height, width, channel = im.shape
        focalLength = focal * width
        cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

        mdists = np.zeros((4, 1), dtype=np.float64)

        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(
            face3Dmodel, refImgPts, cameraMatrix, mdists)

        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(
            noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

        # draw nose line
        p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
        cv2.line(im, p1, p2, (110, 220, 0),
                 thickness=2, lineType=cv2.LINE_AA)

        # calculating angle
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        print('*' * 80)
        print("Angle: ", angles)
        # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
        x = np.arctan2(Qx[2][1], Qx[2][2])
        y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1]) + (Qy[2][2] * Qy[2][2])))
        z = np.arctan2(Qz[0][0], Qz[1][0])
        print("AxisX: ", x)
        print("AxisY: ", y)
        print("AxisZ: ", z)
        print('*' * 80)

        gaze = "Looking: "
        if angles[1] < -25:
            gaze += "Left"
            res.append(0)
        elif angles[1] > 25:
            gaze += "Right"
            res.append(0)
        else:
            gaze += "Forward"
            re=1
            finalExp="flop"
            try:
                #re=drowsiness_detection.get_drowsy(model, im, gray, [face])
                re,finalExp = faceEmoModel.emoPredict(model, im, gray, [face])
                print('yes',finalExp)
            except:
                pass
            res.append(re)

        cv2.putText(im, gaze, p2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 3)
    return im,res

'''if __name__ == "__main__":
    main(args.image)'''
