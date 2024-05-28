import os
import cv2
import numpy as np
import mediapipe as mp
from fastdtw import fastdtw
from utils.processLandmark import *
from utils.dataLog import *

def extractLandmarkFromMasterVideo(masterFolderPath, maxHand, modelComplex, minDetect, minTracking):
    videos, actionName = loadMasterVideo(masterFolderPath)
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
                model_complexity=modelComplex,
                max_num_hands=maxHand,
                min_detection_confidence=minDetect,
                min_tracking_confidence=minTracking,
            )

    for index, videoPath in enumerate(videos):
        cap = cv2.VideoCapture(videoPath)
        frameCount = 1
        while True:
            ret, frame = cap.read()

            frameCount += 1

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            handsResults = hands.process(frame)
            frame.flags.writeable = True

            leftPreProcessedLandmarkList = np.zeros(42)
            rightPreProcessedLandmarkList = np.zeros(42)

            if handsResults.multi_hand_landmarks is not None:
                for handLandmarks, handedness in zip(handsResults.multi_hand_landmarks, handsResults.multi_handedness):
                    # Landmark calculation
                    landmarkList = calculateLandmarkList(frame, handLandmarks)

                    # Changing the Mediapipe result to correct side of hand
                    if handedness.classification[0].index == 0:
                        handedness.classification[0].index = 1
                        handedness.classification[0].label = 'Right'

                        # Conversion to relative coordinates / normalized coordinates
                        rightPreProcessedLandmarkList = preProcessLandmark(landmarkList)

                        loggingMasterCSVNew(masterFolderPath, actionName[index], rightPreProcessedLandmarkList, 'right')

                    else:
                        handedness.classification[0].index = 0
                        handedness.classification[0].label = 'Left'

                        # Conversion to relative coordinates / normalized coordinates
                        leftPreProcessedLandmarkList = preProcessLandmark(landmarkList)

                        # Conversion to relative coordinates / normalized coordinates
                        loggingMasterCSVNew(masterFolderPath, actionName[index], leftPreProcessedLandmarkList, 'left')
                            
            print(videoPath + ' : ' + str(frameCount))
        
        cap.release()

def loadMasterVideo(masterFolderPath):
    videos = []
    actionName = []

    for root, _, file in os.walk(masterFolderPath):
        for file_name in file:
            if file_name.endswith('.mp4'):
                actionName.append(file_name.replace('.mp4', ''))
                videos.append(os.path.join(root, file_name))

    return videos, actionName

def calculateScore(leftMaster, rightMaster, leftAction, rightAction):
    actionDistance = 0
    
    if len(leftMaster) > 10 and len(leftAction) > 10:
        actionDistance += (fastdtw(leftMaster, leftAction))[0]

    if len(rightMaster) > 10 and len(rightAction) > 10:
        actionDistance += (fastdtw(rightMaster, rightAction))[0]

    #print(action_distance)
    score = 100 - (actionDistance / 100)
    if score < 50:
        score = 50

    return score

def calculateScoreNew(master, actual):
    actionDistance = 0
    
    if len(master) >= 10 and len(actual) >= 10:
        actionDistance += (fastdtw(master, actual))[0]
    else:
        return None

    #print(action_distance)
    actionDistance = actionDistance / 100

    return actionDistance