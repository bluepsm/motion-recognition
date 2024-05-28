import os
import cv2
import numpy as np #For working with array in program
import mediapipe as mp

from utils.score import * #Mehod for process or calculate score from master data
from utils.dataLog import * #Method for logging and save data
from utils.processLandmark import * #Method for processing MediaPipe hand result

from PyQt6 import QtCore #For sending Pyqt signal to GUI for trigger some method or function


class processMasterData(QtCore.QThread):
    #region 'Initializing Pyqt signal'
    progress = QtCore.pyqtSignal(list)
    masterGraph = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()
    #endregion

    def __init__(self, programRoot):
        super().__init__()
        self.programRoot = programRoot

        #region 'Initializing variables'
        #Master video folder path
        self.masterFolderPath = None

        #MediaPipe
        self.modelComplex = None
        self.maxHand = None
        self.minDetect = None
        self.minTracking = None

        #Graph data
        self.graphData = 0
        self.graph = {}
        #endregion

    def run(self):
        #Collect video paths and actions from folder path
        videos, actionName = loadMasterVideo(self.masterFolderPath)

        #Delete old data if exists
        if os.path.isdir(f'{self.programRoot}/data'):
            for file in os.listdir(f'{self.programRoot}/data'):
                if file.endswith('.csv'):
                    os.remove(f'{self.programRoot}/data/{file}')

        #Create graph data dictionary with empty value
        for name in actionName:
            self.graph.update({name : []})

        #MediaPipe model setting
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(
                    model_complexity=self.modelComplex,
                    max_num_hands=self.maxHand,
                    min_detection_confidence=self.minDetect,
                    min_tracking_confidence=self.minTracking,
                )

        #Iterate all videos in master folder
        for index, videoPath in enumerate(videos):
            cap = cv2.VideoCapture(videoPath)
            totalFrame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frameCount = 0
            startLogging = False
            while True:
                ret, frame = cap.read()

                frameCount += 1

                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                handsResults = hands.process(frame)
                frame.flags.writeable = True

                leftPreProcessedLandmarkList = np.zeros(10)
                rightPreProcessedLandmarkList = np.zeros(10)

                self.graphData = 0

                if handsResults.multi_hand_landmarks is not None:
                    startLogging = True
                    for handLandmarks, handedness in zip(handsResults.multi_hand_landmarks, handsResults.multi_handedness):
                        # Landmark calculation
                        landmarkList = calculateLandmarkList(frame, handLandmarks)

                        # Changing the Mediapipe result to correct side of hand
                        if handedness.classification[0].index == 0:
                            handedness.classification[0].index = 1
                            handedness.classification[0].label = 'Right'

                            # Conversion to relative coordinates / normalized coordinates
                            _, rightPreProcessedLandmarkList = preProcessLandmark(landmarkList)

                            self.graphData += sum(rightPreProcessedLandmarkList) / len(rightPreProcessedLandmarkList)

                        else:
                            handedness.classification[0].index = 0
                            handedness.classification[0].label = 'Left'

                            # Conversion to relative coordinates / normalized coordinates
                            _, leftPreProcessedLandmarkList = preProcessLandmark(landmarkList)
                            
                            self.graphData += sum(leftPreProcessedLandmarkList) / len(leftPreProcessedLandmarkList)
                
                if startLogging:
                    loggingMasterCSVNew(self.programRoot, actionName[index], rightPreProcessedLandmarkList, 'right')
                    loggingMasterCSVNew(self.programRoot, actionName[index], leftPreProcessedLandmarkList, 'left')
                
                    self.graphData = float('{:.2f}'.format(self.graphData))
                    self.graph[actionName[index]].append(self.graphData)

                #print(videoPath + ' : ' + str(frameCount))
                self.progress.emit([actionName[index], totalFrame, frameCount])
            
            cap.release()

        self.masterGraph.emit(self.graph)
        self.finished.emit()
        self.stop()

    def stop(self):
        self.quit()
