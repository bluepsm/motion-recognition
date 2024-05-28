import copy
import itertools
import numpy as np

def calculateLandmarkList(frame, handLandmark):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    landmarkPoint = []

    # Keypoint
    for _, landmark in enumerate(handLandmark.landmark):
        landmark_x = min(int(landmark.x * frameWidth), frameWidth - 1)
        landmark_y = min(int(landmark.y * frameHeight), frameHeight - 1)

        landmarkPoint.append([landmark_x, landmark_y])

    return landmarkPoint

def preProcessLandmark(landmarkList):
    tempLandmarkList = copy.deepcopy(landmarkList)
    fingerTipLandmarkList = []
    fingerTipIndex = [4, 8, 12, 16, 20]

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmarkPoint in enumerate(tempLandmarkList):
        if index == 0:
            base_x, base_y = landmarkPoint[0], landmarkPoint[1]

        tempLandmarkList[index][0] = tempLandmarkList[index][0] - base_x
        tempLandmarkList[index][1] = tempLandmarkList[index][1] - base_y

        if index in fingerTipIndex:
            fingerTipLandmarkList.append(tempLandmarkList[index])
    
    # Convert to a one-dimensional list
    tempLandmarkList = list(itertools.chain.from_iterable(tempLandmarkList))
    fingerTipLandmarkList = list(itertools.chain.from_iterable(fingerTipLandmarkList))

    # Normalization
    def normalize_(n):
        return n / maxValue

    maxValue = max(list(map(abs, tempLandmarkList)))
    tempLandmarkList = list(map(normalize_, tempLandmarkList))

    maxValue = max(list(map(abs, fingerTipLandmarkList)))
    fingerTipLandmarkList = list(map(normalize_, fingerTipLandmarkList))

    return tempLandmarkList, fingerTipLandmarkList

def preProcessLandmark_Graph(landmarkList):
    allAngles = []
    avgAngles = 0
    joint_list = [[4,3,2], [8,7,6]]
    for joint in joint_list:
        """ a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
        b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
        c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord """

        a = landmarkList[joint[0]] # First coord
        b = landmarkList[joint[1]] # Second coord
        c = landmarkList[joint[2]] # Third coord
        
        radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        
        allAngles.append(angle)

    avgAngles = sum(allAngles) / len(allAngles)

    return avgAngles