import numpy as np
import cv2

def getROICenter(frame, handLandmark):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    landmarkArray = np.empty((0, 2), int)

    for _, landmark in enumerate(handLandmark.landmark):
        landmark_x = min(int(landmark.x * frameWidth), frameWidth - 1)
        landmark_y = min(int(landmark.y * frameHeight), frameHeight - 1)

        landmarkPoint = [np.array((landmark_x, landmark_y))]

        landmarkArray = np.append(landmarkArray, landmarkPoint, axis=0)

    x, y, w, h = cv2.boundingRect(landmarkArray)

    cx = x + w // 2
    cy = y + h // 2

    return [cx, cy]

def calculateROI(frame, handLandmark, roiRadius):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    landmarkArray = np.empty((0, 2), int)

    for _, landmark in enumerate(handLandmark.landmark):
        landmark_x = min(int(landmark.x * frameWidth), frameWidth - 1)
        landmark_y = min(int(landmark.y * frameHeight), frameHeight - 1)

        landmarkPoint = [np.array((landmark_x, landmark_y))]

        landmarkArray = np.append(landmarkArray, landmarkPoint, axis=0)

    x, y, w, h = cv2.boundingRect(landmarkArray)

    cx = x + w // 2
    cy = y + h // 2
    #cr = max(w, h) // 2

    start_x = cx - roiRadius
    end_x = cx + roiRadius
    start_y = cy - roiRadius
    end_y = cy + roiRadius

    return [start_x, end_x, start_y, end_y]

def checkROIOverlap(leftROI, rightROI):
    start_x = max(min(leftROI[0], leftROI[1]), min(rightROI[0], rightROI[1]))
    start_y = max(min(leftROI[2], leftROI[3]), min(rightROI[2], rightROI[3]))
    end_x = min(max(leftROI[0], leftROI[1]), max(rightROI[0], rightROI[1]))
    end_y = min(max(leftROI[2], leftROI[3]), max(rightROI[2], rightROI[3]))
    if start_x < end_x and start_y < end_y:
        return True
    else:
        return False

def keepROIInFrame(frame, roi, roiRadius):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]

    if roi[0] < 0:
        roi[0] = 0
        roi[1] = roiRadius * 2
    
    if roi[1] > frameWidth:
        roi[0] = frameWidth - (roiRadius * 2)
        roi[1] = frameWidth

    if roi[2] < 0:
        roi[2] = 0
        roi[3] = roiRadius * 2

    if roi[3] > frameHeight:
        roi[2] = frameHeight - (roiRadius * 2)
        roi[3] = frameHeight

    return roi