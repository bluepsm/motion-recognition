import os
import cv2
import csv  #For reading .csv file
import copy #For deep copy of frame data
import numpy as np #For working with array in program
import pandas as pd #For convert status dictionary to dataframe that used for updating the status table on GUI
import mediapipe as mp
from collections import deque #For store TensorflowLite predicted results in que before making final result
from posixpath import basename #For extract program root path

from utils.processLandmark import * #Method for processing MediaPipe hand result
from utils.draw import * #Method for draw text or box on video frame
from utils.roi import * #Method for processing ROI area
from utils.dataLog import * #Method for logging and save data
from utils.score import * #Mehod for process or calculate score from master data

from utils import FPS #Get FPS
from utils import VideoClassifier #TensorLite video classification Interpreter

from PyQt6 import QtCore, QtGui #For sending Pyqt signal to GUI for trigger some method or function


class videoThread(QtCore.QThread):
    #region 'Initializing Pyqt signal'
    mainFrameUpdate = QtCore.pyqtSignal(QtGui.QImage)
    leftROIFrameUpdate = QtCore.pyqtSignal(QtGui.QImage)
    rightROIFrameUpdate = QtCore.pyqtSignal(QtGui.QImage)
    bothROIFrameUpdate = QtCore.pyqtSignal(QtGui.QImage)
    videoIndexUpdate = QtCore.pyqtSignal(str)
    videoFPSUpdate = QtCore.pyqtSignal(int)
    actionUpdate = QtCore.pyqtSignal(str)
    scoreUpdate = QtCore.pyqtSignal(object)
    timeUpdate = QtCore.pyqtSignal(float)
    totalTimeUpdate = QtCore.pyqtSignal(float)
    processTableDataFrameUpdate = QtCore.pyqtSignal(object)
    forceStopRecord = QtCore.pyqtSignal()
    recordStatusUpdate = QtCore.pyqtSignal(str)
    startPlot = QtCore.pyqtSignal(str)
    stopPlot = QtCore.pyqtSignal()
    updatePlot = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(Exception)
    #endregion
    
    def __init__(self, source, programRoot):
        super(videoThread, self).__init__()
        self.programRoot = programRoot

        #Set threading status to active
        self.threadActive = True

        #region 'Initializing variables'
        #region 'Video display'
        self.mainDisplayWidth = None
        self.mainDisplayheight = None
        self.leftROIDisplayWidth = None
        self.leftROIDisplayheight = None
        self.rightROIDisplayWidth = None
        self.rightROIDisplayheight = None
        self.bothROIDisplayWidth = None
        self.bothROIDisplayheight = None
        #endregion

        #region 'Video source'
        self.feedFromCamera = False
        self.source = source
        self.videoPath = []
        self.currentVideo = 0
        #endregion

        #region 'Video interact'
        self.videoInterrupt = False
        self.videoPause = False
        #endregion

        #region 'Program path'
        self.masterLabelPath = os.path.join(programRoot, 'classes label')
        self.modelPath = None
        self.labelPath = None
        self.masterPath = None
        self.saveScorePath = None
        self.saveRecordPath = None
        #endregion

        #region 'MediaPipe model'
        self.mpMaxHand = None
        self.mpModelComplex = None
        self.mpMinDetect = None
        self.mpMinTrack = None
        self.roiSize = None
        self.combineROISize = None
        self.mpChanged = False
        self.mpHands = mp.solutions.hands
        #endregion

        #region 'TensorflowLite model'
        self.modelWindowSize = None
        self.modelConfidence = None
        self.inputRatio = None
        self.tfChanged = False
        #endregion

        #region 'Program status'
        self.detectMode = False
        self.recordMode = False
        self.startRecord = False
        self.stopRecord = False
        self.recording = False
        #endregion

        #region 'User requirement'
        self.saveVideo = False
        self.saveVideoROI = False
        self.saveCSV = False
        self.saveImage = False
        self.saveScore = False
        #endregion

        #region 'Video property'
        self.cvFpsCalc = FPS(bufferLen=10)
        self.frameCount = 0
        self.subStartFrame = None
        self.subEndFrame = None
        #endregion

        #region 'ROI area data'
        self.dictLeftRoi = {}
        self.dictRightRoi = {}
        self.dictBothRoi = {}
        self.leftRoi = None
        self.rightRoi = None
        self.bothRoi = None
        #endregion
        
        #region 'Times'
        self.dtime = 0
        self.dwellTime = 0
        self.totalDtime = 0
        self.totalDwellTime = 0
        self.lastestDwellTime = 0
        self.lastestTotalDwellTime = 0
        self.dwellRunning = False
        self.totalDwellRunning = False
        self.timeRecord = {}
        #endregion

        #region 'Score'
        self.recordingDTWCoord = False
        self.currentActionRecording = None
        self.dtwCoordLeft = {}
        self.dtwCoordRight = {}
        self.haveMaster = False
        self.master = {}
        self.score = None
        self.scoreRecord = {}
        #endregion

        #region 'Action state'
        self.action = -1
        self.actionText = None
        self.predictedAction = []
        self.boolActionState = {}
        self.boolHistoryActionState = {}
        self.processTableViewColumn = []
        #endregion

        #region 'Graph data'
        self.graphData = None
        self.temp_graphData = None
        self.temp_landmark_left = None
        self.temp_landmark_right = None
        #endregion
        #endregion

    def run(self):
        def videoCap(source):
            #If threading is not active then exit this function
            if not self.threadActive:
                pass

            cap = cv2.VideoCapture(source)

            #Get dimension of video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            #Set name for record mode output
            if not self.feedFromCamera:
                video_name = (os.path.splitext(basename(source)))
                video_name = ((video_name[0].split('\\', 1))[1])
            else:
                video_name = f'camera{source}_{datetime.datetime.now().date()}'

            #Set frame index to 0
            self.frameCount = 0
            
            #Get total frame of video
            m_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            #While loop by threading status for iterate all frames in video
            while self.threadActive:

                #If user click PREV. VIDEO or NEXT VIDEO button on GUI
                if self.videoInterrupt:
                    #Because user interrupted video from playing normally. All data and state will be reset
                    #for prevent program messed up it's current state and data
                    self.subStartFrame = None
                    self.subEndFrame = None

                    self.dictLeftRoi = self.dictLeftRoi.fromkeys(self.dictLeftRoi, None)
                    self.dictRightRoi = self.dictRightRoi.fromkeys(self.dictRightRoi, None)
                    self.dictBothRoi = self.dictBothRoi.fromkeys(self.dictBothRoi, None)
                    
                    self.dtime = 0
                    self.dwellTime = 0
                    self.totalDtime = 0
                    self.totalDwellTime = 0
                    self.lastestDwellTime = 0
                    self.lastestTotalDwellTime = 0
                    self.dwellRunning = False
                    self.totalDwellRunning = False
                    self.timeRecord = self.timeRecord.fromkeys(self.timeRecord, None)

                    self.predictedAction = []
                    self.score = None
                    self.scoreRecord = self.scoreRecord.fromkeys(self.scoreRecord, None)
                    
                    self.boolActionState = self.boolActionState.fromkeys(self.boolActionState, False)
                    self.boolHistoryActionState = self.boolHistoryActionState.fromkeys(self.boolHistoryActionState, 'PENDING')

                    self.leftRoi = None
                    self.rightRoi = None
                    self.bothRoi = None

                    self.recordingDTWCoord = False
                    self.currentActionRecording = None
                    self.dtwCoordLeft = self.dtwCoordLeft.fromkeys(self.dtwCoordLeft, [])
                    self.dtwCoordRight = self.dtwCoordRight.fromkeys(self.dtwCoordRight, [])

                    self.graphData = None
                    self.temp_graphData = None
                    self.temp_landmark_left = None
                    self.temp_landmark_right = None

                    tableDataFrame = pd.DataFrame.from_dict([self.boolHistoryActionState, self.timeRecord, self.scoreRecord])
                    tableDataFrame.columns = self.processTableViewColumn
                    tableDataFrame.index = ['Status', 'Time(sec)', 'Score']
                    tableDataFrame.fillna('',inplace=True)

                    self.processTableDataFrameUpdate.emit(tableDataFrame)

                    #After reset all data and state then break current loop to go next or previous video
                    break
                
                #If user click PAUSE button on GUI
                while self.videoPause:
                    #If user stop recording in record mode while video is paused
                    #have to force sending signal for immediate stop the record with no need to click the RESUME button 
                    if self.recording and self.stopRecord:
                        self.stopRecord = False

                        #Send signal to GUI for reset button or status text on record mode page after video record
                        #is forced to stop in this thread
                        self.forceStopRecord.emit()

                        self.recording = False

                        #Set end frame index to current frame index
                        subEndFrame = self.frameCount
                        self.recordStatusUpdate.emit(f'{self.actionText} : Stop recorded.')
                        print('Frame {}: Action {} stop recording.'.format(self.frameCount ,self.actionText))
                        if subEndFrame != subStartFrame:
                            #Save video or image if frame index of start record and stop record is not the same index
                            print('\nlandmark points and angles saved!')
                            if self.saveVideo or self.saveVideoROI or self.saveImage:
                                    extractClipAndImage(self.actionText, subStartFrame, subEndFrame, source, video_name, self.dictLeftRoi, self.dictRightRoi, 
                                                        self.dictBothRoi, self.bbox_radius, self.combine_bbox_radius, self.saveVideo, self.saveVideoROI, self.saveImage, self.saveRecordPath)
                        #After save video or image, then clear ROI data for next record
                        self.dictLeftRoi.clear()
                        self.dictRightRoi.clear()
                        self.dictBothRoi.clear()
                    
                    #Use OpenCV waitKey function to pause and stay on this while loop
                    cv2.waitKey(-1)

                #Changing MediaPipe setting in real time with no need to run program over again
                if self.mpChanged:
                    self.hands = self.mpHands.Hands(
                        model_complexity=self.mpModelComplex,
                        max_num_hands=self.mpMaxHand,
                        min_detection_confidence=self.mpMinDetect,
                        min_tracking_confidence=self.mpMinTrack,
                    )
                    self.bbox_radius = self.roiSize
                    self.combine_bbox_radius = self.combineROISize
                    self.mpChanged = False

                #Changing TensorflowLite setting in real time with no need to run program over again
                if self.detectMode and self.tfChanged:
                    self.confidence = self.modelConfidence
                    self.windowSize = self.modelWindowSize
                    self.predictedProbabilitiesDeque = deque(maxlen = self.windowSize)
                    self.tfChanged = False

                #Get FPS
                fps = self.cvFpsCalc.get()

                #Increase frame index
                self.frameCount += 1

                #Force program to stop record, In case user still recording when video is ended
                if self.frameCount == m_frame_count and self.recording:
                    #Send signal to GUI for reset button or status text on record mode page after video record
                    #is forced to stop in this thread
                    self.forceStopRecord.emit()

                    self.recording = False

                    #Set end frame index to current frame index
                    subEndFrame = self.frameCount
                    self.recordStatusUpdate.emit(f'{self.actionText} : Stop recorded.')
                    print('Frame {}: Action {} stop recording.'.format(self.frameCount ,self.actionText))
                    if subEndFrame != subStartFrame:
                        #Save video or image if frame index of start record and stop record is not the same index
                        print('\nlandmark points and angles saved!')
                        if self.saveVideo or self.saveVideoROI or self.saveImage:
                                extractClipAndImage(self.actionText, subStartFrame, subEndFrame, source, video_name, self.dictLeftRoi, self.dictRightRoi, 
                                                    self.dictBothRoi, self.bbox_radius, self.combine_bbox_radius, self.saveVideo, self.saveVideoROI, self.saveImage, self.saveRecordPath)
                    
                    #After save video or image, then clear ROI data for next record
                    self.dictLeftRoi.clear()
                    self.dictRightRoi.clear()
                    self.dictBothRoi.clear()

                #If user click record button in record mode page then start/stop the record
                #Start recording
                if self.recordMode and self.startRecord:
                    self.startRecord = False
                    self.recording = True

                    #Set start frame index to current frame index
                    subStartFrame = self.frameCount
                    self.recordStatusUpdate.emit(f'{self.actionText} : Start recording...')
                    print('\nFrame {}: Action {} start recording...'.format(self.frameCount ,self.actionText))

                #Stop recording and save data
                if self.recordMode and self.recording and self.stopRecord:
                    self.stopRecord = False
                    self.recording = False

                    #Set end frame index to current frame index
                    subEndFrame = self.frameCount
                    self.recordStatusUpdate.emit(f'{self.actionText} : Stop recorded.')
                    print('Frame {}: Action {} stop recording.'.format(self.frameCount ,self.actionText))
                    if subEndFrame != subStartFrame:
                        #Save video or image if frame index of start record and stop record is not the same index
                        print('\nlandmark points and angles saved!')
                        if self.saveVideo or self.saveVideoROI or self.saveImage:
                            extractClipAndImage(self.actionText, subStartFrame, subEndFrame, source, video_name, self.dictLeftRoi, self.dictRightRoi, 
                                                self.dictBothRoi, self.bbox_radius, self.combine_bbox_radius, self.saveVideo, self.saveVideoROI, self.saveImage, self.saveRecordPath)
                    
                    #After save video or image, then clear ROI data for next record
                    self.dictLeftRoi.clear()
                    self.dictRightRoi.clear()
                    self.dictBothRoi.clear()
                            
                #Get video frame
                ret, frame = cap.read()

                #Break from loop if no frame to grab 
                if not ret:
                    break
                
                #Deep copy of frame to use for displaying on GUI. The original frame will be use for processing
                debug_frame = copy.deepcopy(frame)

                #Censored timestamp that display on right bottom corner of frame before sending frame to TensorflowLite model
                #for help model predict more accurate
                cv2.rectangle(frame, (width - 75, height - 20), (width, height), (0, 0, 0), -1)

                #After censored timestamp
                #then deep copy of frame to use for processing ROI area and displaying on GUI
                debug_frame_roi = copy.deepcopy(frame)

                #Convert format of frame from BGR to RGB, because MediaPipe model input have RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #Disable writeable flag to help program process faster
                frame.flags.writeable = False

                #Send frame to MediaPipe model to make detection of hands in frame
                hands_results = self.hands.process(frame)
                
                #Enable writeable flag again after finish MediaPipe process 
                frame.flags.writeable = True

                #Initializing ROI
                left_bbox = None
                right_bbox = None
                both_bbox = None

                #Initializing MediaPipe landmark list
                left_pre_processed_landmark_list = np.zeros(10)
                right_pre_processed_landmark_list = np.zeros(10)

                #Initializing ROI check
                is_overlap = False
                roi_detected = {'left' : False, 'right' : False, 'both' : False}

                #Initializing graph data
                self.graphData = 0

                #If MediaPipe detect any hand in video frame
                if hands_results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):                        
                        #Landmark calculation
                        landmark_list = calculateLandmarkList(debug_frame, hand_landmarks)

                        #Changing the Mediapipe result to correct side of hand
                        if handedness.classification[0].index == 0:
                            #If 'left' change to 'right'
                            handedness.classification[0].index = 1
                            handedness.classification[0].label = 'Right'
    
                            #Conversion to relative coordinates / normalized coordinates
                            _, right_pre_processed_landmark_list = preProcessLandmark(landmark_list)

                            #Calculate data for plotting the graph
                            self.graphData += sum(right_pre_processed_landmark_list) / len(right_pre_processed_landmark_list)

                            #ROI calculation
                            right_bbox = calculateROI(debug_frame, hand_landmarks, self.bbox_radius)

                            #Change ROI state
                            roi_detected['right'] = True
                        else:
                            #If 'right' change to 'left'
                            handedness.classification[0].index = 0
                            handedness.classification[0].label = 'Left'

                            #Conversion to relative coordinates / normalized coordinates
                            _, left_pre_processed_landmark_list = preProcessLandmark(landmark_list)
                            
                            #Calculate data for plotting the graph
                            self.graphData += sum(left_pre_processed_landmark_list) / len(left_pre_processed_landmark_list)

                            #ROI calculation
                            left_bbox = calculateROI(debug_frame, hand_landmarks, self.bbox_radius)

                            #Change ROI state
                            roi_detected['left'] = True
                        
                        #Save CSV dataset if user require program to record CSV in record mode
                        if self.recording and self.saveCSV:
                            #Store landmarks from left and right hand to one single landmark
                            pre_processed_landmark_list = np.concatenate([left_pre_processed_landmark_list, right_pre_processed_landmark_list])
                            
                            #Save landmark list
                            loggingCSV(self.action, self.actionText, pre_processed_landmark_list, self.saveRecordPath)

                        #Checking if ROI overlaped
                        if right_bbox is not None and left_bbox is not None:
                            is_overlap = checkROIOverlap(left_bbox, right_bbox)
                        
                        #Draw hand lamdmarks on frame
                        debug_frame = drawLandmark(debug_frame, landmark_list)

                    #Draw ROI area box
                    if is_overlap:
                        #If 2 ROI area overlaped then marge into 1 ROI area and draw box around
                        debug_frame, both_bbox = drawROIBothHand(debug_frame, left_bbox, right_bbox, self.combine_bbox_radius)
                        
                        #If hands is founded on the edge of frame, We have to calculate new position of ROI box
                        #to keep it stay in frame
                        self.bothRoi = keepROIInFrame(debug_frame, both_bbox, self.combine_bbox_radius)
                        self.leftRoi = keepROIInFrame(debug_frame, left_bbox, self.bbox_radius)
                        self.rightRoi = keepROIInFrame(debug_frame, right_bbox, self.bbox_radius)
                        roi_detected['both'] = True
                    else:
                        #If 2 ROI area not overlaped then draw each hands individual
                        if right_bbox is not None:
                            #Draw box around ROI area
                            debug_frame = drawROIEachHand(debug_frame, right_bbox, 'Right')
                            
                            #If hand is founded on the edge of frame, We have to calculate new position of ROI box
                            #to keep it stay in frame
                            self.rightRoi = keepROIInFrame(debug_frame, right_bbox, self.bbox_radius)
                        
                        if left_bbox is not None:
                            #Draw box around ROI area
                            debug_frame = drawROIEachHand(debug_frame, left_bbox, 'Left')
                            
                            #If hand is founded on the edge of frame, We have to calculate new position of ROI box
                            #to keep it stay in frame
                            self.leftRoi = keepROIInFrame(debug_frame, left_bbox, self.bbox_radius)
                    
                    #If in detect mode
                    if self.detectMode:
                        #Send ROI data to TensorflowLite model
                        #Sending number of frames(%2 = 50%, %3 = 66%, %3.5 = 85%)
                        if self.inputRatio == 1 or (self.frameCount % self.inputRatio != 0):
                            #If have merged ROI
                            #then send this ROI area to TensorflowLite model
                            if roi_detected['both'] is True:
                                #Extract only ROI area from original frame to new frame
                                predict_roi_both = debug_frame_roi[self.bothRoi[2] : self.bothRoi[3], self.bothRoi[0] : self.bothRoi[1]]
                                
                                #Resizing ROI frame to the same of TensorflowLite model input size
                                predict_roi_both = cv2.resize(predict_roi_both, (32, 32), interpolation=cv2.INTER_AREA)

                                #Normalizing ROI frame
                                normalized_roi_both = predict_roi_both / 255

                                #Disable writable flag for faster process
                                normalized_roi_both.flags.writeable = False
                                
                                #TensorflowLite model
                                predicted_probabilities = self.videoClassifier(normalized_roi_both)

                                #Enable writable flag after finish Tensorflow process
                                normalized_roi_both.flags.writeable = True
                            else:
                                #If ROI not overlaped
                                #then send individual ROI area to TensorflowLite model
                                if roi_detected['left'] is True:
                                    #Extract only ROI area from original frame to new frame
                                    predict_roi_left = debug_frame_roi[self.leftRoi[2] : self.leftRoi[3], self.leftRoi[0] : self.leftRoi[1]]
                                    
                                    #Resizing ROI frame to the same of TensorflowLite model input size
                                    predict_roi_left = cv2.resize(predict_roi_left, (32, 32), interpolation=cv2.INTER_AREA)
                                    
                                    #Normalizing ROI frame
                                    normalized_roi_left = predict_roi_left / 255

                                    #Disable writable flag for faster process
                                    normalized_roi_left.flags.writeable = False

                                    #TensorflowLite model
                                    predicted_probabilities = self.videoClassifier(normalized_roi_left)

                                    #Enable writable flag after finish Tensorflow process
                                    normalized_roi_left.flags.writeable = True
                                
                                if roi_detected['right'] is True:
                                    #Extract only ROI area from original frame to new frame
                                    predict_roi_right = debug_frame_roi[self.rightRoi[2] : self.rightRoi[3], self.rightRoi[0] : self.rightRoi[1]]
                                    
                                    #Resizing ROI frame to the same of TensorflowLite model input size
                                    predict_roi_right = cv2.resize(predict_roi_right, (32, 32), interpolation=cv2.INTER_AREA)
                                    
                                    #Normalizing ROI frame
                                    normalized_roi_right = predict_roi_right / 255
                                    
                                    #Disable writable flag for faster process
                                    normalized_roi_right.flags.writeable = False
                                    
                                    #TensorflowLite model
                                    predicted_probabilities = self.videoClassifier(normalized_roi_right)
                                    
                                    #Enable writable flag after finish Tensorflow process
                                    normalized_roi_right.flags.writeable = True

                            #Get prediction from TensorflowLite
                            if len(predicted_probabilities) > 0:
                                #Appending predicted label probabilities to the deque object
                                self.predictedProbabilitiesDeque.append(predicted_probabilities)

                                #Assuring that the Deque is completely filled before starting the averaging process
                                if len(self.predictedProbabilitiesDeque) == self.windowSize:
                                    #Converting Predicted Labels Probabilities Deque into Numpy array
                                    predicted_probabilities_np = np.array(self.predictedProbabilitiesDeque)

                                    #Calculating Average of Predicted Labels Probabilities Column Wise 
                                    predicted_probabilities_averaged = predicted_probabilities_np.mean(axis = 0) 

                                    #Filter only prediction that have probability value equal or larger than confidence value
                                    if np.amax(predicted_probabilities_averaged) >= self.confidence:
                                        #Converting the predicted probabilities into labels by returning the index of the maximum value.
                                        predicted_label = np.argmax(predicted_probabilities_averaged)
                                        
                                        #Get the prediction result
                                        self.predictedAction = [self.actionLabels[predicted_label], np.amax(predicted_probabilities_averaged)]

                                        #region 'States check'
                                        if len(self.predictedAction) > 0:

                                            #Check if all actios status is 'COMPLETED' 
                                            if all(value == 'COMPLETED' for value in self.boolHistoryActionState.values()):
                                                #Then reset all action for do the new action detect loop
                                                for key in self.boolHistoryActionState:
                                                    self.boolHistoryActionState[key] = 'PENDING'

                                            for key in self.boolActionState:
                                                if key == predicted_label:
                                                    #If predict action from Tensorflow is not 'Idle'
                                                    if key != 0:

                                                        #Check if this current predict action is already in process or not
                                                        if self.boolHistoryActionState[key] == 'PENDING':
                                                            #Check if previous action is already in process or not
                                                            if int(key) > 1 and self.boolHistoryActionState[key - 1] == 'IN PROCESS':
                                                                #If current action is in waiting state and previous action is in process state
                                                                #it mean operator is working in the correct sequence
                                                                #and program can changing previous action from 'IN PROCESS' state to 'COMPLETED' state and calculate score
                                                                #and will changing current action state from 'PENDING' state to 'IN PROCESS' as well

                                                                #Changing previous action from 'IN PROCESS' state to 'COMPLETED' state
                                                                self.boolHistoryActionState[key - 1] = 'COMPLETED'

                                                                #Calculate score if master data avaliable
                                                                if self.haveMaster is True:

                                                                    #Initialize score
                                                                    self.score = 100

                                                                    #Total distance will use to subtract score
                                                                    totalDistance = None

                                                                    #Check if program have left hand data of previous action in master data
                                                                    if f'{self.actionLabels[key - 1]}_left' in self.master:

                                                                        #If have data, Then calculate the distance with DTW
                                                                        distance = calculateScoreNew(self.master[f'{self.actionLabels[key - 1]}_left'], self.dtwCoordLeft[key - 1])
                                                                        
                                                                        #If the action is not have enough coordinate data to calculate the distance
                                                                        #distance will be none
                                                                        if distance is not None:
                                                                            #If distance is not none then add to total distance
                                                                            if totalDistance is not None:
                                                                                totalDistance += distance
                                                                            else:
                                                                                totalDistance = distance

                                                                    #Check if program have right hand data of previous action in master data
                                                                    if f'{self.actionLabels[key - 1]}_right' in self.master:
                                                                        
                                                                        #If have data, Then calculate the distance with DTW
                                                                        distance = calculateScoreNew(self.master[f'{self.actionLabels[key - 1]}_right'], self.dtwCoordRight[key - 1])
                                                                        
                                                                        #If the action is not have enough coordinate data to calculate the distance
                                                                        #distance will be none
                                                                        if distance is not None:
                                                                            #If distance is not none then add to total distance
                                                                            if totalDistance is not None:
                                                                                totalDistance += distance
                                                                            else:
                                                                                totalDistance = distance

                                                                    if f'{self.actionLabels[key - 1]}_left' not in self.master and f'{self.actionLabels[key - 1]}_right' not in self.master:
                                                                        #Set score to none if master data dont have both hands data of current action
                                                                        self.score = None
                                                                    else:
                                                                        if totalDistance is not None:
                                                                            #Subtract score with total distance from DTW
                                                                            self.score -= totalDistance

                                                                            #Set min score to 50
                                                                            if self.score < 50:
                                                                                self.score = 50
                                                                        else:
                                                                            self.score = None

                                                                    #After calculate score. Next, store score in dictionary
                                                                    if self.score is not None:
                                                                        self.scoreRecord[key - 1] = float('{:.2f}'.format(self.score))
                                                                    else:
                                                                        self.scoreRecord[key - 1] = None

                                                                    #Then clear coordinates data of current action
                                                                    self.dtwCoordLeft[key - 1] = []
                                                                    self.dtwCoordRight[key - 1] = []

                                                                #Changing current action state from 'PENDING' state to 'IN PROCESS'
                                                                self.boolHistoryActionState[key] = 'IN PROCESS'

                                                                #Start timer
                                                                self.dwellRunning = True
                                                                self.lastestDwellTime = self.dwellTime
                                                                self.timeRecord[key - 1] = float('{:.2f}'.format(self.dwellTime))
                                                                self.dtime = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)%60
                                                                self.dwellTime = 0

                                                                #Plot new graph
                                                                self.stopPlot.emit()
                                                                self.startPlot.emit(self.actionLabels[predicted_label])

                                                                #Start appending hand coordinate for use in calculate score
                                                                self.recordingDTWCoord = True
                                                                self.currentActionRecording = key

                                                            #If current action is first action of the operator work loop
                                                            elif int(key) == 1:

                                                                #Check if all action status is in waiting state
                                                                if all(value == 'PENDING' for value in self.boolHistoryActionState.values()):
                                                                    #If all action is in waiting process it's mean that operator is start working on the new loop
                                                                    #Because program checked and reset if all action completed before
                                                                    #So, program will reset and start new total timer and each action timer. And save score and time record
                                                                    #of previous work loop

                                                                    #Changing current action state from 'PENDING' state to 'IN PROCESS'
                                                                    self.boolHistoryActionState[key] = 'IN PROCESS'

                                                                    #Start timer
                                                                    self.dwellRunning = True
                                                                    self.lastestDwellTime = self.dwellTime
                                                                    self.dtime = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)%60
                                                                    self.dwellTime = 0

                                                                    #Start total timer
                                                                    self.totalDwellRunning = True
                                                                    self.lastestTotalDwellTime = float('{:.2f}'.format(self.totalDwellTime))
                                                                    self.totalDtime = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)%60
                                                                    self.totalDwellTime = 0

                                                                    #Plot new graph
                                                                    self.stopPlot.emit()
                                                                    self.startPlot.emit(self.actionLabels[predicted_label])

                                                                    #Start appending hand coordinate for use in calculate score
                                                                    self.recordingDTWCoord = True
                                                                    self.currentActionRecording = key

                                                                    #if self.saveScore and all(value != None for value in self.timeRecord.values()) and all(value != None for value in self.scoreRecord.values()):
                                                                    if self.saveScore and all(value != None for value in self.timeRecord.values()):
                                                                        loggingTimeAndScore(self.saveScorePath, self.lastestTotalDwellTime, self.timeRecord, self.scoreRecord, self.actionLabels)
                                                                    
                                                                    self.timeRecord[key] = float('{:.2f}'.format(self.dwellTime))

                                                                    #Clear time and score record after save
                                                                    for key_r in self.timeRecord:
                                                                        self.timeRecord[key_r] = None
                                                                        self.scoreRecord[key_r] = None     

                                                    if self.boolActionState[key] is False:
                                                        self.boolActionState[key] = True
                            
                                                else:
                                                    self.boolActionState[key] = False
                                            
                                            #Make program continue running total timer until the next work loop begin
                                            #and stop each action timer, So the program will timer every action except idle action
                                            if predicted_label == 0 or predicted_label == 1:
                                                #If current action is Idle or first action of operator working loop
                                                if self.boolHistoryActionState[len(self.boolHistoryActionState.keys())] == 'IN PROCESS':
                                                    #Check if the last action is in process state
                                                    #If in, That means the previous loop is finish

                                                    #Stop timer and append the last action time into record
                                                    self.dwellRunning = False
                                                    self.lastestDwellTime = self.dwellTime
                                                    self.timeRecord[key] = float('{:.2f}'.format(self.dwellTime))
                                                    self.dtime = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)%60
                                                    self.dwellTime = 0

                                                    #Changing last action of previous working loop from 'IN PROCESS' to 'COMPLETED'
                                                    self.boolHistoryActionState[len(self.boolHistoryActionState.keys())] = 'COMPLETED'

                                                    #All action is completed, So stop graph plotting
                                                    self.stopPlot.emit()

                                                    #Stop record coordinate
                                                    self.recordingDTWCoord = False
                                                    self.currentActionRecording = None
                                                    
                                                    #Calculating score
                                                    if self.haveMaster is True:
                                                        self.score = 100
                                                        totalDistance = None
                                                        if f'{self.actionLabels[key]}_left' in self.master:
                                                            distance = calculateScoreNew(self.master[f'{self.actionLabels[key]}_left'], self.dtwCoordLeft[len(self.boolHistoryActionState.keys())])
                                                            if distance is not None:
                                                                if totalDistance is not None:
                                                                    totalDistance += distance
                                                                else:
                                                                    totalDistance = distance
                                                        if f'{self.actionLabels[key]}_right' in self.master:
                                                            distance = calculateScoreNew(self.master[f'{self.actionLabels[key]}_right'], self.dtwCoordRight[len(self.boolHistoryActionState.keys())])
                                                            if distance is not None:
                                                                if totalDistance is not None:
                                                                    totalDistance += distance
                                                                else:
                                                                    totalDistance = distance
                                                        if f'{self.actionLabels[key]}_left' not in self.master and f'{self.actionLabels[key]}_right' not in self.master:
                                                            self.score = None
                                                        else:
                                                            if totalDistance is not None:
                                                                self.score -= totalDistance
                                                                if self.score < 50:
                                                                    self.score = 50
                                                            else:
                                                                self.score = None
                                                        if self.score is not None:
                                                            self.scoreRecord[key] = float('{:.2f}'.format(self.score))
                                                        else:
                                                            self.scoreRecord[key] = None

                                                        #Clear coordinate data of last action
                                                        self.dtwCoordLeft[len(self.boolHistoryActionState.keys())] = []
                                                        self.dtwCoordRight[len(self.boolHistoryActionState.keys())] = []
                                        #endregion

                self.graphData = float('{:.2f}'.format(self.graphData))

                #Appending finger tip coordinate to dictionary object for use in calculating score later on
                if self.detectMode and len(self.predictedAction) > 0 and self.recordingDTWCoord:
                    self.dtwCoordRight[self.currentActionRecording].append(right_pre_processed_landmark_list)
                    self.dtwCoordLeft[self.currentActionRecording].append(left_pre_processed_landmark_list)

                #Each action timer
                if self.dwellRunning:
                    current_time = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)%60
                    old_time = self.dtime

                    if current_time < old_time:
                        old_time = 0
                    
                    time_diff = current_time - old_time
                    self.dtime = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)%60
                    self.dwellTime += time_diff
                else:
                    self.dwellTime = 0
                    
                #Total action timer
                if self.totalDwellRunning is True:
                    total_current_time = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)%60
                    total_old_time = self.totalDtime

                    if total_current_time < total_old_time:
                        total_old_time = 0

                    total_time_diff = total_current_time - total_old_time
                    self.totalDtime = (cap.get(cv2.CAP_PROP_POS_MSEC)/1000)%60
                    self.totalDwellTime += total_time_diff

                #ROI display
                if self.leftRoi is not None:
                    if self.recording is True:
                        self.dictLeftRoi.update({self.frameCount : self.leftRoi})
                    self.debugRoiLeft = debug_frame_roi[self.leftRoi[2] : self.leftRoi[3], self.leftRoi[0] : self.leftRoi[1]]
                    self.debugRoiLeft = cv2.resize(self.debugRoiLeft, (128, 128), interpolation=cv2.INTER_AREA)
                    self.debugRoiLeft = drawROIFrame(self.debugRoiLeft, 'left', roi_detected)
                if self.rightRoi is not None:
                    if self.recording is True:
                        self.dictRightRoi.update({self.frameCount : self.rightRoi})
                    self.debugRoiRight = debug_frame_roi[self.rightRoi[2] : self.rightRoi[3], self.rightRoi[0] : self.rightRoi[1]]
                    self.debugRoiRight = cv2.resize(self.debugRoiRight, (128, 128), interpolation=cv2.INTER_AREA)
                    self.debugRoiRight = drawROIFrame(self.debugRoiRight, 'right', roi_detected)
                if self.bothRoi is not None:
                    if self.recording is True:
                        self.dictBothRoi.update({self.frameCount : self.bothRoi})
                    self.debugRoiBoth = debug_frame_roi[self.bothRoi[2] : self.bothRoi[3], self.bothRoi[0] : self.bothRoi[1]]
                    self.debugRoiBoth = cv2.resize(self.debugRoiBoth, (128, 128), interpolation=cv2.INTER_AREA)
                    self.debugRoiBoth = drawROIFrame(self.debugRoiBoth, 'both', roi_detected)
                
                if self.recordMode:
                    cv2.putText(debug_frame, 'RECORD MODE', (5, 15), cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 0, 255), 1, cv2.LINE_AA)

                #Convert frame to Qt format for it can display on GUI
                rgbMainFrame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbMainFrame.data , rgbMainFrame.shape[1], rgbMainFrame.shape[0], QtGui.QImage.Format.Format_RGB888)
                debugMainFrame = convertToQtFormat.scaled(self.mainDisplayWidth, self.mainDisplayheight, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

                rgbLeftROIFrame = cv2.cvtColor(self.debugRoiLeft, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbLeftROIFrame.data , rgbLeftROIFrame.shape[1], rgbLeftROIFrame.shape[0], QtGui.QImage.Format.Format_RGB888)
                debugLeftROIFrame = convertToQtFormat.scaled(self.leftROIDisplayWidth, self.leftROIDisplayheight, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

                rgbRightROIFrame = cv2.cvtColor(self.debugRoiRight, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbRightROIFrame.data , rgbRightROIFrame.shape[1], rgbRightROIFrame.shape[0], QtGui.QImage.Format.Format_RGB888)
                debugRightFOIFrame = convertToQtFormat.scaled(self.rightROIDisplayWidth, self.rightROIDisplayheight, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                
                rgbBothROIFrame = cv2.cvtColor(self.debugRoiBoth, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbBothROIFrame.data , rgbBothROIFrame.shape[1], rgbBothROIFrame.shape[0], QtGui.QImage.Format.Format_RGB888)
                debugBothROIFrame = convertToQtFormat.scaled(self.bothROIDisplayWidth, self.bothROIDisplayheight, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                
                #Convert process status from dictionary object to dataframe
                tableDataFrame = pd.DataFrame.from_dict([self.boolHistoryActionState, self.timeRecord, self.scoreRecord])
                tableDataFrame.columns = self.processTableViewColumn
                tableDataFrame.index = ['Status', 'Time(sec)', 'Score']
                tableDataFrame.fillna('',inplace=True)

                #Send signal and data to GUI for displaying
                if self.haveMaster:
                    self.updatePlot.emit(self.graphData)
                self.mainFrameUpdate.emit(debugMainFrame)
                self.leftROIFrameUpdate.emit(debugLeftROIFrame)
                self.rightROIFrameUpdate.emit(debugRightFOIFrame)
                self.bothROIFrameUpdate.emit(debugBothROIFrame)
                self.videoFPSUpdate.emit(fps)
                if len(self.predictedAction) > 0:
                    self.actionUpdate.emit(f'{self.predictedAction[0]} ' + '({:.2f})'.format(self.predictedAction[1]))
                self.scoreUpdate.emit(self.score)
                self.timeUpdate.emit(self.dwellTime)
                self.totalTimeUpdate.emit(self.totalDwellTime)
                self.processTableDataFrameUpdate.emit(tableDataFrame)

            cap.release()

        #Setting MediaPipe model
        self.hands = self.mpHands.Hands(
                    model_complexity=self.mpModelComplex,
                    max_num_hands=self.mpMaxHand,
                    min_detection_confidence=self.mpMinDetect,
                    min_tracking_confidence=self.mpMinTrack,
                )
        
        #Setting ROI radius
        self.bbox_radius = self.roiSize
        self.combine_bbox_radius = self.combineROISize

        if self.detectMode:
            #Setting TensorflowLite model if in detect mode
            self.confidence = self.modelConfidence
            self.windowSize = self.modelWindowSize
            self.predictedProbabilitiesDeque = deque(maxlen = self.windowSize)

            #Load master label path(path is set to 'program path/classes label')
            try:
                for csvname in os.listdir(self.masterLabelPath):
                    if csvname.endswith('.csv'):
                        if csvname == os.path.basename(self.labelPath):
                            self.masterLabelPath = f'{self.masterLabelPath}/{csvname}'
            except ValueError as e:
                print('model path is wrong!')
                self.error.emit(e)
                self.finished.emit()
                return
            
            #Load user custom label path
            try:
                with open(self.labelPath, encoding='utf-8-sig') as f:
                    self.actionLabels = csv.reader(f)
                    self.actionLabels = [row[0] for row in self.actionLabels]
            except FileNotFoundError as e:
                print('Label path is wrong!')
                self.error.emit(e)
                self.finished.emit()
                return

            #Load TensorflowLite model
            try:
                self.videoClassifier = VideoClassifier(self.modelPath, self.masterLabelPath, self.labelPath)
            except ValueError as e:
                print('Cannot setup Tensorflow Lite model!')
                self.error.emit(e)
                self.finished.emit()
                return

            #Load master data if avaliable
            if self.masterPath is not None and self.masterPath != '':
                try:
                    for csvname in os.listdir(f'{self.programRoot}/data'):
                        if csvname.endswith('.csv'):
                            self.haveMaster = True
                            with open(f'{self.programRoot}/data/{csvname}', encoding='utf-8-sig') as f:
                                self.masterName = (os.path.splitext(csvname))[0]
                                self.masterCoord = csv.reader(f)
                                self.masterCoord = [row for row in self.masterCoord]
                                self.master.update({self.masterName : self.masterCoord})
                except (ValueError, FileNotFoundError) as e:
                    print('Master path is wrong!')
                    self.error.emit(e)
                    self.finished.emit()
                    return

            #Create dictionary with initial value wait for update later
            for label in self.actionLabels:
                if label != 'Idle':
                    self.timeRecord.update({(self.actionLabels.index(label)) : None})
                    self.scoreRecord.update({(self.actionLabels.index(label)) : None})
                    self.dtwCoordLeft.update({(self.actionLabels.index(label)) : []}) 
                    self.dtwCoordRight.update({(self.actionLabels.index(label)) : []})
                    self.boolHistoryActionState.update({(self.actionLabels.index(label)) : 'PENDING'})
                    self.processTableViewColumn.append(label)
                
                self.boolActionState.update({(self.actionLabels.index(label)) : False})  
        
        #Create and set up empty ROI windows
        self.debugRoiLeft = np.zeros((128,128,3), dtype=np.uint8)
        self.debugRoiRight = np.zeros((128,128,3), dtype=np.uint8)
        self.debugRoiBoth = np.zeros((128,128,3), dtype=np.uint8)

        cv2.rectangle(self.debugRoiLeft, (12, 61), (116, 81), (255, 255, 255), -1)
        cv2.rectangle(self.debugRoiLeft, (12, 61), (116, 81), (0, 0, 0), 1)
        cv2.putText(self.debugRoiLeft, 'NOT DETECTED', (19, 75), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(self.debugRoiRight, (12, 61), (116, 81), (255, 255, 255), -1)
        cv2.rectangle(self.debugRoiRight, (12, 61), (116, 81), (0, 0, 0), 1)
        cv2.putText(self.debugRoiRight, 'NOT DETECTED', (19, 75), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(self.debugRoiBoth, (12, 61), (116, 81), (255, 255, 255), -1)
        cv2.rectangle(self.debugRoiBoth, (12, 61), (116, 81), (0, 0, 0), 1)
        cv2.putText(self.debugRoiBoth, 'NOT DETECTED', (19, 75), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 255), 1, cv2.LINE_AA)

        if self.feedFromCamera:
            #If user choose source to be live camera
            #then call videoCap function with camera index parameter
            videoCap(int(self.source))
        else:
            #If user choose source to be video files
            #Collect all videos name and path from source path
            try:
                for file in os.listdir(self.source):
                    if file.endswith('.mp4'):
                        self.videoPath.append(os.path.join(self.source, file))
            except FileNotFoundError as e:
                print('Video source is wrong!')
                self.error.emit(e)
                self.finished.emit()
                return
        
        #then call videoCap function with videos path parameter from list that we collected earlier 
        while self.currentVideo < len(self.videoPath):
            self.videoIndexUpdate.emit(f'{self.currentVideo + 1} / {len(self.videoPath)}')
            videoCap(self.videoPath[self.currentVideo])
            
            if self.videoInterrupt:
                self.videoInterrupt = False
            else:
                self.currentVideo += 1
        
        self.finished.emit()

    def stop(self):
        if self.videoPause:
            self.videoPause = False

        self.threadActive = False
        self.quit()
        self.wait()
