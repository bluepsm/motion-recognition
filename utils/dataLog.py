import cv2
import csv
import statistics
import datetime
import os

def extractClipAndImage(action, subStartFrame, subEndFrame, path, videoName, LeftROI, rightROI, bothROI, ROIRadius, combineROIRadius, saveVideo, saveVideoROI, saveImage, savePath):
    parts = [(subStartFrame, subEndFrame)]

    if saveVideo:
        videoSavePathIsExist = os.path.exists(f'{savePath}/Videos/{action}')
        if not videoSavePathIsExist:
            os.makedirs(f'{savePath}/Videos/{action}')
    
    if saveVideoROI:
        videoROISavePathIsExist = os.path.exists(f'{savePath}/ROI Videos/{action}')
        if not videoROISavePathIsExist:
            os.makedirs(f'{savePath}/ROI Videos/{action}')

    if saveImage:
        imageSavePathIsExist = os.path.exists(f'{savePath}/Images/{action}')
        if not imageSavePathIsExist:
            os.makedirs(f'{savePath}/Images/{action}')
    
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    height, width, _ = frame.shape
    """ width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) """

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if saveVideo:
        writers = [cv2.VideoWriter(f'{savePath}/Videos/{action}/{action}_{videoName}_{start}-{end}.mp4', fourcc, 29.81, (width, height)) for start, end in parts]
    
    if saveVideoROI:
        if len(LeftROI) > 0:
            leftROIWriters = [cv2.VideoWriter(f'{savePath}/ROI Videos/{action}/left_{action}_{videoName}_{start}-{end}.mp4', fourcc, 29.81, (ROIRadius * 2, ROIRadius * 2)) for start, end in parts]
        if len(rightROI) > 0:
            rightROIWriters = [cv2.VideoWriter(f'{savePath}/ROI Videos/{action}/right_{action}_{videoName}_{start}-{end}.mp4', fourcc, 29.81, (ROIRadius * 2, ROIRadius * 2)) for start, end in parts]
        if len(bothROI) > 0:
            bothROIWriters = [cv2.VideoWriter(f'{savePath}/ROI Videos/{action}/both_{action}_{videoName}_{start}-{end}.mp4', fourcc, 29.81, (combineROIRadius * 2, combineROIRadius * 2)) for start, end in parts]

    print('\nimages and video saving...')

    f = 0
    while ret:
        f += 1
        """ width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) """
        height, width, _ = frame.shape
        cv2.rectangle(frame, (width - 75, height - 20), (width, height), (0, 0, 0), -1)

        for index, part in enumerate(parts):
            start, end = part
            if start <= f <= end:
                if saveVideo:
                    writers[index].write(frame)

                if saveImage:
                    if f == start or f%15 == 0:
                        cv2.imwrite(f'{savePath}/Images/' + action + '/' + str(videoName) + '_' + str(f) + '.jpg', frame)
                
                if saveVideoROI:
                    if len(LeftROI) > 0:
                        for frameLeft in LeftROI:
                            if f == frameLeft:
                                leftROIFrame = frame[LeftROI[frameLeft][2] : LeftROI[frameLeft][3], LeftROI[frameLeft][0] : LeftROI[frameLeft][1]]
                                leftROIWriters[index].write(leftROIFrame)
                    if len(rightROI) > 0:
                        for frameRight in rightROI:
                            if f == frameRight:
                                rightROIFrame = frame[rightROI[frameRight][2] : rightROI[frameRight][3], rightROI[frameRight][0] : rightROI[frameRight][1]]
                                rightROIWriters[index].write(rightROIFrame)
                    if len(bothROI) > 0:
                        for frameBoth in bothROI:
                            if f == frameBoth:
                                bothROIFrame = frame[bothROI[frameBoth][2] : bothROI[frameBoth][3], bothROI[frameBoth][0] : bothROI[frameBoth][1]]
                                bothROIWriters[index].write(bothROIFrame)
                
        ret, frame = cap.read()

    print('images and video saved!')

    if saveVideo:
        for writer in writers:
            writer.release()
    
    if saveVideoROI:
        if len(LeftROI) > 0:
            for leftWriter in leftROIWriters:
                leftWriter.release()
        if len(rightROI) > 0:
            for rightWriter in rightROIWriters:
                rightWriter.release()
        if len(bothROI) > 0:
            for bothWriter in bothROIWriters:
                bothWriter.release()

    cap.release()

    print('\n================================================================================')

def loggingCSV(action, actionText, preProcessedLandmarkList, savePath):
    csvSavePathIsExist = os.path.exists(f'{savePath}/csv/{actionText}')

    if not csvSavePathIsExist:
        os.makedirs(f'{savePath}/CSV/{actionText}')

    csvCoordinatePath_1 = f'{savePath}/CSV/{actionText}/{actionText}_coordinate_keypoint.csv'
    with open(csvCoordinatePath_1, 'a', newline='') as f1:
        writer1 = csv.writer(f1)
        writer1.writerow([*preProcessedLandmarkList])
    csvCoordinatePath_2 = f'{savePath}/CSV/summary_coordinate_keypoint.csv'
    with open(csvCoordinatePath_2, 'a', newline='') as f2:
        writer2 = csv.writer(f2)
        writer2.writerow([action, *preProcessedLandmarkList])

def loggingMasterCSV(masterRecordModeCheck, predictedAction, preProcessedLandmarkList, hand, actionLabel):
    masterRecordModeCheck.append(predictedAction)
    if hand == 'left':
        leftMasterCSVCoordinatePath = f'master/new record/{actionLabel[predictedAction]}_left.csv'
        with open(leftMasterCSVCoordinatePath, 'a', newline='') as f1:
            writer1 = csv.writer(f1)
            writer1.writerow([*preProcessedLandmarkList])
    else:
        rightMasterCSVCoordinatePath = f'master/new record/{actionLabel[predictedAction]}_right.csv'
        with open(rightMasterCSVCoordinatePath, 'a', newline='') as f2:
            writer2 = csv.writer(f2)
            writer2.writerow([*preProcessedLandmarkList])

    return list(set(masterRecordModeCheck))

def loggingMasterCSVNew(program_root, action, preProcessedLandmarkList, hand):
    if not os.path.isdir(f'{program_root}/data'):
        os.makedirs(f'{program_root}/data')
    
    if hand == 'left':
        leftMasterCSVCoordinatePath = f'{program_root}/data/{action}_left.csv'
        with open(leftMasterCSVCoordinatePath, 'a', newline='') as f1:
            writer1 = csv.writer(f1)
            writer1.writerow([*preProcessedLandmarkList])
    else:
        rightMasterCSVCoordinatePath = f'{program_root}/data/{action}_right.csv'
        with open(rightMasterCSVCoordinatePath, 'a', newline='') as f2:
            writer2 = csv.writer(f2)
            writer2.writerow([*preProcessedLandmarkList])

def loggingTimeAndScore(savePath, lastestTotalDwellTime, timeRecord, scoreRecord, actionLabel):
    csvScorePath = f'{savePath}/{datetime.datetime.now().date()}.csv'
    fileExists = os.path.isfile(csvScorePath)

    headers = []
    for action in actionLabel:
        if action != 'Idle':
            headers.append(action + ' Time(sec)')
            headers.append(action + ' Score')
    headers.append('Total Time(sec)')
    headers.append('Average Score')

    data = []
    for key in timeRecord:
        data.append(timeRecord[key])
        data.append(scoreRecord[key])
        
    if all(scoreRecord.values()):
        avgScore = float('{:.2f}'.format(statistics.mean(value for value in scoreRecord.values())))
    else:
        avgScore = None

    with open(csvScorePath, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not fileExists:
            writer.writerow([*headers])  # file doesn't exist yet, write a header

        writer.writerow([*data, lastestTotalDwellTime, avgScore])