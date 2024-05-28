import cv2

def drawLandmark(frame, landmarkList):
        # Lines
        if len(landmarkList) > 0:
                # Thumb
                cv2.line(frame, (landmarkList[2][0], landmarkList[2][1]), (landmarkList[3][0], landmarkList[3][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[2][0], landmarkList[2][1]), (landmarkList[3][0], landmarkList[3][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[3][0], landmarkList[3][1]), (landmarkList[4][0], landmarkList[4][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[3][0], landmarkList[3][1]), (landmarkList[4][0], landmarkList[4][1]),
                        (255, 255, 255), 1)

                # Index finger
                cv2.line(frame, (landmarkList[5][0], landmarkList[5][1]), (landmarkList[6][0], landmarkList[6][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[5][0], landmarkList[5][1]), (landmarkList[6][0], landmarkList[6][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[6][0], landmarkList[6][1]), (landmarkList[7][0], landmarkList[7][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[6][0], landmarkList[6][1]), (landmarkList[7][0], landmarkList[7][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[7][0], landmarkList[7][1]), (landmarkList[8][0], landmarkList[8][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[7][0], landmarkList[7][1]), (landmarkList[8][0], landmarkList[8][1]),
                        (255, 255, 255), 1)

                # Middle finger
                cv2.line(frame, (landmarkList[9][0], landmarkList[9][1]), (landmarkList[10][0], landmarkList[10][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[9][0], landmarkList[9][1]), (landmarkList[10][0], landmarkList[10][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[10][0], landmarkList[10][1]), (landmarkList[11][0], landmarkList[11][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[10][0], landmarkList[10][1]), (landmarkList[11][0], landmarkList[11][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[11][0], landmarkList[11][1]), (landmarkList[12][0], landmarkList[12][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[11][0], landmarkList[11][1]), (landmarkList[12][0], landmarkList[12][1]),
                        (255, 255, 255), 1)

                # Ring finger
                cv2.line(frame, (landmarkList[13][0], landmarkList[13][1]), (landmarkList[14][0], landmarkList[14][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[13][0], landmarkList[13][1]), (landmarkList[14][0], landmarkList[14][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[14][0], landmarkList[14][1]), (landmarkList[15][0], landmarkList[15][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[14][0], landmarkList[14][1]), (landmarkList[15][0], landmarkList[15][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[15][0], landmarkList[15][1]), (landmarkList[16][0], landmarkList[16][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[15][0], landmarkList[15][1]), (landmarkList[16][0], landmarkList[16][1]),
                        (255, 255, 255), 1)

                # Little finger
                cv2.line(frame, (landmarkList[17][0], landmarkList[17][1]), (landmarkList[18][0], landmarkList[18][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[17][0], landmarkList[17][1]), (landmarkList[18][0], landmarkList[18][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[18][0], landmarkList[18][1]), (landmarkList[19][0], landmarkList[19][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[18][0], landmarkList[18][1]), (landmarkList[19][0], landmarkList[19][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[19][0], landmarkList[19][1]), (landmarkList[20][0], landmarkList[20][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[19][0], landmarkList[19][1]), (landmarkList[20][0], landmarkList[20][1]),
                        (255, 255, 255), 1)

                # Palm
                cv2.line(frame, (landmarkList[0][0], landmarkList[0][1]), (landmarkList[1][0], landmarkList[1][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[0][0], landmarkList[0][1]), (landmarkList[1][0], landmarkList[1][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[1][0], landmarkList[1][1]), (landmarkList[2][0], landmarkList[2][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[1][0], landmarkList[1][1]), (landmarkList[2][0], landmarkList[2][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[2][0], landmarkList[2][1]), (landmarkList[5][0], landmarkList[5][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[2][0], landmarkList[2][1]), (landmarkList[5][0], landmarkList[5][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[5][0], landmarkList[5][1]), (landmarkList[9][0], landmarkList[9][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[5][0], landmarkList[5][1]), (landmarkList[9][0], landmarkList[9][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[9][0], landmarkList[9][1]), (landmarkList[13][0], landmarkList[13][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[9][0], landmarkList[9][1]), (landmarkList[13][0], landmarkList[13][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[13][0], landmarkList[13][1]), (landmarkList[17][0], landmarkList[17][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[13][0], landmarkList[13][1]), (landmarkList[17][0], landmarkList[17][1]),
                        (255, 255, 255), 1)
                cv2.line(frame, (landmarkList[17][0], landmarkList[17][1]), (landmarkList[0][0], landmarkList[0][1]),
                        (0, 0, 0), 2)
                cv2.line(frame, (landmarkList[17][0], landmarkList[17][1]), (landmarkList[0][0], landmarkList[0][1]),
                        (255, 255, 255), 1)

                # Keypoints
                for index, landmark in enumerate(landmarkList):
                        if index == 0:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  1:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  2:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  3:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  4:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  5:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  6:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  7: 
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  8:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  9: 
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  10:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  11:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  12:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  13:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  14:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  15:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  16:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  17:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  18:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  19:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)
                        elif index ==  20:
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (192, 192, 192), -1)
                                cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 0), 1)

        return frame

def drawROIBothHand(frame, leftROI, rightROI, combineROIRadius):
        minStart_x = min(leftROI[0], rightROI[0])
        minStart_y = min(leftROI[2], rightROI[2])
        maxWidth = max(leftROI[1], rightROI[1]) - minStart_x
        maxHeight = max(leftROI[3], rightROI[3]) - minStart_y
        combine_cx = minStart_x + maxWidth // 2
        combine_cy = minStart_y + maxHeight // 2

        combineStart_x = combine_cx - combineROIRadius
        combineStart_y = combine_cy - combineROIRadius
        combineEnd_x = combine_cx + combineROIRadius
        combineEnd_y = combine_cy + combineROIRadius

        cv2.rectangle(frame, (combineStart_x, combineStart_y - 10), 
                                        (combineEnd_x, combineStart_y), (0, 0, 0), -1)
        cv2.rectangle(frame, (combineStart_x, combineStart_y), 
                                        (combineEnd_x, combineEnd_y), (0, 0, 0), 1)

        cv2.putText(frame, 'Both', (combineStart_x, combineStart_y),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Both', (combineStart_x, combineStart_y),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame, [combineStart_x, combineEnd_x, combineStart_y, combineEnd_y]

def drawROIEachHand(frame, roi, handedness):
        cv2.rectangle(frame, (roi[0], roi[2] - 10), 
                                (roi[1], roi[2]), (0, 0, 0), -1)
        cv2.rectangle(frame, (roi[0], roi[2]), 
                                (roi[1], roi[3]), (0, 0, 0), 1)

        cv2.putText(frame, handedness, (roi[0], roi[2]),
                cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, handedness, (roi[0], roi[2]),
                cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

def drawROIFrame(roiFrame, roi, roiDetected):
        if roi == 'left':
                if roiDetected['left'] is False:
                        cv2.rectangle(roiFrame, (12, 61), (116, 81), (255, 255, 255), -1)
                        cv2.rectangle(roiFrame, (12, 61), (116, 81), (0, 0, 0), 1)
                        cv2.putText(roiFrame, 'NOT DETECTED', (19, 75), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
        elif roi == 'right':
                if roiDetected['right'] is False:
                        cv2.rectangle(roiFrame, (12, 61), (116, 81), (255, 255, 255), -1)
                        cv2.rectangle(roiFrame, (12, 61), (116, 81), (0, 0, 0), 1)
                        cv2.putText(roiFrame, 'NOT DETECTED', (19, 75), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
        elif roi == 'both':
                if roiDetected['both'] is False:
                        cv2.rectangle(roiFrame, (12, 61), (116, 81), (255, 255, 255), -1)
                        cv2.rectangle(roiFrame, (12, 61), (116, 81), (0, 0, 0), 1)
                        cv2.putText(roiFrame, 'NOT DETECTED', (19, 75), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
            
        return roiFrame