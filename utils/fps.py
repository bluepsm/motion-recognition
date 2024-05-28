from collections import deque
import cv2 as cv

class FPS(object):
    def __init__(self, bufferLen=1):
        self.startTick = cv.getTickCount()
        self.freq = 1000.0 / cv.getTickFrequency()
        self.diffTimes = deque(maxlen=bufferLen)

    def get(self):
        currentTick = cv.getTickCount()
        differentTime = (currentTick - self.startTick) * self.freq
        self.startTick = currentTick

        self.diffTimes.append(differentTime)

        fps = 1000.0 / (sum(self.diffTimes) / len(self.diffTimes))
        fpsRounded = round(fps, 2)

        return int(fpsRounded)
