import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def frames_extraction(video_path, frame_width, frame_height):
    # Empty List declared to store video frames
    frames_list = []
    
    # Reading the Video File Using the VideoCapture
    video_reader = cv2.VideoCapture(video_path)

    # Iterating through Video Frames
    while True:

        # Reading a frame from the video file 
        success, frame = video_reader.read() 

        # If Video frame was not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (frame_height, frame_width))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        
        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)
    
    # Closing the VideoCapture object and releasing all resources. 
    video_reader.release()

    # returning the frames list 
    return frames_list

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name, outputPath):
    # Get Metric values using metric names as identifiers
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Constructing a range object which will be used as time 
    epochs = range(len(metric_value_1))
    
    # Plotting the Graph
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
    
    # Adding title to the plot
    plt.title(str(plot_name))

    # Adding legend to the plot
    plt.legend()

    plt.savefig(f'{outputPath}/{plot_name}.png')

    plt.close()

def findMaxFrame(datasetPath):
    frameCount = []
    for folder in os.listdir(datasetPath):
        sumFrameCount = 0
        for video in os.listdir(f'{datasetPath}/{folder}'):
            video_reader = cv2.VideoCapture(f'{datasetPath}/{folder}/{video}')
            sumFrameCount += int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            video_reader.release()
        frameCount.append(sumFrameCount)

    return np.amin(frameCount)

def findMaxFrameAllClasses(datasetPath):
    max_frame_per_class = {}

    for className in os.listdir(datasetPath):
        max_frame_per_class.update({className.split('_')[-1] : None})

        sumFrameCount = 0
        for video in os.listdir(f'{datasetPath}/{className}'):
            video_reader = cv2.VideoCapture(f'{datasetPath}/{className}/{video}')
            sumFrameCount += int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            video_reader.release()

        max_frame_per_class[className.split('_')[-1]] = sumFrameCount
        
    return max_frame_per_class
        