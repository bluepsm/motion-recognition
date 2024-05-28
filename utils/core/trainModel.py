import sys
import os
import csv
import random
import numpy as np
import datetime as dt
import tensorflow as tf

#region 'Module for plot image'
import pydot
import pydotplus
import pydot_ng
import graphviz
#endregion

from keras.api._v2.keras.layers import *
from keras.api._v2.keras.models import Sequential
from keras.api._v2.keras.callbacks import EarlyStopping
from keras.api._v2.keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split

from utils.modelTraining import *

from PyQt6 import QtCore


class trainModel(QtCore.QThread):
    consoleLine = QtCore.pyqtSignal(str)
    modelStructurePath = QtCore.pyqtSignal(str)
    lossFigPath = QtCore.pyqtSignal(str)
    accuracyFigPath = QtCore.pyqtSignal(str)
    lossValue = QtCore.pyqtSignal(float)
    accuracyValue = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()

    def __init__(self):
        super(trainModel, self).__init__()

        self.modelPreset = None

        self.dataset_directory = None
        self.outputPath = None

        self.frame_height = None
        self.frame_width = None
        self.max_frame_per_class = None

        self.testSize = None

        self.modelLayers = []

        self.epoch = None
        self.batchSize = None
        self.validateSize = None

        self.terminate = False

    def run(self):
        def terminateFlag():
            if self.terminate:
                return True
            else:
                return False

        class terminateOnFlag(tf.keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                if terminateFlag():
                    self.model.stop_training = True
        
        random_seed = 23
        np.random.seed(random_seed)
        random.seed(random_seed)
        tf.random.set_seed(random_seed)

        #self.max_frame_per_class = None
        classes_list = []
        classesNoIndex_list = []
        classesToLabel = []

        for className in os.listdir(self.dataset_directory):
            #print(className)
            classes_list.append(className)
            classesNoIndex_list.append(className.split('_')[-1])
            
        frameWidth = self.frame_width
        frameHeight = self.frame_height
        numberOfClass = len(classes_list)

        temp_features = [] 
        features = []
        labels = []

        #sys.stdout = self

        # Iterating through all the classes mentioned in the classes list
        for class_index, class_name in enumerate(classesNoIndex_list):
            print(f'Extracting Data of Class: {class_name}')
            
            # Getting the list of video files present in the specific class name directory
            files_list = os.listdir(os.path.join(self.dataset_directory, classes_list[class_index]))

            # Iterating through all the files present in the files list
            for file_name in files_list:

                # Construct the complete video path
                video_file_path = os.path.join(self.dataset_directory, classes_list[class_index], file_name)

                # Calling the frame_extraction method for every video file path
                frames = frames_extraction(video_file_path, self.frame_width, self.frame_height)

                # Appending the frames to a temporary list.
                temp_features.extend(frames)
            
            # Adding randomly selected frames to the features list
            features.extend(random.sample(temp_features, self.max_frame_per_class))

            # Adding Fixed number of labels to the labels list
            labels.extend([class_index] * self.max_frame_per_class)
            
            # Emptying the temp_features list so it can be reused to store all frames of the next class.
            temp_features.clear()

        # Converting the features and labels lists to numpy arrays
        features = np.asarray(features)
        labels = np.array(labels)

        one_hot_encoded_labels = to_categorical(labels)

        features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = self.testSize, shuffle = True, random_state = random_seed)

        # We will use a Sequential model for model construction
        model = Sequential()

        # Defining The Model Architecture
        for layer in self.modelLayers:
            eval(f'model.add({layer})')

        # Printing the models summary
        model.summary()
        
        saveImgPath = f'{self.outputPath}/video_model_structure_plot.png'
        modelStructureImg = plot_model(model, to_file = saveImgPath, show_shapes = True, show_layer_names = True)

        self.modelStructurePath.emit(saveImgPath)

        early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)
        terminateCallBack = terminateOnFlag()

        model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

        model_training_history = model.fit(x = features_train, y = labels_train, epochs = self.epoch, batch_size = self.batchSize , shuffle = True, validation_split = self.validateSize, callbacks = [early_stopping_callback, terminateCallBack])

        model_evaluation_history = model.evaluate(features_test, labels_test)

        if not self.terminate:
            # Creating a useful name for our model, incase you're saving multiple models (OPTIONAL)
            date_time_format = '%Y_%m_%d__%H_%M_%S'
            current_date_time_dt = dt.datetime.now()
            current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
            model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
            model_name = f'{self.outputPath}/Model_Video_Date-{current_date_time_string}_Loss-{model_evaluation_loss}_Accuracy-{model_evaluation_accuracy}.h5'
            tflite_save_path = f'{self.outputPath}/Model_Video_Date-{current_date_time_string}_Loss-{model_evaluation_loss}_Accuracy-{model_evaluation_accuracy}.tflite'
            model.save(model_name)

            # Transform model (quantization)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quantized_model = converter.convert()

            open(tflite_save_path, 'wb').write(tflite_quantized_model)

            print('\n===== Training Completed. =====')
            sys.stdout = sys.__stdout__

            self.accuracyValue.emit(model_evaluation_accuracy)
            self.lossValue.emit(model_evaluation_loss)
            plot_metric(model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy', self.outputPath)
            self.accuracyFigPath.emit(f'{self.outputPath}/Total Accuracy vs Total Validation Accuracy.png')

            plot_metric(model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss', self.outputPath)
            self.lossFigPath.emit(f'{self.outputPath}/Total Loss vs Total Validation Loss.png')

            classesToLabel.append(classesNoIndex_list)
            with open(f'{self.outputPath}/label.csv', 'w', newline='') as f:
                write = csv.writer(f)
                for className in classesNoIndex_list:
                    write.writerow([className])
                f.close()

            self.finished.emit()
        else:
            self.terminate = False

    def stop(self):
        self.quit()

    def write(self, line):
        self.consoleLine.emit(line)

    def flush(self):
        pass
