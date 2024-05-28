import csv
import numpy as np
import tensorflow as tf

class VideoClassifier(object):
    def __init__(
        self,
        modelPath,
        masterLabelPath,
        customLabelPath,
        numThreads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=modelPath,
                                               num_threads=numThreads)

        self.interpreter.allocate_tensors()
        self.inputDetails = self.interpreter.get_input_details()
        self.outputDetails = self.interpreter.get_output_details()

        with open(masterLabelPath, encoding='utf-8-sig') as f:
            self.labelsMain = csv.reader(f)
            self.labelsMain = [row[0] for row in self.labelsMain]
        with open(customLabelPath, encoding='utf-8-sig') as f:
            self.labelsCustom = csv.reader(f)
            self.labelsCustom = [row[0] for row in self.labelsCustom]

    def __call__(
        self,
        normalizedFrame,
    ):
        inputDetailsTensorIndex = self.inputDetails[0]['index']
        self.interpreter.set_tensor(
            inputDetailsTensorIndex,
            np.array([normalizedFrame], dtype=np.float32))
        self.interpreter.invoke()

        outputDetailsTensorIndex = self.outputDetails[0]['index']

        result = self.interpreter.get_tensor(outputDetailsTensorIndex)
        result = np.squeeze(result)
        newResult = []

        for labelMainName in self.labelsMain:
            if (labelMainName in self.labelsCustom):
                if self.labelsMain[np.argmax(result)] in self.labelsCustom:
                    labelMainIndex = self.labelsMain.index(labelMainName)
                    newResult.append(result[labelMainIndex])
                else:
                    if labelMainName == 'Idle':
                        newResult.append(result[np.argmax(result)])
                    else:
                        labelMainIndex = self.labelsMain.index(labelMainName)
                        newResult.append(result[labelMainIndex])
        
        """ print(["%.2f" % elem for elem in result])
        print(["%.2f" % elem for elem in newResult]) """

        return newResult
