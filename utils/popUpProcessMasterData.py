from PyQt6 import QtCore, QtGui, QtWidgets
from utils.core.processMasterData import processMasterData


class popUpProcessMasterData(QtWidgets.QDialog):
    masterGraph = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()

    def __init__(self, program_root):
        super(popUpProcessMasterData, self).__init__()
        self.program_root = program_root
        self.masterFolderPath = None
        self.modelComplex = None
        self.maxHand = None
        self.minDetect = None
        self.minTracking = None

        self.resize(650, 80)
        self.setWindowTitle('Master Data Processing..')
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window |
            QtCore.Qt.WindowType.CustomizeWindowHint |
            QtCore.Qt.WindowType.WindowTitleHint |
            QtCore.Qt.WindowType.WindowMinimizeButtonHint
            )
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setText("Class : ")
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(self)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setText("")
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.label_2)
        self.horizontalLayout.addLayout(self.formLayout)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.progressBar = QtWidgets.QProgressBar(self)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        
    def startProcess(self):
        self.processMaster = processMasterData(self.program_root)
        self.processMaster.masterFolderPath = self.masterFolderPath
        self.processMaster.modelComplex = self.modelComplex
        self.processMaster.maxHand = self.maxHand
        self.processMaster.minDetect = self.minDetect
        self.processMaster.minTracking = self.minTracking

        self.processMaster.progress.connect(self.updateProgressTextSlot)
        self.processMaster.masterGraph.connect(self.sendGraph)
        self.processMaster.finished.connect(self.processFinish)

        self.processMaster.start()

    def sendGraph(self, dict):
        self.masterGraph.emit(dict)

    def processFinish(self):
        self.processMaster.progress.disconnect(self.updateProgressTextSlot)
        self.processMaster.finished.disconnect(self.processFinish)
        self.finished.emit()
        self.close()

    def updateProgressTextSlot(self, list):
        """ print(f'Class : {list[0]}')
        print(f'Total : {list[1]}')
        print(f'Current : {list[2]}') """
        self.label_2.setText(list[0])
        self.progressBar.setValue(int((list[2]/list[1])*100))
