from PyQt6 import QtCore

class StdErrHandler(QtCore.QObject):
    '''
    This class provides an alternate write() method for stderr messages.
    Messages are sent by pyqtSignal to the pyqtSlot in the main window.
    '''
    err_msg = QtCore.pyqtSignal(str)

    def __init__(self):
        super(StdErrHandler, self).__init__()
        self.matplotlibWarning = 'UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.'
        self.tensorflowWarning = 'WARNING:absl:Found untraced functions such as'

    def write(self, msg):
        # stderr messages are sent to this method.
        if self.matplotlibWarning not in msg and self.tensorflowWarning not in msg:
            self.err_msg.emit(msg)