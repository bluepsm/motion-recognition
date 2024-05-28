from PyQt6 import QtCore, QtGui


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        elif role == QtCore.Qt.ItemDataRole.BackgroundRole:
            value = self._data.iloc[0, index.column()]
            if value == 'IN PROCESS':
                return QtGui.QBrush(QtGui.QColor(133, 193, 233))
            elif value == 'COMPLETED':
                return QtGui.QBrush(QtGui.QColor(130, 224, 170))
        elif role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return QtCore.Qt.AlignmentFlag.AlignCenter

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self._data.columns[section])

            if orientation == QtCore.Qt.Orientation.Vertical:
                return str(self._data.index[section])
