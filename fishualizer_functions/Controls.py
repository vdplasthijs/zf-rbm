from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import logging
from locale import setlocale, LC_NUMERIC, atof
import psutil as ps
import numpy as np
from pathlib import Path
import colors
logger = logging.getLogger('Fishlog')


def remove_alpha(gradient):
    """
    Remove transparency from the colors of a QLinearGradient

    Parameters
    ----------
    gradient: QLinearGradient

    Returns
    -------
     n_gradient: QLinearGradient
        Opaque gradient
    """
    params = gradient.stops()       # Get the gradient parameters
    pos = [p[0] for p in params]    # Positions of the stops
    colors = [p[1] for p in params]  # Colors
    colors = [QtGui.QColor(*c.getRgb()) for c in colors]  # Removing the alpha value but keeping the same RGB
    n_gradient = QtGui.QLinearGradient(gradient.start(), gradient.finalStop())  # Creating a similar gradient
    n_gradient.setStops(zip(pos, colors))

    return n_gradient


class GLViewWidgetPos(pg.opengl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

    def mousePressEvent(self, ev):
        super(GLViewWidgetPos, self).mousePressEvent(ev)
        # If you do not accept the event, then no move/release events will be received.
        ev.accept()

    def mouseReleaseEvent(self, ev):
        # print(ev.pos())
        modifiers = QtGui.QGuiApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            self.parent.select_cell(ev.pos(), 0)
        elif modifiers == QtCore.Qt.AltModifier:
            self.parent.select_cell(ev.pos(), 1)

    def orbit(self, azim, elev):
        """Orbits the camera around the center position. *azim* and *elev* are given in degrees."""
        # Overriden to allow free rotation insteas of clipping at +/-90
        self.opts['azimuth'] += azim
        self.opts['elevation'] += elev
        self.opts['elevation'] %= 360
        # self.opts['elevation'] = np.clip(self.opts['elevation'] + elev, -90, 90)
        self.update()


class ColorRangePicker(QtWidgets.QLabel):
    """
    Subclass of QLabel used to make color range picker.
    A custom color gradient is shown and the user is able to set a range by left/right clicks to chose the two ends
    of the range
    """
    # Custom signals. Are class attributes per Qt design
    top_limit = QtCore.pyqtSignal(float)
    bottom_limit = QtCore.pyqtSignal(float)

    def __init__(self, cmap):
        super().__init__()
        # Storing the colormap object from which we are making the gradient
        self._cmap = cmap
        self._stops = np.linspace(0, 1, 256)
        self.colors = cmap(self._stops)

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        # When the colormap is changed, trigger a refresh of the color picker
        self._cmap = value
        self.colors = self.cmap(self._stops)
        self.repaint()

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        """
        Overridding the default mouse press event to trigger a custom signal
        If left click, triggers a top_limit event, if right then triggers a bottom_limit event.
        The custom events carry information on the relative position of the click as to make finding the corresponding
        color easy

        Parameters
        ----------
        ev: QtGui.QMouseEvent

        """
        super(ColorRangePicker, self).mousePressEvent(ev)
        if ev.button() == 1:
            self.top_limit.emit(ev.y() / self.height())
        elif ev.button() == 2:
            self.bottom_limit.emit(ev.y() / self.height())

    def paintEvent(self, a0: QtGui.QPaintEvent):
        """
        Overridding the default paint event (refresh of the QLabel) to plot the gradient
        It is a Qt thing to add any custom display in the paintEvent so that it is done when necessary

        Parameters
        ----------
        a0: QtGui.QPaintEvent
            Just passed to the parent function

        Returns
        -------

        """
        super(ColorRangePicker, self).paintEvent(a0)
        # Create a painter (object making the drawing)
        painter = QtGui.QPainter(self)
        # Remove border
        painter.setPen(QtGui.QPen(0))
        # Get the gradient
        # Make sure not to use the alpha channel
        gradient = colors.get_qt_gradient(self.colors[::-1, :3], (0, 0), (self.width(), self.height()))
        # Set the brush to the gradient
        brush = QtGui.QBrush(gradient)
        painter.setBrush(brush)
        # Draw a rectangle the size of the QLabel
        painter.drawRect(0, self.height()-1, self.width()-1, -self.height()+1)


class ShortcutsDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.v_layout = QtWidgets.QVBoxLayout()
        self.h_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.v_layout)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.label_title = QtWidgets.QLabel('<big><b>List of shortcuts</b></big>', self)
        self.label_shortcuts = QtWidgets.QLabel('List of shortcuts', self)
        font = QtGui.QFont('Monospace')
        font.setStyleHint(QtGui.QFont.TypeWriter)
        self.label_shortcuts.setFont(font)
        try:
            with open('Content/shortcuts.txt', 'r') as f:
                st = f.readlines()
        except FileNotFoundError:
            st = ['File Not Found\n', 'Should be shortcuts.txt in Content folder']
            logger.error(st)
        s_text = ''.join(st)
        self.label_shortcuts.setText(s_text)

#         self.label_shortcuts.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
#         self.label_title.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.h_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.h_layout.addWidget(self.label_title)
        self.h_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.v_layout.addLayout(self.h_layout)
        self.v_layout.addSpacerItem(QtWidgets.QSpacerItem(10, 10, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding))
        self.v_layout.addWidget(self.label_shortcuts)


class StaticDataDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.label_title = QtWidgets.QLabel('<big><b>Create/Load Static Datasets</b></big>', self)
        font = QtGui.QFont('Monospace')
        font.setStyleHint(QtGui.QFont.TypeWriter)

        self.v_layout = QtWidgets.QVBoxLayout()
        self.v_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.v_layout.addWidget(self.label_title)
        self.v_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.setLayout(self.v_layout)
        self.compute_layout = QtWidgets.QHBoxLayout()
        self.conn_layout = QtWidgets.QHBoxLayout()
        self.load_layout = QtWidgets.QHBoxLayout()

        self.load_layout.addWidget(QtWidgets.QLabel('Choose Dataset '))
        self.load_dataname = QtWidgets.QLineEdit()
        self.load_layout.addWidget(self.load_dataname)
        self.load_button = QtWidgets.QPushButton('Load')
        self.load_button.setFixedWidth(100)
        self.load_button.clicked.connect(parent.load_static)
        self.load_layout.addWidget(self.load_button)

        self.conn_layout.addWidget(QtWidgets.QLabel('Choose Connections '))
        self.conn_dataname = QtWidgets.QLineEdit()
        self.conn_layout.addWidget(self.conn_dataname)
        self.conn_button = QtWidgets.QPushButton('Load')
        self.conn_button.setFixedWidth(100)
        self.conn_button.clicked.connect(parent.add_connectivity)
        self.conn_layout.addWidget(self.conn_button)

        self.compute_layout.addWidget(QtWidgets.QLabel('Choose Function '))
        self.compute_formula = QtWidgets.QLineEdit('np.var')
        self.compute_layout.addWidget(self.compute_formula)
        self.compute_button = QtWidgets.QPushButton('Compute')
        self.compute_button.setFixedWidth(100)
        self.compute_button.clicked.connect(parent.compute_static)
        self.compute_layout.addWidget(self.compute_button)

        self.v_layout.addLayout(self.load_layout)
        self.v_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.v_layout.addLayout(self.conn_layout)
        self.v_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.v_layout.addLayout(self.compute_layout)


class SelectCellDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.h_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.h_layout)

        self.h_layout.addWidget( QtWidgets.QLabel('Enter Cell Number '))
        self.cellnumber = QtWidgets.QLineEdit()
        self.h_layout.addWidget(self.cellnumber)
        self.find_button = QtWidgets.QPushButton('Find')
        self.find_button.setFixedWidth(100)
        self.find_button.clicked.connect(parent.show_cell)
        self.h_layout.addWidget(self.find_button)


class DsetConflictDialog(QtWidgets.QDialog):
    def __init__(self, parent, title, text, items):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setWindowTitle(title)
        self.v_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.v_layout)

        self.v_layout.addWidget(QtWidgets.QLabel(text))
        self.items_cbx = QtWidgets.QComboBox(self)
        self.items_cbx.addItems(items)
        self.v_layout.addWidget(self.items_cbx)
        self.ok_button = QtWidgets.QPushButton('OK')
        self.ok_button.setFixedWidth(100)
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setDefault(True)
        self.v_layout.addWidget(self.ok_button)

    def get_datapath(self):
        if self.exec_():
            return self.items_cbx.currentText()


class LogModel(QtCore.QAbstractTableModel):
    def __init__(self, data: list, columns=('time', 'file', 'function', 'line', 'level', 'message'),  parent=None):
        """
        Initialize a Abstract Table Model to store the log data

        Parameters
        ----------
        data: list of dict
        columns: tuple of str
        parent
        """
        super().__init__(parent)
        if len(data) == 0:
            data = [{k: '' for k in columns}]
        self.d_data = data
        self.columns = columns
        self._colors = {'CRITICAL': QtGui.QColor(250, 0, 00),
                        'ERROR': QtGui.QColor(200, 50, 50),
                        'WARNING': QtGui.QColor(255, 129, 0),
                        'INFO': QtGui.QColor(21, 171, 0),
                        'DEBUG': QtGui.QColor(49, 51, 53),
                        'NOTSET': QtGui.QColor(49, 51, 53),
                        '': QtGui.QColor(255, 255, 255)}

    def rowCount(self, parent):
        return len(self.d_data)

    def columnCount(self, parent):
        try:
            return len(self.d_data[0].keys())
        except IndexError:
            return 0

    def data(self, index, role):
        level = self.d_data[index.row()]['level']
        if not index.isValid():
            return QtCore.QVariant()
        elif role == QtCore.Qt.ForegroundRole:
            if index.column() != 4:
                return self._colors[level]
            else:
                return QtGui.QColor(255, 255, 255)
        elif role == QtCore.Qt.BackgroundColorRole and index.column() == 4:
            return self._colors[level]
        elif role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.d_data[index.row()][self.columns[index.column()]])
        else:
            return QtCore.QVariant()
        # Return the d_data at index.row() and index.column()

    def insertRow(self, row) -> bool:
        if self.d_data[0]['time'] == '':
            self.beginRemoveRows(QtCore.QModelIndex(), 0, 0)
            self.d_data.pop()
            self.endRemoveRows()
        self.beginInsertRows(QtCore.QModelIndex(), 0, 0)
        self.d_data.insert(0, row)
        self.endInsertRows()
        return True

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.columns[col])
        return QtCore.QVariant()


class QTableLog(logging.Handler):
    def __init__(self, parent, level=logging.NOTSET):
        super().__init__(level)
        self.model = LogModel([], parent=parent)
        self.widget = QtWidgets.QTableView(parent)
        self.widget.setModel(self.model)
        self.widget.setWordWrap(True)
        # self.widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

    def emit(self, record):
        msg = self.format(record)
        entry = msg.split(' :: ')
        d_entry = {k: v for k, v in zip(self.model.columns, entry)}
        self.model.insertRow(d_entry)
        self.widget.setVisible(False)
        self.widget.resizeColumnsToContents()
        self.widget.resizeRowsToContents()
        self.widget.setVisible(True)


class LogWindow(QtWidgets.QDialog):
    logger = logging.getLogger('Fishlog')

    def __init__(self, parent):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(600, 100)
        self.h_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.h_layout)
        self.log_handler = QTableLog(self, level=logging.DEBUG)
        self.logger.addHandler(self.log_handler)
        self.table = self.log_handler.widget
        self.h_layout.addWidget(self.table)


class AssignLoadedData(QtWidgets.QInputDialog):
    """Widget to manually assign data sets to default options."""
    def assign_data(self, loaded_data, options):
        str_question = "To what default data should {} be assigned? \n Or click Cancel to abort assigning".format(loaded_data)
        choice, ok_pressed = self.getItem(self, "Assign data", str_question, options, 0, False)
        if ok_pressed:
            return choice
        else:
            return None

class WarningDialog(QtWidgets.QDialog):
    """ Creates a warningDialog when there is too little memory to do the calculation.
    """
    def __init__(self, parent, mem):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.l1 = QtWidgets.QLabel("<center><big><b>Warning!</b></big>")
        self.l2 = QtWidgets.QLabel("<center> You have: </center> \n <center>" + str(round(int(mem) / 1000**3,2))
                 + " Gb available </center> \n <center> This is not enough for this correlation calculation </center>")
        self.btn = QtWidgets.QPushButton("return")

        # self.dialogIcon = QtWidgets.addWidget
        self.dialogTitle = self.setWindowTitle("Warning!")
        self.dialog_layout = QtWidgets.QVBoxLayout()
        self.dialog_layout.addWidget(self.dialogTitle)
        self.dialog_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.dialog_layout.addWidget(self.l1)
        self.dialog_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.dialog_layout.addWidget(self.l2)
        self.dialog_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))

        self.setLayout(self.dialog_layout)

# class ProgressBarDialog(QtWidgets.QDialog):
#     """ Creates a progressBarDialog to show the progress
#     it is not in use and can maybe be removed
#     """
#     def __init__(self, parent):
#         super().__init__(parent)
#         self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
#
#         #create layouts for progressBar and labels
#         self.layout = QtWidgets.QVBoxLayout()
#         self.bar_layout = QtWidgets.QHBoxLayout()
#
#         #create widgets
#         self.dialogTitle = self.setWindowTitle("Loading")
#         self.loading = QtWidgets.QLabel("<center><b>Loading please wait</b>")
#         self.btn = QtWidgets.QPushButton('test')
#         self.progressBar = QtGui.QProgressBar(self)
#
#         self.progressBar.setGeometry(200, 80, 250, 20)
#         self.btn.clicked.connect(self.progress)
#
#         # add widgets to the layout
#         self.layout.addWidget(self.dialogTitle)
#         self.layout.addWidget(self.loading)
#         self.layout.addWidget(self.btn)
#         self.bar_layout.addWidget(self.progressBar)
#
#
#         self.layout.addLayout(self.bar_layout)
#         self.setLayout(self.layout)
#
#     def progress(progress):
#         # # progressBarClass().show()
#         # result  = parent.compute_correlation_function
#         self.progressBar.setValue(progress)
#         # return result


class CorrelationDialog(QtWidgets.QDialog):
    """Class that creates the compute correlation dialog
    correlation_single: QComboBox
    correlation_multi: QComboBox
    press_button: QPushButton that connects to parent.compute_correlation_function
        to compute correlation between correlation_single and correlation_multi
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.label_title = QtWidgets.QLabel('<big><b>Compute Correlation</b></big>', self)
        font = QtGui.QFont('Monospace')
        font.setStyleHint(QtGui.QFont.TypeWriter)

        self.v_layout = QtWidgets.QVBoxLayout()
        self.v_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.v_layout.addWidget(self.label_title)
        self.v_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.setLayout(self.v_layout)
        self.correlation_single_multi_layout = QtWidgets.QHBoxLayout()
        self.press_button_layout = QtWidgets.QHBoxLayout()

        self.correlation_single_multi_layout.addWidget(QtWidgets.QLabel('Choose single data'))
        self.correlation_single = QtWidgets.QComboBox() # items are added in Fishualizer.py
        self.correlation_single_multi_layout.addWidget(self.correlation_single)

        self.correlation_single_multi_layout.addWidget(QtWidgets.QLabel('Choose multi data'))
        self.correlation_multi = QtWidgets.QComboBox() # items are added in Fishualizer.py
        self.correlation_single_multi_layout.addWidget(self.correlation_multi)

        self.press_button = QtWidgets.QPushButton("Confirm")
        self.press_button.clicked.connect(parent.compute_correlation_function) # compute correlation if pressed
        self.press_button_layout.addWidget(self.press_button)

        self.v_layout.addLayout(self.correlation_single_multi_layout)
        self.v_layout.addSpacerItem(QtWidgets.QSpacerItem(40, 10, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.v_layout.addLayout(self.press_button_layout)


class ColorModel(QtCore.QAbstractTableModel):
    def __init__(self, data: list, version, columns=('Dataset', 'Color', 'Alpha', 'Selected'), parent=None):
        """
        Initialize a Abstract Table Model to store the data used to assign user defined colors to clusters

        Parameters
        ----------
        data: list of dict
        columns: tuple of str
        parent
        """

        super(ColorModel, self).__init__()
        self._empty_row = {k: '' for k in columns}
        if version == "Cluster":
            self._empty_row['Alpha'] = 0.8
        if version == "Region":
            self._empty_row['Alpha'] = 0.1
        self._empty_row['Selected'] = 1

        if len(data) == 0:
            self.d_data = [self._empty_row.copy()]
        else:
            self.d_data = data
            self.d_data.append(self._empty_row.copy())
        self.columns = columns

    def rowCount(self, parent):
        return len(self.d_data)

    def columnCount(self, parent):
        try:
            return len(self.d_data[0].keys())
        except IndexError:
            return 0

    def data(self, index, role):
        if not index.isValid():
            return QtCore.QVariant()
        elif role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.d_data[index.row()][self.columns[index.column()]])
        else:
            return QtCore.QVariant()
        # Return the d_data at index.row() and index.column()

    def insertRow(self, row) -> bool:
        self.beginInsertRows(QtCore.QModelIndex(), 0, 0)
        self.d_data.append(row)
        self.endInsertRows()
        return True

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if index.row() == self.rowCount(self.parent()) - 1 and value != '':
            # Add a row if we are editing the last one
            self.insertRow(self._empty_row.copy())
        if index.column() == 1:
            # Editing the color, just get the hex value
            try:
                value = value[value.index('#'):]
            except ValueError:
                logger.error('Invalid color value')
                return False
        elif index.column() == 2:
            value = atof(value)
        self.d_data[index.row()][self.columns[index.column()]] = value
        self.dataChanged.emit(index, index, (QtCore.Qt.DisplayRole, ))
        return True

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.columns[col])
        return QtCore.QVariant()

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable


class ColorPickerCombo(QtWidgets.QComboBox):
    """
    Subclassing a combobox to make an automatic list of Qt colors with a colored square for each item
    """
    def __init__(self, parent) -> None:
        super().__init__(parent)
        for ix, c in enumerate(QtGui.QColor.colorNames()):
            self.addItem(f'{c} - {QtGui.QColor(c).name()}')
            self.setItemData(ix, QtGui.QColor(c), QtCore.Qt.DecorationRole)
        self.setEditable(True)


class TableDelegate(QtWidgets.QItemDelegate):
    """
    Take care of creating the widgets used to edit the cluster/color table so that we can fine tune them to our needs
    """
    def __init__(self, parent, datasets=()) -> None:
        super(TableDelegate, self).__init__()
        self.parent = parent
        self._datasets = datasets
        # Setting the locale for string to float conversion
        setlocale(LC_NUMERIC, '')

    def createEditor(self, parent: QtWidgets.QWidget, option: 'QStyleOptionViewItem', index: QtCore.QModelIndex) -> QtWidgets.QWidget:
        if index.column() == 0:
            # Making the first column ie the dataset column
            # We do it ourselves
            dset_combo = QtWidgets.QComboBox(parent)
            dset_combo.addItems(self._datasets)
            editor = dset_combo
        elif index.column() == 1:
            # Making the second column
            # We do it ourselves to make a color picker
            color_combo = ColorPickerCombo(parent)
            editor = color_combo
        elif index.column() == 3:
            # Checkbox column is weird
            return None
        else:
            # Alpha value editor
            alpha_editor = QtWidgets.QLineEdit(parent)
            alpha_editor.setValidator(QtGui.QDoubleValidator(0, 1, 3, parent))
            editor = alpha_editor
        header = self.parent.horizontalHeader()
        header.resizeSection(index.column(), editor.width())
        return editor

    def paint(self, painter: QtGui.QPainter, option: 'QStyleOptionViewItem', index: QtCore.QModelIndex) -> None:
        if index.column() == 3:
            new_rect = QtWidgets.QStyle.alignedRect(option.direction, QtCore.Qt.AlignCenter,
                                                    QtCore.QSize(option.decorationSize.width(),
                                                                 option.decorationSize.height()),
                                                    QtCore.QRect(option.rect.x(), option.rect.y(),
                                                                 option.rect.width(), option.rect.height())
                                                   )
            self.drawCheck(painter, option, new_rect,
                           QtCore.Qt.Unchecked if int(index.data()) == 0 else QtCore.Qt.Checked)
        else:
            super().paint(painter, option, index)

    def editorEvent(self, event, model, option, index):
        """
        Change the data in the model and the state of the checkbox
        if the user presses the left mousebutton and this cell is editable. Otherwise do nothing.
        """
        if index.column() == 3:
            if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                # Change the checkbox-state
                model.setData(index, 1 if int(index.data()) == 0 else 0, QtCore.Qt.EditRole)
                return True
        return super(TableDelegate, self).editorEvent(event, model, option, index)


class ColoringDialog(QtWidgets.QDialog):
    """
    Window showed to select which clusters to display and in which color
    """
    def __init__(self, parent,  datasets=(), data=None, version=None):  #version= 'Region' or 'Cluster'
        super().__init__(parent)
        if data is None:
            data = []

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.setMinimumSize(600, 100)
        self.v_layout = QtWidgets.QVBoxLayout()
        self.h_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.v_layout)
        # Table
        self.cluster_model = ColorModel(data, version)
        self.cluster_table = QtWidgets.QTableView(self)
        self.cluster_table.setModel(self.cluster_model)
        self.cluster_table.setItemDelegate(TableDelegate(self.cluster_table, datasets=datasets))
        header = self.cluster_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        # Buttons
        self.ok_button = QtWidgets.QPushButton("Apply")
        self.ok_button.clicked.connect(self.return_colors)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.quit)
        # Layouts
        self.h_layout.addSpacerItem(
            QtWidgets.QSpacerItem(600, 3, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.h_layout.addWidget(self.cancel_button)
        self.h_layout.addWidget(self.ok_button)

        self.v_layout.addWidget(self.cluster_table)
        self.v_layout.addLayout(self.h_layout)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        # Return value
        self.colors = None

    def return_colors(self):
        table_data = self.cluster_model.d_data
        self.colors = {row['Dataset']: [row['Dataset'], row['Color'], row['Alpha'], row['Selected']]
                       for row in table_data if row['Dataset'] != ''}
        self.colors = {k: [v[0], v[1] if v[1] != '' else None, v[3], v[2]] for k, v in self.colors.items()}
        self.accept()

    def quit(self):
        self.reject()


class OpenDataDialog(QtWidgets.QFileDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loadram_cbx = QtWidgets.QCheckBox('Load data in memory', self)
        self.ignorelags_cbx = QtWidgets.QCheckBox('Ignore time lags between Z-layers', self)
        self.forceinterp_cbx = QtWidgets.QCheckBox('Force time interpolation of Z-layers (even if time-aligned data is present)', self)
        self.ignoreunknowndata_cbx = QtWidgets.QCheckBox('Ignore data sets that are not recognised automatically', self)
        self.path_le = QtWidgets.QLineEdit('')
        self.path_completer = QtWidgets.QCompleter([], self.path_le)
        self.path_completer.setModel(QtWidgets.QDirModel(self.path_completer))
        self.path_completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.path_le.setCompleter(self.path_completer)
        self._c_path = ''
        self.path_le.textChanged.connect(self.path_modified)
        self.path_le.returnPressed.connect(self.path_validated)
        self.directoryEntered.connect(self.dir_entered)

    def dir_entered(self, directory: str):
        self.c_path = directory
        logger.debug(f'Directory {directory} entered')

    def getOpenFileName(self, parent, caption: str, directory: str, filter: str, initialFilter='', options=None):
        """
        Open a window to select a data file and set a few options.

        Parameters
        ----------
        parent:
        caption: str
            Window caption
        directory: str
            Current directory
        filter: str
            Valid file types
        initialFilter: str
        options:

        Returns
        -------
        filepath: str
            Path to data. '' if cancel
        filter: str
            Valid file extension
        ignorelags: bool
            Should we ignore the lags when loading data
        foreceinterp: bool
            Should the interpolation be forced, ie done even if already stored in the data file


        """
        self.setWindowTitle(caption)
        self.setDirectory(directory)
        self.setNameFilter(filter)
        self.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        if initialFilter != "":
            self.selectNameFilter(initialFilter)
        if options is not None:
            self.setOptions(options)
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.c_path = directory
        layout = self.layout()
        layout.addWidget(QtWidgets.QLabel('Options'), 4, 0)
        layout.addWidget(self.ignorelags_cbx, 4, 1)
        layout.addWidget(self.forceinterp_cbx, 5, 1)
        layout.addWidget(self.ignoreunknowndata_cbx, 6, 1)
        layout.addWidget(self.loadram_cbx, 7, 1)
        layout.addWidget(QtWidgets.QLabel('Path to data'), 8, 0)
        layout.addWidget(self.path_le, 8, 1)
        if self.exec_():
            return list(self.selectedFiles())[0], filter, self.ignorelags_cbx.isChecked(), \
                   self.forceinterp_cbx.isChecked(), self.ignoreunknowndata_cbx.isChecked(), \
                   self.loadram_cbx.isChecked()
        else:
            return "", filter, False, False, False, False

    @property
    def c_path(self):
        return self._c_path

    @c_path.setter
    def c_path(self, value):
        self.path_le.setText(value)
        self._c_path = value

    def path_modified(self, value):
        p = Path(value)
        if p.is_dir():
            self.setDirectory(p.as_posix())

    def path_validated(self):
        p = Path(self.path_le.text())
        if p.is_file() and p.suffix == '.h5':
            self.selectFile(p.as_posix())


# Extended combobox with search capabilities
# From: https://stackoverflow.com/questions/4827207/how-do-i-filter-the-pyqt-qcombobox-items-based-on-the-text-input
class ExtendedComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(ExtendedComboBox, self).__init__(parent)

        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setEditable(True)

        # add a filter model to filter matching items
        self.pFilterModel = QtCore.QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())

        # add a completer, which uses the filter model
        self.completer = QtWidgets.QCompleter(self.pFilterModel, self)
        # always show all (filtered) completions
        self.completer.setCompletionMode(QtWidgets.QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)

        # connect signals
        self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)

    # on selection of an item from the completer, select the corresponding item from combobox
    def on_completer_activated(self, text):
        if text:
            index = self.findText(text)
            self.setCurrentIndex(index)
            self.activated[str].emit(self.itemText(index))

    # on model change, update the models of the filter and completer as well
    def setModel(self, model):
        super(ExtendedComboBox, self).setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    # on model column change, update the model column of the filter and completer as well
    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super(ExtendedComboBox, self).setModelColumn(column)


class ComboColorMapDelegate(QtWidgets.QItemDelegate):
    """
    Take care of creating the widgets used to show the color map in the drop down menu
    """
    def __init__(self, parent, colormaps=()) -> None:
        super(ComboColorMapDelegate, self).__init__()
        self.parent = parent
        self._colormaps = colormaps
        self._colors = [colors.get_cmap(c_cmap) for c_cmap in colormaps]

    def paint(self, painter: QtGui.QPainter, option: 'QStyleOptionViewItem', index: QtCore.QModelIndex) -> None:
        # Geometry of current item
        width = option.rect.width() // 2
        height = option.rect.height()
        x = option.rect.x()
        y = option.rect.y()
        c_cmap = self._colormaps[index.row()]
        c_colors = self._colors[index.row()]
        painter.save()
        # Remove border
        painter.setPen(QtGui.QPen(0))
        # Get the gradient
        # Make sure not to use the alpha channel
        gradient = colors.get_qt_gradient(c_colors[::-1, :3], (x, y), (width, y))
        # Set the brush to the gradient
        brush = QtGui.QBrush(gradient)
        painter.setBrush(brush)
        # Draw a rectangle
        painter.drawRect(x, y, width, height - 1)
        painter.restore()
        text_rect = QtCore.QRect(x + width, y, width, height)
        if option.state & QtWidgets.QStyle.State_Selected:
            painter.fillRect(text_rect, option.palette.highlight())
        painter.drawText(text_rect, QtCore.Qt.AlignCenter, c_cmap)


class ColorMapCombobox(QtWidgets.QComboBox):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setItemDelegate(ComboColorMapDelegate(self, colors.get_all_cmaps()))
        self.setSizeAdjustPolicy(self.AdjustToContents)


class ColorMapDialog(QtWidgets.QDialog):
    def __init__(self, parent, title, text, items):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setWindowTitle(title)
        self.v_layout = QtWidgets.QVBoxLayout()
        self.h_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.v_layout)

        self.v_layout.addWidget(QtWidgets.QLabel(text))
        self.items_cbx = ColorMapCombobox(self)
        self.items_cbx.addItems(items)
        self.v_layout.addWidget(self.items_cbx)
        self.ok_button = QtWidgets.QPushButton('OK')
        self.ok_button.setFixedWidth(100)
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setDefault(True)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.setFixedWidth(100)
        self.cancel_button.clicked.connect(self.reject)

        self.h_layout.addWidget(self.cancel_button)
        self.h_layout.addWidget(self.ok_button)
        self.v_layout.addLayout(self.h_layout)

    def get_colormap(self):
        if self.exec_():
            return self.items_cbx.currentText(), True
        else:
            return '', False
