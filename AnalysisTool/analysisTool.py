import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QSplitter, QLabel,\
    QSizePolicy, QAction, QToolBar, QStatusBar, QFileDialog, QSpacerItem
    
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent

from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from UGParameterEstimator import Result

class AnalysisTool(QApplication):
    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        self.build_ui()

    def build_ui(self):
        # build a main GUI window
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle('Analysis tool')
        self.main_window.show()

        self.canvas = pg.ScatterPlotWidget()

        # add a label to the main window
        splitter = QSplitter(self.main_window)
        splitter.addWidget(self.canvas)
        self.main_window.setCentralWidget(splitter)


        # add a toolbar with an action button to the main window
        action = QAction('Open result file', self)
        action.triggered.connect(self.openFile)
        toolbar = QToolBar()
        toolbar.addAction(action)
        self.main_window.addToolBar(toolbar)

        # self.main_window.addToolBar(Qt.BottomToolBarArea,
                        # NavigationToolbar(self.canvas, self.main_window))

        # add a status bar to the main window
        self.status_openfile = QLabel("No file open.")
        self.status_data = QLabel("")
        status_bar = QStatusBar()
        status_bar.addWidget(self.status_openfile, 1)
        status_bar.addWidget(self.status_data, 0)
        self.main_window.setStatusBar(status_bar)

        self.result = None
        self.index = 0
        self.plotdata = [[], [], []]
        self.currentData = [[], [], []]

        self.installEventFilter(self)

    def openFile(self):
        resultfilename = QFileDialog.getOpenFileName(self.main_window, "Select result file", filter="Result files (*.pkl)")[0]
        self.result = Result.load(resultfilename)
        self.status_openfile.setText(resultfilename)
        self.status_data.setText("Iterations: " + str(self.result.iterationCount))
        
        fields = []
        for p in self.result.metadata["parametermanager"].parameters:
            fields.append((p.name, {}))

        fields.append(("norm", {}))
        self.canvas.setFields(fields)

        self.index = 0
        self.plotNext()

    def eventFilter(self, obj, event):
        
        if isinstance(event, QKeyEvent) and event.type() == QEvent.KeyPress and isinstance(obj, QMainWindow):
            if event.key() == Qt.Key_Space:
                self.plotNext()
                return True
        
        return False


    def plotNext(self):
        if self.result is None:
            return

        if self.index == len(self.result.evaluations):
            return

        data, tag, _ = self.result.evaluations[self.index]

        target = self.result.metadata["target"]
        targetdata = target.getNumpyArray()

        parameterdata = []

        self.plotdata[0].extend(self.currentData[0])
        self.plotdata[1].extend(self.currentData[1])
        self.plotdata[2].extend(self.currentData[2])

        self.currentData = [[],[],[]]

        for ev in data:
            parameterdata.append(ev.parameters)

            measurement = ev.getNumpyArrayLike(target)
            residual = measurement - targetdata

            self.currentData[2].append(np.linalg.norm(residual))

        parameterdata = list(map(list, zip(*parameterdata)))
        
        self.currentData[0] = parameterdata[0]
        self.currentData[1] = parameterdata[1]

        self.canvas.plot(self.plotdata, self.currentData)
        self.index += 1


if __name__ == '__main__':
    app = AnalysisTool(sys.argv)
    sys.exit(app.exec_())
 

