#!/usr/bin/env python3
import sys

sys.path.append("../processing")
import autoproc
import convert

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from PyQt5.QtWidgets import *
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
import matplotlib
import matplotlib.pylab as pl

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from select_peaks_gui import SelectPeaksWindow
from results_gui import ResultsWindow


class App(QWidget):
    def __init__(self):
        ## REGAE AUTOPROC main window.
        super().__init__()
        self.title = "REGAE pump and probe"
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 500

        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon("../../../doc/regae.png"))
        self.setGeometry(self.left, self.top, self.width, self.height)

        disableWidgetsCheckBox = QCheckBox("&Skip data processing")

        self.pumpModeCheckBox = QCheckBox("&Pump and probe")
        self.magnetscanModeCheckBox = QCheckBox("&Magnet scan mode")

        self.createTable()
        self.createRunButton()
        self.createNextButton()
        self.createSkipButton()
        self.w = None
        self.w2 = None

        disableWidgetsCheckBox.toggled.connect(self.runButton.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.nextButton.setDisabled)
        disableWidgetsCheckBox.toggled.connect(self.tableWidget.setEnabled)
        disableWidgetsCheckBox.setChecked(True)

        self.magnetscanModeCheckBox.setChecked(False)
        self.pumpModeCheckBox.setChecked(False)

        self.magnetscanModeCheckBox.toggled.connect(self.pumpModeCheckBox.setDisabled)
        self.pumpModeCheckBox.toggled.connect(self.magnetscanModeCheckBox.setDisabled)

        self.runButton.clicked.connect(self._run_button_clicked)
        self.nextButton.clicked.connect(self._next_button_clicked)
        self.skipButton.clicked.connect(self._skip_button_clicked)

        self.layout = QVBoxLayout()
        self.layout.addWidget(disableWidgetsCheckBox)
        self.layout.addWidget(self.tableWidget)
        self.layout.addWidget(self.magnetscanModeCheckBox)
        self.layout.addWidget(self.pumpModeCheckBox)
        self.layout.addWidget(self.runButton)
        self.layout.addWidget(self.nextButton)
        self.layout.addWidget(self.skipButton)

        self.setLayout(self.layout)

        ## Show window

        self.show()

    def cell(self, var=""):
        ## Text in a cell for table.
        item = QTableWidgetItem()
        item.setText(var)
        return item

    ## Create table
    def createTable(self):
        self.tableWidget = QTableWidget()

        ## Row count
        self.tableWidget.setRowCount(12)
        rows = self.tableWidget.rowCount()
        ## Column count
        self.tableWidget.setColumnCount(2)
        columns = self.tableWidget.columnCount()

        contents = [
            ["Pedestal date:", "20221114_0"],
            ["Raw files date:", "20221114"],
            [
                "Raw files path:",
                "/asap3/fs-bmx/gpfs/regae/2022/data/11015669/raw/Au_free_mag",
            ],
            ["Sample label:", "Au"],
            ["Total number of shots per file:", "100"],
            ["Number of shots to accumulate:", "100"],
            ["Pump and probe zero delay (mm):", "0"],
            ["First file index:", "0"],
            ["First file scan stage (mm/A):", "0"],
            ["Last file index:", "12"],
            ["Last file scan stage (mm/A):", "2.4"],
            ["Scan stage step (mm/A):", "0.2"],
        ]

        # contents=[["Pedestal date:","20220722"],["Raw files date:","20220722"],["Raw files path:","/asap3/fs-bmx/gpfs/regae/2022/data/11015669/raw/220722_Au_free"],["Sample label:","Au_free"],["Total number of shots per file:","500"],["Number of shots to accumulate:","10"],["Pump and probe zero delay (mm):","0"],["First file index:","5"],["First file scan stage (mm/A):","3"],["Last file index:","10"],["Last file scan stage (mm/A):","4"], ["Scan stage step (mm/A):","0.2"]]

        # contents=[["Pedestal date:","20220524"],["Raw files date:","20220524"],["Raw files path:","/asap3/fs-bmx/gpfs/regae/2022/data/11015323/raw/220524_Au30"],["Sample label:","Au30"],["Total number of shots per file:","500"],["Number of shots to accumulate:","10"],["Pump and probe zero delay (mm):","227.5"],["First file index:","0"],["First file scan stage (mm/A):","220"],["Last file index:","81"],["Last file scan stage (mm/A):","240"], ["Scan stage step (mm/A):","0.5"]]

        for i in range(rows):
            for j in range(columns):

                item = self.cell(contents[i][j])
                if j == 0:
                    item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.tableWidget.setItem(i, j, item)

        ## Table will fit the screen horizontally
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def createRunButton(self):
        self.runButton = QPushButton("Run")
        self.runButton.setCheckable(True)
        self.runButton.setChecked(False)

    def createNextButton(self):
        self.nextButton = QPushButton("Next step: Peaks selection")
        self.nextButton.setCheckable(True)
        self.nextButton.setChecked(False)

    def createSkipButton(self):
        self.skipButton = QPushButton("Skip data processing: show results")
        self.skipButton.setCheckable(True)
        self.skipButton.setChecked(False)

    def _run_button_clicked(self) -> None:
        ## Manages clicks on the 'run' button.
        self.runButton.setDisabled(True)
        self.nextButton.setDisabled(True)
        self.tableWidget.setDisabled(True)
        args = []
        rows = self.tableWidget.rowCount()
        for i in range(rows):
            args.append(self.tableWidget.item(i, 1).text())
        status = convert.run(args)

        self.config_delay = args[6:]

        if status == 1:
            self.runButton.setDisabled(True)
            self.tableWidget.setDisabled(True)
            self.nextButton.setDisabled(False)

    def _next_button_clicked(self) -> None:
        ## Manages clicks on the 'Select peaks' button.
        self.runButton.setDisabled(True)
        self.nextButton.setDisabled(True)
        self.nextButton.setChecked(False)
        args = []
        rows = self.tableWidget.rowCount()
        for i in range(rows):
            args.append(self.tableWidget.item(i, 1).text())

        self.config_delay = args[6:]

        self.tableWidget.setDisabled(True)
        if self.w2 is None:
            self.w2 = SelectPeaksWindow()
            self.w2.init_table = args.copy()
            self.w2.config_delay = self.config_delay.copy()
            if self.magnetscanModeCheckBox.isChecked() == True:
                self.w2.mode = 0
            if self.pumpModeCheckBox.isChecked() == True:
                self.w2.mode = 1
        self.w2.show()

    def _skip_button_clicked(self) -> None:
        ## Manages clicks on the 'Show results' button.
        self.runButton.setDisabled(True)
        self.nextButton.setDisabled(True)
        self.skipButton.setCheckable(True)
        self.skipButton.setChecked(False)
        args = []
        rows = self.tableWidget.rowCount()
        for i in range(rows):
            args.append(self.tableWidget.item(i, 1).text())

        self.config_delay = args[6:]

        self.tableWidget.setDisabled(True)
        if self.w is None:
            self.w = ResultsWindow()
            self.w.config_delay = self.config_delay.copy()
            if self.magnetscanModeCheckBox.isChecked() == True:
                self.w.mode = 0
            if self.pumpModeCheckBox.isChecked() == True:
                self.w.mode = 1
        self.w.show()


if __name__ == "__main__":
    """
    REGAE AUTOPROC

    This software converts Jungfrau 1M images to powder radial plots for pump and probe experiments at REGAE - DESY.

    Copyright � 2021-2023 Deutsches Elektronen-Synchrotron DESY, a research centre of the Helmholtz Association.
    ## Authors:
    Ana Carolina Rodrigues

    mail: ana.rodrigues@desy.de
    """
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
