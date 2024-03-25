#!/usr/bin/env python3

import sys

sys.path.append("../processing")
import autoproc

from scipy import constants
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

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


def element_in_list(x, y):
    """
    Check if element x is in List y.

    Parameters
    ----------
    x: Any
        Element of search.
    y: List
        List in which the search will be perfomed.
    Returns
    ----------
    bool:
        x is in y.
    """
    for i in y:
        if x == i:
            return 1
    return 0


class RadialPlotsWindow(QWidget):
    def __init__(self):
        ## Results window with radial plots
        super().__init__()
        layout = QVBoxLayout()
        self.setWindowIcon(QIcon("../../../doc/radial.png"))
        self.beam_energy = 5.86 * 1e-13
        self.px_size = 75 * 1e-6
        self._lambda = (
            1e10
            * constants.h
            * constants.c
            / math.sqrt(
                (self.beam_energy) ** 2
                + (2 * self.beam_energy * constants.electron_mass * (constants.c**2))
            )
        )

        self.createTable()
        self.createOpenButton()
        self.createClearButton()
        self.figure = Figure(figsize=(20, 10))
        self.canvas = FigureCanvas(self.figure)
        self.initial = 0
        self.unit = "x"

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.label = QLabel(f"Radial averages:")

        unitComboBox = QComboBox()
        unitComboBox.addItems(["x", "2θ (deg)", "q vector(1/Å)"])
        unitComboBox.activated[str].connect(self.change_unit)

        self.goldPosCheckBox = QCheckBox("&Show gold reflections (theory)")
        self.goldPosCheckBox.stateChanged.connect(self._open_button_clicked)
        self.normIntCheckBox = QCheckBox("&Normalize intensity to first peak")
        self.normIntCheckBox.stateChanged.connect(self._open_button_clicked)
        self.normIntCheckBox.setChecked(False)
        self.normRadCheckBox = QCheckBox("&Normalize radius to first peak")
        self.normRadCheckBox.setChecked(False)
        self.normRadCheckBox.stateChanged.connect(self._open_button_clicked)
        self.hideOnCheckBox = QCheckBox("&Hide laser on")
        self.hideOnCheckBox.setChecked(False)
        self.hideOnCheckBox.stateChanged.connect(self._open_button_clicked)
        self.hideOffCheckBox = QCheckBox("&Hide laser off")
        self.hideOffCheckBox.stateChanged.connect(self._open_button_clicked)
        self.hideOffCheckBox.setChecked(False)

        unitLabel = QLabel("Peak position unit:")

        self.e1 = QLineEdit("")
        self.e1.editingFinished.connect(self.textchanged)
        label = QLabel("First peak positions")
        self.sub_layout = QHBoxLayout()
        self.sub_layout.addWidget(self.goldPosCheckBox)
        self.sub_layout.addWidget(self.hideOnCheckBox)
        self.sub_layout.addWidget(self.hideOffCheckBox)
        self.sub_layout.addWidget(self.normIntCheckBox)
        self.sub_layout.addWidget(self.normRadCheckBox)
        self.sub_layout.addWidget(label)
        self.sub_layout.addWidget(self.e1)

        self.sub_layout_2 = QHBoxLayout()
        self.sub_layout_2.addWidget(unitLabel)
        self.sub_layout_2.addWidget(unitComboBox)

        self.setYLim = QCheckBox("&Set y lim")
        self.setYLim.setChecked(False)
        self.setYLim.stateChanged.connect(self._open_button_clicked)
        self.e2 = QLineEdit("")
        self.e2.editingFinished.connect(self.textchanged_2)

        self.sub_layout_3 = QHBoxLayout()
        self.sub_layout_3.addWidget(self.setYLim)
        self.sub_layout_3.addWidget(self.e2)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.openButton.clicked.connect(self._open_button_clicked)
        self.clearButton.clicked.connect(self._clear_button_clicked)

        layout.addWidget(self.tableWidget)
        layout.addWidget(self.openButton)
        layout.addWidget(self.clearButton)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(self.sub_layout_3)
        layout.addLayout(self.sub_layout)
        layout.addLayout(self.sub_layout_2)

        self.setLayout(layout)

    def cell(self, var=""):
        ## Creates cell text for table
        item = QTableWidgetItem()
        item.setText(var)
        return item

    def change_unit(self, name):
        ## Options for xscale plot unit
        norm = ["x", "two_theta", "q"]
        if name == "x":
            self.unit = norm[0]
        if name == "2θ (deg)":
            self.unit = norm[1]
        elif name == "q vector(1/Å)":
            self.unit = norm[2]
        self.e1.clear()
        self.normIntCheckBox.setChecked(False)
        self.normRadCheckBox.setChecked(False)
        self._open_button_clicked()

    def createTable(self):
        ## Create table of time points to be plotted
        self.tableWidget = QTableWidget()

        ## Row count
        self.tableWidget.setRowCount(8)
        rows = self.tableWidget.rowCount()
        ## Column count
        self.tableWidget.setColumnCount(2)
        columns = self.tableWidget.columnCount()

        contents = [
            ["Time points:", "0.0"],
            ["", ""],
            ["", ""],
            ["", ""],
            ["", ""],
            ["", ""],
            ["", ""],
            ["", ""],
        ]

        for i in range(rows):
            for j in range(columns):
                item = self.cell(contents[i][j])
                if j == 0:
                    item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.tableWidget.setItem(i, j, item)

        # Table will fit the screen horizontally
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def createOpenButton(self):
        self.openButton = QPushButton("Open")
        self.openButton.setCheckable(True)
        self.openButton.setChecked(False)

    def createClearButton(self):
        self.clearButton = QPushButton("Clear")
        self.clearButton.setCheckable(True)
        self.clearButton.setChecked(False)

    def _clear_button_clicked(self) -> None:
        ## Manages clicks on the 'Clear' button. Clear plots.
        self.openButton.setDisabled(False)
        self.clearButton.setCheckable(True)
        self.clearButton.setChecked(False)
        self.e1.clear()
        self.normIntCheckBox.setChecked(False)
        self.normRadCheckBox.setChecked(False)
        self.canvas.figure.clear()
        self.canvas.draw()

        self.openButton.setCheckable(True)
        self.openButton.setChecked(False)

    def update_from_parent(self):
        ## Manages changes in normalization options.
        name = self.normComboBox.currentText()
        norm = ["_no", "_beg", "_sum"]
        if name == "no":
            self.Norm = norm[0]
        elif name == "begin":
            self.Norm = norm[1]
        elif name == "total_sum":
            self.Norm = norm[2]
        else:
            self.Norm = norm[0]
        name = self.subComboBox.currentText()
        if name == "yes" and self.Norm[-4:] != "_sub":
            self.Norm += "_sub"

        self.f_cal = float(self.f_cal_line.text())
        self.j_sol = float(self.j_sol_line.text())
        print("Norm", self.Norm, "f_cal", self.f_cal, "j_sol", self.j_sol)

    def _open_button_clicked(self) -> None:
        ## Manages clicks on the 'Open' button.

        self.update_from_parent()

        f = h5py.File(
            f"{self.Root}/{self.Date}/{self.file_name}{self.Norm}_time_scans.h5", "r"
        )
        time_points = np.array(f["scan_points"])
        self.time_points = []
        for i in time_points:
            self.time_points.append(round(float(i), 1))
        if self.initial == 0:
            print(f"{self.time_points}")

        args = [
            "-i",
            f"{self.Root}/{self.Date}/{self.file_name}{self.Norm}_time_scans.h5",
            "-o",
            f"{self.Root}/../plot/{self.Date}_{self.Label}_time_scans",
        ]
        times = []
        rows = self.tableWidget.rowCount()
        for i in range(rows):
            point = self.tableWidget.item(i, 1).text()

            if point != "":
                point_f = round(float(self.tableWidget.item(i, 1).text()), 1)
                is_time_point = element_in_list(point_f, self.time_points)
                if is_time_point == 1:
                    args.append("-t")
                    args.append(str(point_f))

        if len(args) == 4:
            args.append("-t")
            args.append("0.0")
        self.plot_rad(args)
        self.openButton.setChecked(False)

    def textchanged(self) -> None:
        ## Manages text changed in e1.
        text = self.e1.text()

    def textchanged_2(self) -> None:
        ## Manages text changed in e2.
        text = self.e2.text()

    def on_press(self, event):

        text = self.e1.text()
        if self.unit == "x":
            self.e1.setText(f"{text} {round(event.xdata)}")
        else:
            self.e1.setText(f"{text} {round(event.xdata,4)}")

        x = [round(event.xdata, 4)]
        y = [round(event.ydata, 4)]
        self.ax.scatter(x, y, marker="X", color="c", s=100)
        self.canvas.draw()

    def pixel_to_two_theta(self, x, f_cal, j_sol):
        ## Transformation from pixels to Bragg angles in degrees.
        tm = self.transfer_matrix(j_sol)
        two_theta_c = f_cal * x * self.px_size * 180 / (tm[0][1] * np.pi)

        return two_theta_c

    def pixel_to_q(self, x, f_cal, j_sol):
        ## Transformation from pixels to scattering vector q 1/A.
        tm = self.transfer_matrix(j_sol)
        q_c = f_cal * x * self.px_size / (tm[0][1] * self._lambda)

        return q_c

    def gold_peaks_pos(self, j_sol):

        reflections_list = [[1, 1, 1], [0, 0, 2], [0, 2, 2], [1, 1, 3]]
        ## Unit cell parameter gold in angstroms as reference sample for calibration factor determination.
        a = 407.3 * 1e-2

        q_t = []
        two_theta_t = []
        x_t = []
        for idx, i in enumerate(reflections_list):
            tm = self.transfer_matrix(j_sol)
            q = math.sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2) / a
            q_t.append(q)
            two_theta_t.append(self._lambda * q * 180 / np.pi)
            x_t.append(tm[0][1] * (self._lambda * q) / (self.px_size * self.f_cal))
        if self.unit == "q":
            return q_t
        if self.unit == "two_theta":
            return two_theta_t
        if self.unit == "x":
            return x_t

    def transfer_matrix(self, j_sol):
        ## From Linear-ray propagation model.
        e_kin = 3.66
        f = e_kin * (1.274 + 1.247 * e_kin) / (0.127 + j_sol**2)
        d_sol = 6.45
        d_sample = 5.5
        d_det = 10.42
        d1 = d_sol - d_sample
        d2 = d_det - d_sol
        m = [[1 - d2 / f, d1 * (1 - d2 / f) + d2], [-1 / f, 1 - d1 / f]]
        return m

    def text_to_list(self, text, n_dec=0):
        lst = []
        n = ""

        for i in text:
            if i != " ":
                n += i
            else:
                if n_dec != 0:
                    lst.append(float(n))
                else:
                    lst.append(int(n))
                n = ""
        lst.sort(reverse=True)
        return lst

    def plot_rad(self, raw_args=None):

        # Plot azimuthal integration spectra.

        parser = argparse.ArgumentParser(
            description="Plot peak positions according to angle."
        )
        parser.add_argument(
            "-i", "--input", type=str, action="store", help="hdf5 input image"
        )
        parser.add_argument(
            "-o", "--output", type=str, action="store", help="hdf5 output image"
        )
        parser.add_argument(
            "-t", "--time", action="append", help="time points in scan plots"
        )

        args = parser.parse_args(raw_args)
        self.canvas.figure.clear()

        self.ax = self.figure.add_subplot(111)
        ax = self.ax
        ax.clear()

        self.canvas.draw()

        try:
            self.legend.remove()
            self.initial += 1
        except:
            self.initial = 0

        file_path = f"{args.input}"
        hf = h5py.File(file_path, "r")
        point = list(args.time)
        conv_point = []
        for i in point:
            conv_point.append(float(i))

        conv_point.sort()
        point = conv_point

        n = round(len(point))
        colors = pl.cm.jet(np.linspace(0.1, 0.9, n))
        if self.setYLim.isChecked() == True:
            ax.set_ylim(-100, int(self.e2.text()))

        for idx, i in enumerate(point):

            if self.mode == 1:

                if self.hideOffCheckBox.isChecked() == False:
                    data_name = f"time_scan_{i}_laser_off"
                    data = np.array(hf[data_name])
                    x = data[:150, 0]
                    if self.unit == "two_theta":
                        x = self.pixel_to_two_theta(x, self.f_cal, self.j_sol)
                    if self.unit == "q":
                        x = self.pixel_to_q(x, self.f_cal, self.j_sol)

                    if self.normIntCheckBox.isChecked() == False:
                        norm_rad_0 = data[:150, 1]
                    else:
                        if self.unit == "x":
                            peak_pos = self.text_to_list(self.e1.text()[1:] + " ")
                        else:
                            peak_pos = self.text_to_list(
                                self.e1.text()[1:] + " ", n_dec=5
                            )
                        sub = []
                        for k in x:
                            sub.append(abs(k - peak_pos[idx]))
                        index = np.argmin(sub)
                        norm_factor = data[index, 1]
                        norm_rad_0 = data[:150, 1] / norm_factor
                    if self.normRadCheckBox.isChecked() == True:
                        peak_pos = self.text_to_list(self.e1.text()[1:] + " ", n_dec=5)
                        x = x / peak_pos[idx]
                    ax.plot(
                        x,
                        norm_rad_0,
                        ":",
                        color=colors[idx],
                        label=f"{i}ps laser off",
                        linewidth=2,
                    )

                if self.hideOnCheckBox.isChecked() == False:
                    data_name = f"time_scan_{i}_laser_on"
                    data = np.array(hf[data_name])

                    x = data[:150, 0]
                    if self.unit == "two_theta":
                        x = self.pixel_to_two_theta(x, self.f_cal, self.j_sol)
                    if self.unit == "q":
                        x = self.pixel_to_q(x, self.f_cal, self.j_sol)
                    if self.normIntCheckBox.isChecked() == False:
                        norm_rad_0 = data[:150, 1]
                    else:
                        if self.unit == "x":
                            peak_pos = self.text_to_list(self.e1.text()[1:] + " ")

                        else:
                            peak_pos = self.text_to_list(
                                self.e1.text()[1:] + " ", n_dec=5
                            )
                        sub = []
                        for k in x:
                            sub.append(abs(k - peak_pos[idx]))
                        index = np.argmin(sub)
                        norm_factor = data[index, 1]
                        norm_rad_0 = data[:150, 1] / norm_factor

                    if self.normRadCheckBox.isChecked() == True:
                        if self.unit == "x":
                            peak_pos = self.text_to_list(self.e1.text()[1:] + " ")
                        else:
                            peak_pos = self.text_to_list(
                                self.e1.text()[1:] + " ", n_dec=5
                            )
                        x = x / peak_pos[idx]

                    ax.plot(
                        x,
                        norm_rad_0,
                        ".-",
                        label=f"{i}ps laser on",
                        color=colors[idx],
                        linewidth=2,
                    )

                if self.goldPosCheckBox.isChecked() == True:
                    gold_peaks = self.gold_peaks_pos(self.j_sol)

                    for j in gold_peaks:

                        if self.normIntCheckBox.isChecked() == True:
                            y_t = np.linspace(0, 1.1, 100)
                        else:
                            y_t = np.linspace(0, 2 * max(norm_rad_0), 100)

                        x_t = j * np.ones(len(y_t))
                        if self.normRadCheckBox.isChecked() == True:
                            if self.unit == "x":
                                peak_pos = self.text_to_list(self.e1.text()[1:] + " ")
                            else:
                                peak_pos = self.text_to_list(
                                    self.e1.text()[1:] + " ", n_dec=5
                                )
                            x_t = x_t / peak_pos[idx]
                        ax.scatter(x_t, y_t, color=colors[idx], marker="|", alpha=0.4)
            else:
                data_name = f"magnet_scan_{i}"
                data = np.array(hf[data_name])
                x = data[:300, 0]
                if self.unit == "two_theta":
                    x = self.pixel_to_two_theta(x, self.f_cal, self.j_sol)
                if self.unit == "q":
                    x = self.pixel_to_q(x, self.f_cal, self.j_sol)

                if self.normIntCheckBox.isChecked() == False:
                    norm_rad_0 = data[:300, 1]

                else:
                    if self.unit == "x":
                        peak_pos = self.text_to_list(self.e1.text()[1:] + " ")
                    else:
                        peak_pos = self.text_to_list(self.e1.text()[1:] + " ", n_dec=5)
                    sub = []
                    for k in x:
                        sub.append(abs(k - peak_pos[idx]))
                    index = np.argmin(sub)
                    norm_factor = data[index, 1]
                    norm_rad_0 = data[:300, 1] / norm_factor

                if self.normRadCheckBox.isChecked() == True:
                    if self.unit == "x":
                        peak_pos = self.text_to_list(self.e1.text()[1:] + " ")
                    else:
                        peak_pos = self.text_to_list(self.e1.text()[1:] + " ", n_dec=5)
                    x = x / peak_pos[idx]

                ax.plot(
                    x, norm_rad_0, ".-", color=colors[idx], label=f"{i}A", linewidth=2
                )

                if self.goldPosCheckBox.isChecked() == True:
                    gold_peaks = self.gold_peaks_pos(i)

                    for j in gold_peaks:

                        if self.normIntCheckBox.isChecked() == True:
                            y_t = np.linspace(0, 1.1, 100)
                        else:
                            y_t = np.linspace(0, 2 * max(norm_rad_0), 100)

                        x_t = j * np.ones(len(y_t))
                        if self.normRadCheckBox.isChecked() == True:
                            if self.unit == "x":
                                peak_pos = self.text_to_list(self.e1.text()[1:] + " ")
                            else:
                                peak_pos = self.text_to_list(
                                    self.e1.text()[1:] + " ", n_dec=5
                                )
                            x_t = x_t / peak_pos[idx]
                        ax.scatter(x_t, y_t, color=colors[idx], marker="|", alpha=0.4)

        if self.normRadCheckBox.isChecked() == True:
            ax.set_xlabel("Normalized radius [a.u.]")
        else:
            if self.unit == "x":
                ax.set_xlabel("Peak position (pixel)")
            if self.unit == "two_theta":
                ax.set_xlabel(r"2$\theta$ (deg)")
            if self.unit == "q":
                ax.set_xlabel(r"q ($\AA^{-1}$)")

        if self.normIntCheckBox.isChecked() == True:
            ax.set_ylabel("Normalized intensity [a.u.]")
        else:
            ax.set_ylabel("Intensity [a.u.]")
        ax.set_facecolor("grey")

        ax.grid()
        self.legend = ax.legend()
        # plt.savefig(args.output+'.png')
        self.canvas.draw()


class ResultsWindow(QWidget):
    def __init__(self):
        ## Window with summary of results in a scan, displays the average of intensity, fwhm and peak positions for each point of the scan.

        super().__init__()
        self.beam_energy = 5.86 * 1e-13
        self.px_size = 75 * 1e-6
        self._lambda = (
            1e10
            * constants.h
            * constants.c
            / math.sqrt(
                (self.beam_energy) ** 2
                + (2 * self.beam_energy * constants.electron_mass * (constants.c**2))
            )
        )
        print(self._lambda)
        layout = QVBoxLayout()
        self.normComboBox = QComboBox()
        self.normComboBox.addItems(["no", "begin", "total_sum"])
        self.subComboBox = QComboBox()
        self.subComboBox.addItems(["yes", "no"])

        self.unit = "x"
        styleLabel = QLabel("Normalization:")
        styleLabel.setBuddy(self.normComboBox)
        subLabel = QLabel("Background subtraction:")
        subLabel.setBuddy(self.subComboBox)
        self.createTable()
        self.createOpenButton()
        self.createPlotButton()
        self.Norm = "_no"

        self.figure = Figure(figsize=(20, 10))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.setWindowIcon(QIcon("./intensity.png"))

        self.openButton.clicked.connect(self._open_button_clicked)
        self.plotButton.clicked.connect(self._plot_button_clicked)
        self.normComboBox.activated[str].connect(self.change_norm)
        self.subComboBox.activated[str].connect(self.change_sub)
        self.label = QLabel(f"Results summary:")
        self.w = None
        self.initial = 0

        unitComboBox = QComboBox()
        unitComboBox.addItems(["x", "2θ (deg)", "q vector(1/Å)"])
        unitComboBox.activated[str].connect(self.change_unit)

        self.hideOnCheckBox = QCheckBox("&Hide laser on")
        self.hideOnCheckBox.stateChanged.connect(self._open_button_clicked)
        self.hideOnCheckBox.setChecked(False)

        self.hideOffCheckBox = QCheckBox("&Hide laser off")
        self.hideOffCheckBox.stateChanged.connect(self._open_button_clicked)
        self.hideOffCheckBox.setChecked(False)

        self.hideSubCheckBox = QCheckBox("&Hide laser subtraction (on-off)")
        self.hideSubCheckBox.stateChanged.connect(self._open_button_clicked)
        self.hideSubCheckBox.setChecked(False)

        self.normRadCheckBox = QCheckBox("&Normalize peak positon to:")
        self.normRadCheckBox.stateChanged.connect(self._open_button_clicked)
        self.normRadCheckBox.setChecked(False)

        self.normIntCheckBox = QCheckBox("&Normalize intensity to:")
        self.normIntCheckBox.stateChanged.connect(self._open_button_clicked)
        self.normIntCheckBox.setChecked(False)

        self.normFwhmCheckBox = QCheckBox("&Normalize FWHM/R to:")
        self.normFwhmCheckBox.stateChanged.connect(self._open_button_clicked)
        self.normFwhmCheckBox.setChecked(False)

        self.e1 = QLineEdit("")
        self.e1.editingFinished.connect(self.textchanged_1)

        self.e2 = QLineEdit("")
        self.e2.editingFinished.connect(self.textchanged_2)

        self.e3 = QLineEdit("")
        self.e3.editingFinished.connect(self.textchanged_3)

        unitLabel = QLabel("Peak position unit:")
        label_2 = QLabel("f cal:")
        self.e4 = QLineEdit("1.099615")
        self.e4.editingFinished.connect(self.textchanged_4)
        label_3 = QLabel("j sol 67 (A):")
        self.e5 = QLineEdit("3.8")
        self.e5.editingFinished.connect(self.textchanged_5)

        self.sub_layout = QHBoxLayout()
        self.sub_layout.addWidget(self.hideOnCheckBox)
        self.sub_layout.addWidget(self.hideOffCheckBox)
        self.sub_layout.addWidget(self.hideSubCheckBox)

        self.sub_layout_3 = QHBoxLayout()
        self.sub_layout_3.addWidget(self.normIntCheckBox)
        self.sub_layout_3.addWidget(self.e1)

        self.sub_layout_4 = QHBoxLayout()
        self.sub_layout_4.addWidget(self.normRadCheckBox)
        self.sub_layout_4.addWidget(self.e2)

        self.sub_layout_5 = QHBoxLayout()
        self.sub_layout_5.addWidget(self.normFwhmCheckBox)
        self.sub_layout_5.addWidget(self.e3)

        self.sub_layout_6 = QHBoxLayout()
        self.sub_layout_6.addWidget(unitLabel)
        self.sub_layout_6.addWidget(unitComboBox)
        self.sub_layout_6.addWidget(label_2)
        self.sub_layout_6.addWidget(self.e4)
        self.sub_layout_6.addWidget(label_3)
        self.sub_layout_6.addWidget(self.e5)

        self.sub_layout_2 = QVBoxLayout()
        self.sub_layout_2.addLayout(self.sub_layout_3)
        self.sub_layout_2.addLayout(self.sub_layout_4)
        self.sub_layout_2.addLayout(self.sub_layout_5)
        self.sub_layout_2.addLayout(self.sub_layout_6)

        self.canvas.mpl_connect("button_press_event", self.on_press)

        layout.addWidget(self.label)
        layout.addWidget(self.tableWidget)
        layout.addWidget(styleLabel)
        layout.addWidget(self.normComboBox)
        layout.addWidget(subLabel)
        layout.addWidget(self.subComboBox)
        layout.addWidget(self.openButton)
        layout.addWidget(self.plotButton)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(self.sub_layout)
        layout.addLayout(self.sub_layout_2)
        self.setLayout(layout)

    def on_press(self, event):
        ## Manages clicks in plot
        if event.inaxes == self.ax1:
            self.e1.setText(f"{round(event.ydata,2)}")
            x = [round(event.xdata)]
            y = [round(event.ydata, 2)]
            self.ax1.scatter(x, y, marker="X", color="c", s=100)
            self.canvas.draw()

        if event.inaxes == self.ax2:
            self.e2.setText(f"{round(event.ydata,5)}")
            x = [round(event.xdata)]
            y = [round(event.ydata, 5)]
            self.ax2.scatter(x, y, marker="X", color="c", s=100)
            self.canvas.draw()
        if event.inaxes == self.ax3:
            self.e3.setText(f"{round(event.ydata,4)}")
            x = [round(event.xdata)]
            y = [round(event.ydata, 4)]
            self.ax3.scatter(x, y, marker="X", color="c", s=100)
            self.canvas.draw()

    def change_unit(self, name):
        ## Change unit of x axis
        norm = ["x", "two_theta", "q"]
        if name == "x":
            self.unit = norm[0]
        if name == "2θ (deg)":
            self.unit = norm[1]
        elif name == "q vector(1/Å)":
            self.unit = norm[2]
        self.e1.clear()
        self.e2.clear()
        self.e3.clear()
        self.normIntCheckBox.setChecked(False)
        self.normRadCheckBox.setChecked(False)
        self.normFwhmCheckBox.setChecked(False)
        self._open_button_clicked()

    def change_norm(self, name):
        ## Change normalization
        norm = ["_no", "_beg", "_sum"]
        if name == "no":
            self.Norm = norm[0]
        elif name == "begin":
            self.Norm = norm[1]
        elif name == "total_sum":
            self.Norm = norm[2]
        else:
            self.Norm = norm[0]

    def change_sub(self, name):
        ## Change radial subtraction
        if name == "yes" and self.Norm[-4:] != "_sub":
            self.Norm += "_sub"
        if self.w is not None:
            self.w.Norm = self.Norm

    def cell(self, var=""):
        ## Creates cell text for table
        item = QTableWidgetItem()
        item.setText(var)
        return item

    def textchanged_1(self) -> None:
        ## Detects text changes in e1
        text = self.e1.text()

    def textchanged_2(self) -> None:
        ## Detects text changes in e2
        text = self.e2.text()

    def textchanged_3(self) -> None:
        ## Detects text changes in e3
        text = self.e3.text()

    def textchanged_4(self) -> None:
        ## Detects text changes in e4
        text = self.e4.text()

    def textchanged_5(self) -> None:
        ## Detects text changes in e5
        text = self.e5.text()

    def createTable(self):
        ## Create table for input arguments
        self.tableWidget = QTableWidget()

        # Row count
        self.tableWidget.setRowCount(4)
        rows = self.tableWidget.rowCount()
        # Column count
        self.tableWidget.setColumnCount(2)
        columns = self.tableWidget.columnCount()

        contents = [
            [
                "Root path:",
                "/asap3/fs-bmx/gpfs/regae/2022/data/11016614/processed/average",
            ],
            ["Raw files date:", "20221114"],
            ["Label:", "Au"],
            ["Peak number:", "0"],
        ]
        # contents=[["Root path:","/asap3/fs-bmx/gpfs/regae/2022/data/11015323/processed/average"],["Raw files date:","20220524"],["Label:","Au30"], ["Peak number:","0"]]

        for i in range(rows):
            for j in range(columns):

                item = self.cell(contents[i][j])
                if j == 0:
                    item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.tableWidget.setItem(i, j, item)

        # Table will fit the screen horizontally
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def createOpenButton(self):
        self.openButton = QPushButton("Open")
        self.openButton.setCheckable(True)
        self.openButton.setChecked(False)

    def createPlotButton(self):
        self.plotButton = QPushButton("Plot Radial averages")
        self.plotButton.setCheckable(False)
        self.plotButton.setChecked(False)

    def _open_button_clicked(self) -> None:
        ## Manages clicks on the 'open' button.
        self.openButton.setDisabled(False)
        self.openButton.setChecked(False)
        self.tableWidget.setDisabled(False)
        args = []
        rows = self.tableWidget.rowCount()
        for i in range(rows):
            args.append(self.tableWidget.item(i, 1).text())

        self.Root = args[0]
        self.Date = args[1]
        self.Label = args[2]

        scan_file = f"{self.Root}/{self.Date}/{self.Date}_{self.Label}{self.Norm}"
        args = [
            "-i",
            scan_file,
            "-n",
            f"{args[3]}",
            "-o",
            f"{self.Root}/../plot/{self.Date}_{self.Label}{self.Norm}",
            "-fi",
            f"{int(self.config_delay[1])}",
        ]
        self.plot_scan(args)

        self.plotButton.setCheckable(True)
        self.plotButton.setChecked(False)

    def _plot_button_clicked(self) -> None:
        ## Manages clicks on the 'run' button.
        self.openButton.setDisabled(False)
        self.tableWidget.setDisabled(False)
        args = []
        for i in range(3):
            args.append(self.tableWidget.item(i, 1).text())

        self.Root = args[0]
        self.Date = args[1]
        self.Label = args[2]
        scan_file = f"{self.Root}/{self.Date}/{self.Date}_{self.Label}"
        args = ["-i", scan_file, "-o", f"{self.Root}/../plot/{self.Date}_{self.Label}"]

        self.file_name = f"{self.Date}_{self.Label}"

        if self.w is None:
            self.w = RadialPlotsWindow()
            self.w.mode = self.mode
            self.w.Root = self.Root
            self.w.Date = self.Date
            self.w.f_cal_line = self.e4
            self.w.j_sol_line = self.e5
            self.w.Label = self.Label
            self.w.file_name = self.file_name
            self.w.normComboBox = self.normComboBox
            self.w.subComboBox = self.subComboBox

        self.w.show()
        self.plotButton.setCheckable(True)
        self.plotButton.setChecked(False)

    def pixel_to_two_theta(self, x, f_cal, j_sol):
        ## Transforms pixel to Bragg angles in degrees.
        tm = self.transfer_matrix(j_sol)
        two_theta_c = f_cal * x * self.px_size * 180 / (tm[0][1] * np.pi)
        return two_theta_c

    def pixel_to_q(self, x, f_cal, j_sol):
        ## Transforms pixel to scattering vector q in 1/A.
        tm = self.transfer_matrix(j_sol)
        q_c = f_cal * x * self.px_size / (tm[0][1] * self._lambda)
        return q_c

    def transfer_matrix(self, j_sol):
        ## From Linear-ray propagation model.
        e_kin = 3.66
        f = e_kin * (1.274 + 1.247 * e_kin) / (0.127 + j_sol**2)
        d_sol = 6.45
        d_sample = 5.5
        d_det = 10.42
        d1 = d_sol - d_sample
        d2 = d_det - d_sol
        m = [[1 - d2 / f, d1 * (1 - d2 / f) + d2], [-1 / f, 1 - d1 / f]]
        return m

    def plot_scan(self, raw_args=None):

        ##Plot scan comparison.

        parser = argparse.ArgumentParser(
            description="Plot peak positions according to angle."
        )
        parser.add_argument(
            "-i", "--input", type=str, action="store", help="hdf5 input image"
        )
        parser.add_argument(
            "-fi", "--first", type=str, action="store", help="hdf5 input image"
        )
        parser.add_argument(
            "-n",
            "--n_peak",
            type=int,
            action="store",
            help="peak n of azimuthal integration",
        )
        parser.add_argument(
            "-o", "--output", type=str, action="store", help="plot output image"
        )
        args = parser.parse_args(raw_args)

        n = args.n_peak
        self.canvas.figure.clear()

        self.canvas.draw()

        if self.initial == 0:
            file_path = f"{args.input}_{args.first}.h5"
            hf = h5py.File(file_path, "r")
            peak_pos = np.array(hf["peak_position"])
            for idx, i in enumerate(peak_pos):
                print(f"peak_{idx}:{round(i)}")
            self.initial = 1
        else:
            try:
                self.initial = 1
                self.legend.remove()
            except:
                self.initial = 0

        t0_stage = round(float(self.config_delay[0]), 2)
        first_index = int(self.config_delay[1])
        first_pos_stage = round(float(self.config_delay[2]), 2)
        last_index = int(self.config_delay[3])
        last_pos_stage = round(float(self.config_delay[4]), 2)
        delay_step = round((3.33 / 0.5) * float(self.config_delay[5]), 2)

        if self.mode == 0:
            size = np.arange(first_index, last_index + 1, 1)
            start = first_pos_stage - t0_stage
            finish = last_pos_stage - t0_stage
            label = np.arange(start, finish + 0.1, float(self.config_delay[5]))
        else:

            if float(last_pos_stage) > float(first_pos_stage):
                size = np.flip(np.arange(first_index, last_index + 1, 1))
                start = -1 * (last_pos_stage - t0_stage) * 6.66
                finish = (t0_stage - first_pos_stage) * 6.66
                label = np.arange(start, finish + 1, delay_step)
            else:
                ####not tested yet!
                size = np.arange(first_index, last_index + 1, 1)
                start = -1 * (first_pos_stage - t0_stage) * 6.66
                finish = (t0_stage - last_pos_stage) * 6.66
                label = np.arange(start, finish + 1, delay_step)

        data_name = "intensity"

        ax1 = self.figure.add_subplot(131)
        self.ax1 = ax1
        ax1.clear()
        rad_0 = []
        rad_0_err = []
        rad_0_on = []
        rad_0_on_err = []
        sub = []
        sub_err = []

        for idx, i in enumerate(size[:]):
            file_path = f"{args.input}_{i}.h5"
            hf = h5py.File(file_path, "r")
            if self.mode == 0:

                rad_0_on.append(np.array(hf[data_name][n]))
                rad_0_on_err.append(np.array(hf[data_name + "_err"][n]))

            else:
                if idx % 2 == 0:
                    try:
                        rad_0_on.append(np.array(hf[data_name][n]))
                        rad_0_on_err.append(np.array(hf[data_name + "_err"][n]))
                    except ValueError:
                        rad_0_on.append(0)
                        rad_0_on_err.append(0)

                if idx % 2 != 0:
                    try:
                        rad_0.append(np.array(hf[data_name][n]))
                        rad_0_err.append(np.array(hf[data_name + "_err"][n]))
                    except ValueError:
                        rad_0.append(0)
                        rad_0_err.append(0)

        if self.normIntCheckBox.isChecked() == True:
            norm_factor = float(self.e1.text())
            rad_0_on[:] = [x / norm_factor for x in rad_0_on]
            rad_0_on_err[:] = [x / norm_factor for x in rad_0_on_err]
            if self.mode == 1:
                rad_0[:] = [x / norm_factor for x in rad_0]
                rad_0_err[:] = [x / norm_factor for x in rad_0_err]

        if self.mode == 1 and self.hideSubCheckBox.isChecked() == False:
            for idx, i in enumerate(rad_0):
                try:
                    sub.append(rad_0_on[idx] - i)
                    sub_err.append(
                        math.sqrt((rad_0_on_err[idx]) ** 2 + (rad_0_err[idx]) ** 2)
                    )
                except ValueError:
                    sub.append(0)
                    sub_err.append(0)
            ax1.plot(label, sub, ".:", color="red")
            ax1.errorbar(label, sub, yerr=sub_err, color="red")

        if self.mode == 0 or (
            self.mode == 1 and self.hideOnCheckBox.isChecked() == False
        ):
            ax1.scatter(label, rad_0_on, color="blue", marker="o")
            ax1.errorbar(
                label, rad_0_on, yerr=rad_0_on_err, color="blue", label="laser_on"
            )
        if self.mode == 1 and self.hideOffCheckBox.isChecked() == False:
            ax1.scatter(label, rad_0, color="grey", marker="o", label="laser_off")
            ax1.errorbar(label, rad_0, yerr=rad_0_err, color="grey")

        data_name = "peak_position"

        ax2 = self.figure.add_subplot(132)
        self.ax2 = ax2
        ax2.clear()
        rad_0 = []
        rad_0_err = []
        rad_0_on = []
        rad_0_on_err = []
        sub = []
        sub_err = []
        rad_theory = []

        for idx, i in enumerate(size[:]):

            file_path = f"{args.input}_{i}.h5"
            hf = h5py.File(file_path, "r")

            f_cal = float(self.e4.text())
            j_sol = float(self.e5.text())

            if self.mode == 0:
                transf_factor = self.transfer_matrix(label[idx])
                x = np.array(hf[data_name][n])
                if self.unit == "two_theta":
                    x = self.pixel_to_two_theta(x, f_cal, j_sol)
                if self.unit == "q":
                    x = self.pixel_to_q(x, f_cal, j_sol)
                rad_0_on.append(x)

                x = np.array(hf[data_name + "_err"][n])
                if self.unit == "two_theta":
                    x = self.pixel_to_two_theta(x, f_cal, j_sol)
                if self.unit == "q":
                    x = self.pixel_to_q(x, f_cal, j_sol)
                rad_0_on_err.append(x)

            else:
                if idx % 2 == 0:
                    try:
                        x = np.array(hf[data_name][n])
                        if self.unit == "two_theta":
                            x = self.pixel_to_two_theta(x, f_cal, j_sol)
                        if self.unit == "q":
                            x = self.pixel_to_q(x, f_cal, j_sol)
                        rad_0_on.append(x)
                        x = np.array(hf[data_name + "_err"][n])
                        if self.unit == "two_theta":
                            x = self.pixel_to_two_theta(x, f_cal, j_sol)
                        if self.unit == "q":
                            x = self.pixel_to_q(x, f_cal, j_sol)
                        rad_0_on_err.append(x)
                    except ValueError:
                        rad_0_on.append(0)
                        rad_0_on_err.append(0)

                if idx % 2 != 0:
                    try:
                        x = np.array(hf[data_name][n])
                        if self.unit == "two_theta":
                            x = self.pixel_to_two_theta(x, f_cal, j_sol)
                        if self.unit == "q":
                            x = self.pixel_to_q(x, f_cal, j_sol)
                        rad_0.append(x)
                        x = np.array(hf[data_name + "_err"][n])
                        if self.unit == "two_theta":
                            x = self.pixel_to_two_theta(x, f_cal, j_sol)
                        if self.unit == "q":
                            x = self.pixel_to_q(x, f_cal, j_sol)
                        rad_0_err.append(x)
                    except ValueError:
                        rad_0.append(0)
                        rad_0_err.append(0)

        if self.normRadCheckBox.isChecked() == True:
            norm_factor = float(self.e2.text())
            rad_0_on[:] = [x / norm_factor for x in rad_0_on]
            rad_0_on_err[:] = [x / norm_factor for x in rad_0_on_err]
            if self.mode == 1:
                rad_0[:] = [x / norm_factor for x in rad_0]
                rad_0_err[:] = [x / norm_factor for x in rad_0_err]

        if self.mode == 1 and self.hideSubCheckBox.isChecked() == False:
            for idx, i in enumerate(rad_0):
                try:
                    sub.append(rad_0_on[idx] - i)
                    sub_err.append(
                        math.sqrt((rad_0_on_err[idx]) ** 2 + (rad_0_err[idx]) ** 2)
                    )
                except ValueError:
                    sub.append(0)
                    sub_err.append(0)
            ax2.plot(label, sub, ".:", color="red", label="on-off")
            ax2.errorbar(label, sub, yerr=sub_err, color="red")

        if self.mode == 0 or (
            self.mode == 1 and self.hideOnCheckBox.isChecked() == False
        ):
            ax2.scatter(label, rad_0_on, color="blue", marker="o", label="laser_on")
            ax2.errorbar(label, rad_0_on, yerr=rad_0_on_err, color="blue")
        if self.mode == 1 and self.hideOffCheckBox.isChecked() == False:
            ax2.scatter(label, rad_0, color="grey", marker="o", label="laser_off")
            ax2.errorbar(label, rad_0, yerr=rad_0_err, color="grey")

        data_name = "fwhm_radius"

        ax3 = self.figure.add_subplot(133)
        self.ax3 = ax3
        ax3.clear()
        rad_0 = []
        rad_0_err = []
        rad_0_on = []
        rad_0_on_err = []
        sub = []
        sub_err = []

        for idx, i in enumerate(size[:]):
            file_path = f"{args.input}_{i}.h5"
            hf = h5py.File(file_path, "r")
            if self.mode == 0:
                rad_0_on.append(np.array(hf[data_name][n]))
                rad_0_on_err.append(np.array(hf[data_name + "_err"][n]))

            else:
                if idx % 2 == 0:
                    try:
                        rad_0_on.append(np.array(hf[data_name][n]))
                        rad_0_on_err.append(np.array(hf[data_name + "_err"][n]))
                    except ValueError:
                        rad_0_on.append(0)
                        rad_0_on_err.append(0)

                if idx % 2 != 0:
                    try:
                        rad_0.append(np.array(hf[data_name][n]))
                        rad_0_err.append(np.array(hf[data_name + "_err"][n]))
                    except ValueError:
                        rad_0.append(0)
                        rad_0_err.append(0)

        if self.normFwhmCheckBox.isChecked() == True:
            norm_factor = float(self.e3.text())
            rad_0_on[:] = [x / norm_factor for x in rad_0_on]
            rad_0_on_err[:] = [x / norm_factor for x in rad_0_on_err]
            if self.mode == 1:
                rad_0[:] = [x / norm_factor for x in rad_0]
                rad_0_err[:] = [x / norm_factor for x in rad_0_err]

        if self.mode == 1 and self.hideSubCheckBox.isChecked() == False:
            for idx, i in enumerate(rad_0):
                try:
                    sub.append(rad_0_on[idx] - i)
                    sub_err.append(
                        math.sqrt((rad_0_on_err[idx]) ** 2 + (rad_0_err[idx]) ** 2)
                    )
                except ValueError:
                    sub.append(0)
                    sub_err.append(0)
            ax3.plot(label, sub, ".:", color="red")
            ax3.errorbar(label, sub, yerr=sub_err, color="red")

        if self.mode == 0 or (
            self.mode == 1 and self.hideOnCheckBox.isChecked() == False
        ):
            ax3.scatter(label, rad_0_on, color="blue", marker="o", label="laser_on")
            ax3.errorbar(label, rad_0_on, yerr=rad_0_on_err, color="blue")

        if self.mode == 1 and self.hideOffCheckBox.isChecked() == False:
            ax3.scatter(label, rad_0, color="grey", marker="o", label="laser_off")
            ax3.errorbar(label, rad_0, yerr=rad_0_err, color="grey")

        file_path = f"{args.input}_{first_index}.h5"
        hf = h5py.File(file_path, "r")
        peak_pos = np.array(hf["peak_position"])

        self.legend = ax2.legend()

        if self.mode == 1:
            ax1.set_xlabel("Time delay (ps)")
            ax2.set_xlabel("Time delay (ps)")
            ax3.set_xlabel("Time delay (ps)")
        else:
            ax1.set_xlabel("Sol67 current (A)")
            ax2.set_xlabel("Sol67 current (A)")
            ax3.set_xlabel("Sol67 current (A)")

        ax1.set_ylabel("Normal. intensity (a.u.)")
        if self.unit == "x":
            ax2.set_ylabel("Peak position (pixel)")
        if self.unit == "two_theta":
            ax2.set_ylabel(r"2$\theta$ (deg)")
        if self.unit == "q":
            ax2.set_ylabel(r"q ($\AA^{-1}$)")
        ax3.set_ylabel("FWHM/R (a.u.)")

        ax1.grid()
        ax2.grid()
        ax3.grid()

        ##refresh canvas
        self.canvas.draw()
