#!/usr/bin/env python3 

import sys
from PyQt5.QtWidgets import * 
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
import matplotlib
import matplotlib.pylab as pl
import pyqtgraph as pg
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy import constants
import time
import autoproc
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import glob
from results_gui import ResultsWindow

class SelectPeaksWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        #self.setWindowIcon(QIcon('./radial.png'))
        self.input_order=[] 
        
        self.createTable()
        self.createOpenButton()
        self.createPeaksTable()
        self.createPeaksRangeTable()
        self._ring_pen: Any = pg.mkPen("w", width=4)

        self._image_view: Any = pg.ImageView(view=pg.PlotItem())
        #self.setGeometry(1200, 1200, 1200, 600)
        self._peak_canvas: Any = pg.PlotDataItem()


        self.initial=0
        self.draw_peaks=0
        #self.toolbar = NavigationToolbar(self.canvas, self)
        self.label = QLabel(f"Radial averages:")

        self.createNextFileButton()
        self.createBackFileButton()
        self.createMarkPeaksButton()
        self.createConfirmPeaksButton()

        self.openButton.clicked.connect(self._open_button_clicked)
        self.nextFileButton.clicked.connect(self._next_file_button_clicked)
        self.backFileButton.clicked.connect(self._back_file_button_clicked)
        self.markPeaksButton.clicked.connect(self._mark_peaks_button_clicked)
        self.confirmPeaksButton.clicked.connect(self._confirm_button_clicked)


        self._vertical_layout_2:Any = QVBoxLayout()
        self._vertical_layout_2.addWidget(self.tableWidget)
        self._vertical_layout_2.addWidget(self.openButton)      
        self._vertical_layout_2.addWidget(self.peaksTableWidget)
        self._vertical_layout_2.addWidget(self.peaksRangeTableWidget)
        
        self._vertical_layout:Any = QVBoxLayout()
        self._vertical_layout.addWidget(self._image_view)
        self._vertical_layout.addWidget(self.nextFileButton)
        self._vertical_layout.addWidget(self.backFileButton)
        self._vertical_layout.addWidget(self.markPeaksButton)
        self._vertical_layout.addWidget(self.confirmPeaksButton)
        
        self._horizontal_layout: Any = QHBoxLayout()
        splitter_1 = QSplitter(QtCore.Qt.Vertical)

        self._horizontal_layout.addLayout(self._vertical_layout_2)
        self._horizontal_layout.addWidget(splitter_1)
        self._horizontal_layout.addLayout(self._vertical_layout)

        self.setLayout(self._horizontal_layout)
        self.show()

    def cell(self,var=""):
        item = QTableWidgetItem()
        item.setText(var)
        return item

    def createTable(self):
        self.tableWidget = QTableWidget()
  
        #Row count
        self.tableWidget.setRowCount(1) 
        rows=self.tableWidget.rowCount()
        #Column count
        self.tableWidget.setColumnCount(2)
        columns=self.tableWidget.columnCount()

        contents=[["Files path:", '/asap3/fs-bmx/gpfs/regae/2022/data/11015669/processed/centered/20220722/20220722_Au_free']]

        for i in range(rows):
            for j in range(columns):
                item = self.cell(contents[i][j])
                if j==0:
                    item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.tableWidget.setItem(i, j, item)
   
        #Table will fit the screen horizontally
        #self.tableWidget.horizontalHeader().setStretchLastSection(True)
        #self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def createPeaksTable(self):
        self.peaksTableWidget = QTableWidget()
  
        #Row count
        self.peaksTableWidget.setRowCount(1) 
        rows=self.tableWidget.rowCount()
        #Column count
        self.peaksTableWidget.setColumnCount(2)
        columns=self.peaksTableWidget.columnCount()

        contents=[["Number of peaks per file to follow:", '1']]

        for i in range(rows):
            for j in range(columns):
                item = self.cell(contents[i][j])
                if j==0:
                    item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.peaksTableWidget.setItem(i, j, item)
   
        #Table will fit the screen horizontally
        #self.peaksTableWidget.horizontalHeader().setStretchLastSection(True)
        #self.peaksTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def createPeaksRangeTable(self):
        self.peaksRangeTableWidget = QTableWidget()
  
        #Row count
        self.peaksRangeTableWidget.setRowCount(2) 
        rows=self.tableWidget.rowCount()
        #Column count
        self.peaksRangeTableWidget.setColumnCount(11)
        columns=self.peaksRangeTableWidget.columnCount()

 
        for i in range(rows):
            for j in range(columns):
                item = self.cell('')
                self.peaksRangeTableWidget.setItem(i, j, item)
   
        #Table will fit the screen horizontally
        #self.peaksRangeTableWidget.horizontalHeader().setStretchLastSection(True)
        #self.peaksRangeTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


    def createOpenButton(self):
        self.openButton = QPushButton("Open")
        self.openButton.setCheckable(True)
        self.openButton.setChecked(False)

    def createNextFileButton(self):
        self.nextFileButton = QPushButton("Next")
        self.nextFileButton.setCheckable(True)
        self.nextFileButton.setChecked(False)

    def createBackFileButton(self):
        self.backFileButton = QPushButton("Back")
        self.backFileButton.setCheckable(True)
        self.backFileButton.setChecked(False)

    def createMarkPeaksButton(self):
        self.markPeaksButton = QPushButton("Mark peaks")
        self.markPeaksButton.setCheckable(True)
        self.markPeaksButton.setChecked(False)

    def createConfirmPeaksButton(self):
        self.confirmPeaksButton = QPushButton("Confirm peak selection")
        self.confirmPeaksButton.setCheckable(True)
        self.confirmPeaksButton.setChecked(False)

    def _mark_peaks_button_clicked(self) -> None:
        # Manages clicks on the 'run' button.
        self.markPeaksButton.setDisabled(False)
        self.markPeaksButton.setCheckable(True)
        self.markPeaksButton.setChecked(False)

        self.canvas.figure.clear()
        self.canvas.draw()
        self.initial=0
        self.draw_peaks=1

        args=['-i']

        rows=self.tableWidget.rowCount()
        for i in range(rows):
            point=self.tableWidget.item(i,1).text()
            if point!='':
                args.append(point)
        rows=self.peaksTableWidget.rowCount()
        for i in range(rows):
            point=self.peaksTableWidget.item(i,1).text()
            self.number_of_peaks=int(point)
        self.show_plots(args)
        
        self.markPeaksButton.setCheckable(True)
        self.markPeaksButton.setChecked(False)

    def _open_button_clicked(self) -> None:
        # Manages clicks on the 'run' button.

        args=['-i']
        times=[]
        rows=self.tableWidget.rowCount()
        for i in range(rows):
            point=self.tableWidget.item(i,1).text()
            args.append(point)
   
        self.show_plots(args)
        self.openButton.setCheckable(False)
        self.openButton.setChecked(False)


    def _next_file_button_clicked(self) -> None:
        # Manages clicks on the 'run' button.
        if self.initial==len(self.input_order)-1:
            self.nextFileButton.setCheckable(False) 
            self.nextFileButton.setChecked(False)
        else:
            self.nextFileButton.setCheckable(True) 
            self.nextFileButton.setChecked(False)
            args=['-i']
	    
            rows=self.tableWidget.rowCount()
            for i in range(rows):
                point=self.tableWidget.item(i,1).text()
                args.append(point)

            self.initial+=1

            self.show_plots(args)
            


    def _back_file_button_clicked(self) -> None:
        # Manages clicks on the 'run' button.
        if self.initial==0:
            self.backFileButton.setCheckable(False) 
            self.backFileButton.setChecked(False)
        else:
            self.backFileButton.setCheckable(True) 
            self.backFileButton.setChecked(False)
            args=['-i']
            times=[]
            rows=self.tableWidget.rowCount()
            for i in range(rows):
                point=self.tableWidget.item(i,1).text()
                args.append(point)
        
            self.initial-=1
        
            self.show_plots(args)


 
    def _confirm_button_clicked(self) -> None:
        # Manages clicks on the 'run' button.
        self.confirmPeaksButton.setCheckable(True)
        self.confirmPeaksButton.setChecked(False)
        args=['-n',f'{self.number_of_peaks}','-i']
        args.append(self.input_order)
        columns=self.peaksRangeTableWidget.columnCount()
        max_column=0
        for j in range(columns):
            point=self.peaksRangeTableWidget.item(0,j).text()
            if point!='':
                max_column+=1
            

        for j in range(max_column):
            for i in range(2):
                point=self.peaksRangeTableWidget.item(i,j).text()
                args.append('-p')
                args.append(point)
        print(args)
        status=autoproc.run(args,self.init_table)
        if status==1:
             self.w=ResultsWindow()
             self.w.config_delay=self.config_delay
        self.w.show()

    def show_plots(self,raw_args=None):

        # create an axis

        parser = argparse.ArgumentParser(
        description="Plot peak positions according to angle.")
        parser.add_argument("-i", "--input", type=str, action="store",
            help="hdf5 input image")
                
        args = parser.parse_args(raw_args)


        files=list(glob.glob(f"{args.input}*.h5"))
        files.sort()
        self.input_order=files.copy()
        
        file_path=file_path=files[int(self.initial)]
        hf = h5py.File(file_path, 'r')
        x= np.array(hf['rad_x'])[:]
        y= np.array(hf['rad_sub'])
        data=np.array(list(zip(x,y)))
        print(x.shape,y.shape,data.shape)
        hf.close()

        self._image_view.setImage(
            data,
            autoLevels=False,
            autoRange=False)
        self._image_view.setMaximumHeight(1000)
        self._image_view.setMinimumWidth(1000)
        
        self._image_view.enableAutoRange("y")
        self._peak_canvas.getViewBox().invertY(False)
        self._image_view.show()

        



