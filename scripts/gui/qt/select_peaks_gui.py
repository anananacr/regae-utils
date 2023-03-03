#!/usr/bin/env python3 

import sys
from PyQt5.QtWidgets import * 
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
import matplotlib
import matplotlib.pylab as pl
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy import constants
import time
import autoproc
import os
import pandas as pd
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

        layout = QVBoxLayout()
        #self.setWindowIcon(QIcon('./radial.png'))
        self.input_order=[] 
        
        self.createTable()
        self.createOpenButton()
        self.createPeaksRangeTable()
        self.setWindowIcon(QIcon('./select.png')) 

        self.figure = Figure(figsize=(10,10))
        self.canvas = FigureCanvas(self.figure)
        self.initial=0
        self.draw_peaks=0
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.label = QLabel(f"Radial averages:")

        self.createImportFileButton()
        self.createExportFileButton()

        self.createNextFileButton()
        self.createBackFileButton()
        self.createMarkPeaksButton()
        self.createConfirmPeaksButton()

        self.setYLim = QCheckBox("&Set y lim")
        self.setYLim.setChecked(False)
        self.setYLim.stateChanged.connect(self._open_button_clicked)
        self.e2 = QLineEdit('')
        self.e2.editingFinished.connect(self.textchanged)

        self.sub_layout_3=QHBoxLayout()
        self.sub_layout_3.addWidget(self.setYLim)
        self.sub_layout_3.addWidget(self.e2)

        self.samePeaksCheckBox = QCheckBox("&Apply same peak(s) range to all files:")
        self.samePeaksCheckBox.setChecked(False)

        self.sub_layout_2=QHBoxLayout()
        self.sub_layout_2.addWidget(self.importFileButton)
        self.sub_layout_2.addWidget(self.exportFileButton)


        self.e1 = QLineEdit('0')
        self.e1.editingFinished.connect(self.textchanged)
        label=QLabel('File Number')
        self.sub_layout=QHBoxLayout()
        self.sub_layout.addWidget(self.backFileButton)
        self.sub_layout.addWidget(label)
        self.sub_layout.addWidget(self.e1)
        self.sub_layout.addWidget(self.nextFileButton)

        self.count_clicks_open=0
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.openButton.clicked.connect(self._open_button_clicked)
        self.nextFileButton.clicked.connect(self._next_file_button_clicked)
        self.importFileButton.clicked.connect(self._import_file_button_clicked)
        self.exportFileButton.clicked.connect(self._export_file_button_clicked)
        self.backFileButton.clicked.connect(self._back_file_button_clicked)
        self.markPeaksButton.clicked.connect(self._mark_peaks_button_clicked)
        self.confirmPeaksButton.clicked.connect(self._confirm_button_clicked)

        layout.addWidget(self.tableWidget)
        layout.addWidget(self.openButton)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(self.sub_layout_3)
        layout.addLayout(self.sub_layout_2)
        layout.addLayout(self.sub_layout)
        layout.addWidget(self.samePeaksCheckBox)
        layout.addWidget(self.peaksRangeTableWidget)
        layout.addWidget(self.markPeaksButton)
        layout.addWidget(self.confirmPeaksButton)

        self.setLayout(layout)

    def cell(self,var=""):
        item = QTableWidgetItem()
        item.setText(var)
        return item

    def createTable(self):
        self.tableWidget = QTableWidget()
  
        #Row count
        self.tableWidget.setRowCount(3) 
        rows=self.tableWidget.rowCount()
        #Column count
        self.tableWidget.setColumnCount(2)
        columns=self.tableWidget.columnCount()
        contents=[["Files path:", '/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/processed/centered/20221114/202201114_Au'],["Number of peaks per file to follow:", '4'], ["Peak(s) file:", '/asap3/fs-bmx/gpfs/regae/2022/data/11016614/scratch_cc/rodria/processed/calib/2022114_Au_free_peak_0_4.txt']]
        #contents=[["Files path:", '/asap3/fs-bmx/gpfs/regae/2022/data/11015669/processed/centered/20220722/20220722_Au_free'],["Number of peaks per file to follow:", '1'], ["Peak(s) file:", '/asap3/fs-bmx/gpfs/regae/2022/data/11015669/processed/calib/20220524_Au_free_peak_0.txt']]
        #contents=[["Files path:", '/asap3/fs-bmx/gpfs/regae/2022/data/11015323/processed/centered/20220524/20220524_Au30'],["Number of peaks per file to follow:", '1'], ["Peak(s) file:", '/asap3/fs-bmx/gpfs/regae/2022/data/11015323/processed/calib/20220524_Au30_peak_0.txt']]

        for i in range(rows):
            for j in range(columns):
                item = self.cell(contents[i][j])
                if j==0:
                    item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.tableWidget.setItem(i, j, item)
   
        #Table will fit the screen horizontally
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    
    def textchanged(self) -> None:
        # Manages clicks on the 'run' button.
        text=self.e1.text()

    def update_peak_map(self):
        index=np.where(self.peak_map==1)
        row=index[0][0]
        col=index[1][0]
        self.peak_map[row][col]=0
        if row==0:
             self.peak_map[row+1][col]=1
        else:
             if col==self.peak_map.shape[1]-1:
                 self.peak_map[0][0]=1
             else:
                 self.peak_map[row-1][col+1]=1

    def on_press(self,event):
        #print('coord:', round(event.xdata))
        total_of_frames=int(self.init_table[4])
        step=int(self.init_table[5])
        position_in_table=int(self.initial/(total_of_frames/step))
        index=np.where(self.peak_map==1)

        item = self.cell(f'{round(event.xdata)}')
        #print(index,position_in_table)

        self.peaksRangeTableWidget.setItem(index[0][0], index[1][0]+(position_in_table*self.number_of_peaks), item)
        x=[round(event.xdata)]
        y=[round(event.ydata)]
        self.ax.scatter(x,y,marker='X',color='c',s=100)
        self.canvas.draw()
        #jump to next position in peak map
        self.update_peak_map()
        
    def createPeaksRangeTable(self):
        self.peaksRangeTableWidget = QTableWidget()
  
        #Row count
        self.peaksRangeTableWidget.setRowCount(2) 
        rows=self.tableWidget.rowCount()
        #Column count
        self.peaksRangeTableWidget.setColumnCount(200)
        columns=self.peaksRangeTableWidget.columnCount()

 
        for i in range(rows):
            for j in range(columns):
                item = self.cell('')
                self.peaksRangeTableWidget.setItem(i, j, item)
   
        #Table will fit the screen horizontally
        self.peaksRangeTableWidget.horizontalHeader().setStretchLastSection(True)
        self.peaksRangeTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)



    def createOpenButton(self):
        self.openButton = QPushButton("Open")
        self.openButton.setCheckable(True)
        self.openButton.setChecked(False)
    
    def createImportFileButton(self):
        self.importFileButton = QPushButton("Import peaks")
        self.importFileButton.setCheckable(True)
        self.importFileButton.setChecked(False)

    def createExportFileButton(self):
        self.exportFileButton = QPushButton("Export peaks")
        self.exportFileButton.setCheckable(True)
        self.exportFileButton.setChecked(False)

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
        point=self.tableWidget.item(0,1).text()
        if point!='':
            args.append(point)
        
        self.show_plots(args)
        
        self.markPeaksButton.setCheckable(True)
        self.markPeaksButton.setChecked(False)

    def _open_button_clicked(self) -> None:
        # Manages clicks on the 'run' button.
        
        args=['-i']
        times=[]
        point=self.tableWidget.item(0,1).text()
        args.append(point)
        point=self.tableWidget.item(1,1).text()
        self.number_of_peaks=int(point)
        self.peak_map=np.zeros((2, self.number_of_peaks))
        self.peak_map[0][0]=1
        self.show_plots(args)
        
        point=self.tableWidget.item(2,1).text()
        self.peak_file=point

        self.openButton.setCheckable(False)
        self.openButton.setChecked(False)
        if self.count_clicks_open==0:
            self.label_2=QLabel(f'/{round((int(self.init_table[4])/int(self.init_table[5]))*(int(self.init_table[9])-int(self.init_table[7])+1))}')
            self.sub_layout.removeWidget(self.nextFileButton)
            self.sub_layout.addWidget(self.label_2)
            self.sub_layout.addWidget(self.nextFileButton)
        self.count_clicks_open+=1

    def _import_file_button_clicked(self) -> None:
        # Manages clicks on the 'import' button.

        self.importFileButton.setCheckable(True)
        self.importFileButton.setChecked(False)

        #open txt read mode and write in table
	    
        data=pd.read_csv(self.peak_file,sep=' ')

        for i in data['index']:
            item = self.cell(str(data['min'][i]))
            self.peaksRangeTableWidget.setItem(0, i, item)
            item = self.cell(str(data['max'][i]))
            self.peaksRangeTableWidget.setItem(1, i, item)
        print(f'Peak ranges successfully imported from {self.peak_file}!')

    def _export_file_button_clicked(self) -> None:
        # Manages clicks on the 'import' button.

        self.exportFileButton.setCheckable(True)
        self.exportFileButton.setChecked(False)

        #read table open txt write in txt
        columns=self.peaksRangeTableWidget.columnCount()
        rows=self.peaksRangeTableWidget.rowCount()
        f=open(self.peak_file,'w')
        f.write('index min max')

        for j in range(columns):
            point=self.peaksRangeTableWidget.item(0,j).text()
            if point!='':
                line=f'\n{j} {self.peaksRangeTableWidget.item(0,j).text()} {self.peaksRangeTableWidget.item(1,j).text()}'
                f.write(line)
        f.close()
        print(f'Peak ranges successfully saved in {self.peak_file}!')

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
            
            point=self.tableWidget.item(0,1).text()
            args.append(point)

            self.initial+=1

            self.show_plots(args)

    def textchanged(self) -> None:
        # Manages clicks on the 'run' button.
        text=self.e1.text()
        #print(text)
        if int(text)>len(self.input_order)-1 or (int(text))<0:
            self.nextFileButton.setCheckable(True) 
            self.nextFileButton.setChecked(False)
        else:
            self.nextFileButton.setCheckable(True) 
            self.nextFileButton.setChecked(False)
            args=['-i']
	    
            rows=self.tableWidget.rowCount()
            
            point=self.tableWidget.item(0,1).text()
            args.append(point)

            self.initial=int(text)

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
            
            point=self.tableWidget.item(0,1).text()
            args.append(point)
        
            self.initial-=1
        
            self.show_plots(args)


    def _confirm_button_clicked(self) -> None:
        # Manages clicks on the 'run' button.
        self.confirmPeaksButton.setCheckable(True)
        self.confirmPeaksButton.setChecked(False)
        args=['-n',f'{self.number_of_peaks}','-i',self.input_order,'-a',self.samePeaksCheckBox.isChecked()]
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
        #print(args)
        self.init_table.append(int(self.mode))
        #print(len(self.init_table))
        status=autoproc.run(args,self.init_table)
        if status==1:
             self.w=ResultsWindow()
             self.w.config_delay=self.config_delay
             self.w.mode=self.mode
        self.w.show()


   

    def show_plots(self,raw_args=None):

        # create an axis

        parser = argparse.ArgumentParser(
        description="Plot peak positions according to angle.")
        parser.add_argument("-i", "--input", type=str, action="store",
            help="hdf5 input image")
                
        args = parser.parse_args(raw_args)

        self.canvas.figure.clear()
        
        ax=self.figure.add_subplot(111)
        ax.clear()
        self.ax=ax
        self.canvas.draw()
        files=list(glob.glob(f"{args.input}*.h5"))
        files.sort()
        self.input_order=files.copy()
        file_path=files[int(self.initial)]
        #print(file_path)

        hf = h5py.File(file_path, 'r')
        x= np.array(hf['radial_x'])
        y= np.array(hf['radial'])
        hf.close()

        ax.plot(x,y,linewidth=2,c='b',label=file_path[-25:],marker='.')
        ax.set_xlim(0,200)
        if self.setYLim.isChecked()==True:
            ax.set_ylim(-100,int(self.e2.text()))
        ax.set_xlabel('Radius (px)')
        ax.set_ylabel('Intensity [a.u.]')
        ax.set_facecolor('grey')
        self.legend = ax.legend()

        if self.draw_peaks==1:
            total_of_frames=int(self.init_table[4])
            step=int(self.init_table[5])
            position_in_table=int(self.initial/(total_of_frames/step))

            if self.samePeaksCheckBox.isChecked()==True:
                xi=0
                xf=self.number_of_peaks
            else:
                xi=position_in_table*self.number_of_peaks
                xf=(position_in_table+1)*self.number_of_peaks

            peak_range=[]
            list_of_peaks=[]
            for j in range(xi,xf):
                for i in range(2):
                    peak_range.append(self.peaksRangeTableWidget.item(i,j).text())
                list_of_peaks.append(peak_range)
                peak_range=[]
            #print(list_of_peaks)
            y_range=np.arange(-100,np.ndarray.max(y),1)
            color=['r','g','k','m','c','y']
            n_peak=0
            for i in list_of_peaks:             
                for j in i:
                     x_range=np.ones(y_range.shape)
                     x_range=int(j)*x_range
                     ax.scatter(x_range,y_range,marker='|',color=color[n_peak])
                n_peak+=1

        
        ax.grid()
        
        
        self.canvas.draw()
    
    





