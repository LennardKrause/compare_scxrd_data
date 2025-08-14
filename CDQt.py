#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#   CompareDataQt.py reads SCXRD data formats .raw, .hkl, .fcf and
#   calculates the diff / mean vs. intensity values of equivalent observation.
#   It currently reads SAINT .raw, XD2006 .fco and general SHELX .hkl files.
#   Copyright (C) 2018, L.Krause <lkrause@chem.au.dk>, Aarhus University, DK.
#
#   This program is free software: you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the Free
#   Software Foundation, either version 3 of the license, or (at your option)
#   any later version.
#
#   This program is distributed in the hope that it will be useful, but WITHOUT
#   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
#   more details. <http://www.gnu.org/licenses/>
#
_REVISION = 'v2025-08-14'

# todo:
# - add support for .fcf files


import os
import sys
import traceback
import logging
import numpy as np
import pandas as pd
from PyQt6 import QtWidgets, QtCore
import matplotlib as mpl
import matplotlib.pyplot as plt

class WorkerSignals(QtCore.QObject):
    '''
    SOURCE: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    '''
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(tuple)

class Worker(QtCore.QRunnable):
    '''
    SOURCE: https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        logging.info(self.__class__.__name__)
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        logging.info(self.__class__.__name__)
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            r = self.fn(*self.args)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit((r, self.kwargs))
        finally:
            self.signals.finished.emit(self.kwargs)

class QLineEditDropHandler(QtCore.QObject):
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        logging.info(self.__class__.__name__)
        self.valid_extensions = parent.valid_extensions

    def eventFilter(self, obj, event):
        #logging.info(self.__class__.__name__)
        if event.type() == QtCore.QEvent.Type.DragEnter:
            md = event.mimeData()
            if md.hasUrls():
                for url in md.urls():
                    filePath = url.toLocalFile()
                    root, ext = os.path.splitext(filePath)
                    if ext in self.valid_extensions:
                        event.accept()
        
        if event.type() == QtCore.QEvent.Type.Drop:
            md = event.mimeData()
            if md.hasUrls():
                for url in md.urls():
                    filePath = url.toLocalFile()
                    root, ext = os.path.splitext(filePath)
                    if ext in self.valid_extensions:
                        obj.clear()
                        obj.setText(filePath)
                        obj.returnPressed.emit()
                        return True
            
        return QtCore.QObject.eventFilter(self, obj, event)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        logging.info(self.__class__.__name__)
        super(MainWindow, self).__init__(*args, **kwargs)
        self.valid_extensions = ['.raw', '.fco', '.hkl', '.sortav']
        self.homedir = os.path.dirname(__file__)

        self.build_ui()
        
        self.le_data_1.installEventFilter(QLineEditDropHandler(self))
        self.le_data_2.installEventFilter(QLineEditDropHandler(self))
        self.le_data_1.returnPressed.connect(lambda: self.prepare_read_data_1(self.le_data_1.text()))
        self.le_data_2.returnPressed.connect(lambda: self.prepare_read_data_2(self.le_data_2.text()))
        self.tb_plot.clicked.connect(self.plot_data)
        self.tb_data_1.clicked.connect(lambda: self.open_file_browser(self.le_data_1))
        self.tb_data_2.clicked.connect(lambda: self.open_file_browser(self.le_data_2))
        self.cb_sym.currentIndexChanged.connect(self.set_symmetry_operations)
        self.btn_clear.clicked.connect(self.clear_all)
        
        self.threadpool = QtCore.QThreadPool()
        self.ready_data = False
        self.ready_data_1 = False
        self.ready_data_2 = False
        self.data_1 = None
        self.data_2 = None
        self.data = None
        self.last_dir_1 = None
        self.last_dir_2 = None

        # helper dicts
        self.data_links = {self.le_data_1.objectName(): self.last_dir_1,
                           self.le_data_2.objectName(): self.last_dir_2}
        
        self.group_scale = QtWidgets.QButtonGroup()
        self.group_scale.addButton(self.rb_scale_1)
        self.group_scale.addButton(self.rb_scale_2)
        
        self.init_custom_styles()
        self.init_symmetry()

    def build_ui(self):
        # apply palette to app
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint | QtCore.Qt.WindowType.WindowCloseButtonHint)
        app = QtWidgets.QApplication.instance()
        app.setStyle('Fusion')

        logging.info(self.__class__.__name__)
        self.setWindowTitle('SCXRD Data Compare {}'.format(_REVISION))
        self.layout_v_main = QtWidgets.QVBoxLayout()
        self.layout_h_top = QtWidgets.QHBoxLayout()
        self.layout_h_bottom = QtWidgets.QHBoxLayout()

        self.gbox_data_1 = QtWidgets.QGroupBox('Data 1', alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_data_1 = QtWidgets.QGridLayout()
        self.gbox_data_1.setLayout(self.layout_data_1)
        self.le_data_1 = QtWidgets.QLineEdit()
        self.le_data_1.setObjectName('le_data_1')
        self.layout_data_1.addWidget(self.le_data_1, 0, 0)
        self.tb_data_1 = QtWidgets.QToolButton()
        self.tb_data_1.setText('...')
        self.tb_data_1.setObjectName('tb_data_1')
        self.tb_data_1.setToolTip('Open file browser')
        self.layout_data_1.addWidget(self.tb_data_1, 0, 1)
        self.la_data_1 = QtWidgets.QLabel('Reflections: 0')
        self.la_data_1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_data_1.addWidget(self.la_data_1, 1, 0, 1, 2)
        self.le_label_1 = QtWidgets.QLineEdit('1')
        self.le_label_1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_data_1.addWidget(self.le_label_1, 2, 0)
        self.la_label_1 = QtWidgets.QLabel('Label')
        self.layout_data_1.addWidget(self.la_label_1, 2, 1)

        self.gbox_data_2 = QtWidgets.QGroupBox('Data 2', alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_data_2 = QtWidgets.QGridLayout()
        self.gbox_data_2.setLayout(self.layout_data_2)
        self.le_data_2 = QtWidgets.QLineEdit()
        self.le_data_2.setObjectName('le_data_2')
        self.layout_data_2.addWidget(self.le_data_2, 0, 0)
        self.tb_data_2 = QtWidgets.QToolButton()
        self.tb_data_2.setText('...')
        self.tb_data_2.setObjectName('tb_data_2')
        self.tb_data_2.setToolTip('Open file browser')
        self.layout_data_2.addWidget(self.tb_data_2, 0, 1)
        self.la_data_2 = QtWidgets.QLabel('Reflections: 0')
        self.la_data_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_data_2.addWidget(self.la_data_2, 1, 0, 1, 2)
        self.le_label_2 = QtWidgets.QLineEdit('2')
        self.le_label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_data_2.addWidget(self.le_label_2, 2, 0)
        self.la_label_2 = QtWidgets.QLabel('Label')
        self.layout_data_2.addWidget(self.la_label_2, 2, 1)
        
        self.gbox_symmetry = QtWidgets.QGroupBox('Symmetry', alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_symmetry = QtWidgets.QGridLayout()
        self.gbox_symmetry.setLayout(self.layout_symmetry)
        self.cb_sym = QtWidgets.QComboBox()
        self.layout_symmetry.addWidget(self.cb_sym, 0, 0, 1, 2)
        self.la_data_sym = QtWidgets.QLabel('-')
        self.la_data_sym.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_symmetry.addWidget(self.la_data_sym, 1, 0, 1, 2)
        self.le_prefix = QtWidgets.QLineEdit('')
        self.le_prefix.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.la_prefix = QtWidgets.QLabel('Prefix')
        self.layout_symmetry.addWidget(self.le_prefix, 2, 0)
        self.layout_symmetry.addWidget(self.la_prefix, 2, 1)

        self.gbox_plot = QtWidgets.QGroupBox('Plot', alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_plot = QtWidgets.QGridLayout()
        self.gbox_plot.setLayout(self.layout_plot)
        self.tb_plot = QtWidgets.QToolButton(text='Plot')
        self.tb_plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.db_sigcut = QtWidgets.QDoubleSpinBox()
        self.db_sigcut.setRange(0.0, 100.0)
        self.db_sigcut.setSingleStep(0.1)
        self.db_sigcut.setValue(0.5)
        self.db_sigcut.setToolTip('Sigma cutoff')
        self.la_sigcut = QtWidgets.QLabel('Sigcut')
        self.la_scale = QtWidgets.QLabel('Auto')
        self.rb_scale_1 = QtWidgets.QRadioButton()
        self.ds_scale = QtWidgets.QDoubleSpinBox()
        self.ds_scale.setRange(0.0, 100.0)
        self.ds_scale.setDecimals(3)
        self.ds_scale.setSingleStep(0.001)
        self.ds_scale.setValue(1.0)
        self.rb_scale_2 = QtWidgets.QRadioButton()
        self.rb_scale_1.setChecked(True)
        self.rb_scale_1.setAutoExclusive(True)
        self.rb_scale_2.setAutoExclusive(True)
        self.cb_save = QtWidgets.QCheckBox('Save')
        self.cb_title = QtWidgets.QCheckBox('Title')
        self.btn_clear = QtWidgets.QToolButton(text='Clear')
        self.btn_clear.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.layout_plot.addWidget(self.tb_plot, 0, 0, 5, 1)
        self.layout_plot.addWidget(self.db_sigcut, 0, 1)
        self.layout_plot.addWidget(self.la_sigcut, 0, 2)
        self.layout_plot.addWidget(self.la_scale, 1, 1)
        self.layout_plot.addWidget(self.rb_scale_1, 1, 2)
        self.layout_plot.addWidget(self.ds_scale, 2, 1)
        self.layout_plot.addWidget(self.rb_scale_2, 2, 2)
        self.layout_plot.addWidget(self.cb_save, 3, 1)
        self.layout_plot.addWidget(self.cb_title, 4, 1)
        self.layout_plot.addWidget(self.btn_clear, 5, 0, 1, 3)

        centralwidget = QtWidgets.QWidget()
        centralwidget.setLayout(self.layout_v_main)
        self.setCentralWidget(centralwidget)

        self.layout_v_main.addLayout(self.layout_h_top)
        self.layout_v_main.addLayout(self.layout_h_bottom)
        self.layout_h_top.addWidget(self.gbox_data_1)
        self.layout_h_top.addWidget(self.gbox_symmetry)
        self.layout_h_top.addWidget(self.gbox_data_2)
        
        self.layout_plot_spacer = QtWidgets.QHBoxLayout()
        self.layout_plot_spacer.addStretch()
        self.layout_plot_spacer.addWidget(self.gbox_plot)
        self.layout_plot_spacer.addStretch()
        self.layout_h_bottom.addLayout(self.layout_plot_spacer)

        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
    
    def open_file_browser(self, aWidget):
        logging.info(self.__class__.__name__)
        last_dir = self.data_links.get(aWidget.objectName(), self.homedir)
        aPath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', last_dir, 'SCXRD Data Formats (*.raw, *.fco, *.hkl)', '-')[0]
        if not os.path.exists(aPath):
            return
        aWidget.setText(aPath)
        aWidget.returnPressed.emit()
    
    def init_symmetry(self):
        logging.info(self.__class__.__name__)
        self.Symmetry = {  '1':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]]]),
                         
                          '-1':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]]]),
                         
                         '2/m':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]]]),
                     
                         '222':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]]]),
                         
                         'mmm':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0,  1]]]),
                     
                         '4/m':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0, -1]]]),
                     
                       '4/mmm':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0,  1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  0, -1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[  0,  1,  0],[  1,  0,  0],[  0,  0,  1]]]),
                                
                        'm-3m':np.array([[[  1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  0,  0,  1],[  1,  0,  0],[  0,  1,  0]],
                                         [[  0,  0,  1],[ -1,  0,  0],[  0, -1,  0]],
                                         [[  0,  0, -1],[ -1,  0,  0],[  0,  1,  0]],
                                         [[  0,  0, -1],[  1,  0,  0],[  0, -1,  0]],
                                         [[  0,  1,  0],[  0,  0,  1],[  1,  0,  0]],
                                         [[  0, -1,  0],[  0,  0,  1],[ -1,  0,  0]],
                                         [[  0,  1,  0],[  0,  0, -1],[ -1,  0,  0]],
                                         [[  0, -1,  0],[  0,  0, -1],[  1,  0,  0]],
                                         [[  0,  1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  0, -1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0,  1]],
                                         [[  1,  0,  0],[  0,  0,  1],[  0, -1,  0]],
                                         [[ -1,  0,  0],[  0,  0,  1],[  0,  1,  0]],
                                         [[ -1,  0,  0],[  0,  0, -1],[  0, -1,  0]],
                                         [[  1,  0,  0],[  0,  0, -1],[  0,  1,  0]],
                                         [[  0,  0,  1],[  0,  1,  0],[ -1,  0,  0]],
                                         [[  0,  0,  1],[  0, -1,  0],[  1,  0,  0]],
                                         [[  0,  0, -1],[  0,  1,  0],[  1,  0,  0]],
                                         [[  0,  0, -1],[  0, -1,  0],[ -1,  0,  0]],
                                         [[ -1,  0,  0],[  0, -1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0,  1,  0],[  0,  0, -1]],
                                         [[  1,  0,  0],[  0, -1,  0],[  0,  0,  1]],
                                         [[ -1,  0,  0],[  0,  1,  0],[  0,  0,  1]],
                                         [[  0,  0, -1],[ -1,  0,  0],[  0, -1,  0]],
                                         [[  0,  0, -1],[  1,  0,  0],[  0,  1,  0]],
                                         [[  0,  0,  1],[  1,  0,  0],[  0, -1,  0]],
                                         [[  0,  0,  1],[ -1,  0,  0],[  0,  1,  0]],
                                         [[  0, -1,  0],[  0,  0, -1],[ -1,  0,  0]],
                                         [[  0,  1,  0],[  0,  0, -1],[  1,  0,  0]],
                                         [[  0, -1,  0],[  0,  0,  1],[  1,  0,  0]],
                                         [[  0,  1,  0],[  0,  0,  1],[ -1,  0,  0]],
                                         [[  0, -1,  0],[ -1,  0,  0],[  0,  0,  1]],
                                         [[  0,  1,  0],[  1,  0,  0],[  0,  0,  1]],
                                         [[  0, -1,  0],[  1,  0,  0],[  0,  0, -1]],
                                         [[  0,  1,  0],[ -1,  0,  0],[  0,  0, -1]],
                                         [[ -1,  0,  0],[  0,  0, -1],[  0,  1,  0]],
                                         [[  1,  0,  0],[  0,  0, -1],[  0, -1,  0]],
                                         [[  1,  0,  0],[  0,  0,  1],[  0,  1,  0]],
                                         [[ -1,  0,  0],[  0,  0,  1],[  0, -1,  0]],
                                         [[  0,  0, -1],[  0, -1,  0],[  1,  0,  0]],
                                         [[  0,  0, -1],[  0,  1,  0],[ -1,  0,  0]],
                                         [[  0,  0,  1],[  0, -1,  0],[ -1,  0,  0]],
                                         [[  0,  0,  1],[  0,  1,  0],[  1,  0,  0]]])}
        
        [self.cb_sym.addItem(i) for i in sorted(self.Symmetry.keys())]
        self.cb_sym.setCurrentText('-1')
    
    def set_symmetry_operations(self):
        logging.info(self.__class__.__name__)
        self.SymOp = self.Symmetry[self.cb_sym.currentText()]
        self.la_data_sym.setText('Unique: 0')
        if self.ready_data_1 and self.ready_data_2:
            self.thread_run(self.merge_data, flag='ready_data_all')
        
    def init_custom_styles(self):
        logging.info(self.__class__.__name__)
        self.tb_style = ('QToolButton          {background-color: rgb(240, 250, 240); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75); border-radius: 5px}'
                         'QToolButton:hover    {background-color: rgb(250, 255, 250); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}'
                         'QToolButton:pressed  {background-color: rgb(255, 255, 255); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}'
                         'QToolButton:checked  {background-color: rgb(220, 220, 220); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}'
                         'QToolButton:disabled {background-color: rgb(220, 200, 200); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}')
        self.tb_plot.setStyleSheet(self.tb_style)

    def prepare_read_data_1(self, aPath):
        logging.info(self.__class__.__name__)
        if not os.path.exists(aPath):
            print('invalid path!')
            return
        self.ready_data_1 = False
        self.data_links[self.le_data_1.objectName()] = aPath
        self.le_data_1.setEnabled(False)
        self.tb_data_1.setEnabled(False)
        self.thread_run(self.read_data, aPath, parent_widget=self.le_data_1)

    def prepare_read_data_2(self, aPath):
        logging.info(self.__class__.__name__)
        if not os.path.exists(aPath):
            print('invalid path!')
            return
        self.ready_data_2 = False
        self.data_links[self.le_data_2.objectName()] = aPath
        self.le_data_2.setEnabled(False)
        self.tb_data_2.setEnabled(False)
        self.thread_run(self.read_data, aPath, parent_widget=self.le_data_2)

    def read_data(self, fname, use_columns=None, used_only=True):
        logging.info(self.__class__.__name__)
        '''
        
        '''
        name, ext = os.path.splitext(fname)
        ints = ['h','k','l']
        floats = ['Fo','Fs']
        self.use_stl = False
        
        if ext == '.raw':
            if not use_columns:
                use_columns = (0,1,2,3,4)
            raw_data = np.genfromtxt(fname, dtype=float, usecols=use_columns, delimiter=[4,4,4,8,8,4,8,8,8,8,8,8,3,7,7,8,7,7,8,6,5,7,7,7,2,5,9,7,7,4,6,11,3,6,8,8,8,8,4])
        elif ext == '.fco':
            # delimiter=[6,5,5,11,11,11,11,4])
            # skip_header=26
            if not use_columns:
                use_columns = (0,1,2,4,5,6,7)
            raw_data = np.genfromtxt(fname, dtype=float, skip_header=26, usecols=use_columns)
            if used_only:
                raw_data = raw_data[raw_data[::,6] == 0]
            floats = ['Fo','Fs','stl']
            self.use_stl = True
            raw_data = raw_data[:,[0,1,2,3,4,5]]
        elif ext == '.sortav':
            if not use_columns:
                use_columns = (0,1,2,3,6)
            raw_data = np.genfromtxt(fname, dtype=float, usecols=use_columns, comments='c')
        elif ext == '.hkl':
            with open(fname) as ofile:
                temp = ofile.readline()
            if len(temp.split()) == 4 and 'NDAT' in temp:
                # XD2006
                # HEADER:XDNAME F^2 NDAT 7
                # delimiter=[4,4,4,2,8,8,8])
                if not use_columns:
                    use_columns = (0,1,2,4,5)
                raw_data = np.genfromtxt(fname, dtype=float, skip_header=1, usecols=use_columns)
            else:
                # SHELX
                # delimiter=[4,4,4,8,8,4]
                # skip_footer=17
                if not use_columns:
                    use_columns = (0,1,2,3,4)
                raw_data = np.genfromtxt(fname, dtype=float, skip_footer=17, usecols=use_columns, delimiter=[4,4,4,8,8,4])
        else:
            data = None
        data = pd.DataFrame(raw_data, columns=ints+floats)
        data = data.astype(dict(zip(ints,[int]*len(ints))))
        return data
    
    def merge_data(self):
        logging.info(self.__class__.__name__)

        # find common base for hkl
        def reducehkltofam(hkl, SymOps):
            return tuple(np.unique((hkl.to_numpy()).dot(SymOps), axis=0)[-1])
        # Reduce hkls according to symmetry
        self.data_1['base'] = self.data_1[['h','k','l']].apply(reducehkltofam, args=(self.SymOp,), axis=1)
        self.data_2['base'] = self.data_2[['h','k','l']].apply(reducehkltofam, args=(self.SymOp,), axis=1)
        
        self.data = pd.merge(self.data_1, self.data_2, how='outer', on='base', suffixes=('_1','_2'))
        self.data.dropna(axis=0, how='any', inplace=True)

        nunique = self.data.groupby(['base']).nunique()
        unique = len(nunique)
        total = nunique[['Fo_1', 'Fo_2']].sum(axis=1).sum()
        self.la_data_sym.setText(f'Unique: {unique} ({total})')

        if 'stl_1' not in self.data.columns and 'stl_2' not in self.data.columns:
            self.use_stl = False
        else:
            self.data['stl'] = self.data[['stl_1', 'stl_2']].mean(axis=1)
            self.data.drop(columns=['stl_1','stl_2'], inplace=True)

        self.ready_data = True
    
    def plot_data(self):
        logging.info(self.__class__.__name__)
        _SIGCUT  = round(self.db_sigcut.value(), 1)
        _SCALE   = self.rb_scale_1.isChecked()
        _SAVE    = self.cb_save.isChecked()
        _FILE_1  = self.le_data_1.text()
        _FILE_2  = self.le_data_2.text()
        _LABEL_1 = self.le_label_1.text()
        _LABEL_2 = self.le_label_2.text()
        _PREFIX  = self.le_prefix.text()
        _TITLE   = self.cb_title.isChecked()
        
        mpl.rcParams['figure.figsize']   = [13.66, 7.68]
        #mpl.rcParams['figure.dpi']      = 600
        mpl.rcParams['savefig.dpi']      = 300
        mpl.rcParams['font.size']        = 12
        mpl.rcParams['legend.fontsize']  = 12
        mpl.rcParams['figure.titlesize'] = 12
        mpl.rcParams['figure.titlesize'] = 12
        mpl.rcParams['axes.titlesize']   = 12
        mpl.rcParams['axes.labelsize']   = 12
        mpl.rcParams['lines.linewidth']  = 1
        mpl.rcParams['lines.markersize'] = 8
        mpl.rcParams['xtick.labelsize']  = 8
        mpl.rcParams['ytick.labelsize']  = 8
        scatter_marker_size = 12
        
        fig = plt.figure()
        if self.use_stl:
            grid = plt.GridSpec(12, 13, wspace=0.0, hspace=0.0)
        else:
            grid = plt.GridSpec(7, 13, wspace=0.0, hspace=0.0)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.08, wspace=0.0, hspace=0.0)
        
        # apply sigma cutoff
        cond1 = self.data['Fo_1']/self.data['Fs_1'] >= _SIGCUT
        cond2 = self.data['Fo_2']/self.data['Fs_2'] >= _SIGCUT
        cut = self.data.where(cond1 & cond2)
        # group by base and calculate mean
        grp = cut.groupby(['base'])[['Fo_1', 'Fo_2']].mean()
        f1cut = grp['Fo_1']
        f2cut = grp['Fo_2']
        
        check_1 = f1cut.count()
        check_2 = f2cut.count()
        if check_1 != check_2:
            self.statusBar.showMessage('Data mismatch: {} != {}'.format(check_1, check_2))
            return
        if check_1 == 0:
            self.statusBar.showMessage('No data (1) after sigma cutoff!')
            return
        if check_2 == 0:
            self.statusBar.showMessage('No data (2) after sigma cutoff!')
            return
        
        self.statusBar.showMessage('')

        scale = self.ds_scale.value()
        if _SCALE:
            scale = np.nansum(f1cut*f2cut)/np.nansum(np.square(f1cut))
        
        if _TITLE:
            fig.suptitle('Scalefactor: {:6.3f}, cutoff: {} [data: {}], symmetry: {}\n1: {}\n2: {}'.format(scale, _SIGCUT, cut['Fo_1'].count(), self.cb_sym.currentText(), _FILE_1, _FILE_2))
            fig.subplots_adjust(left=0.10, right=0.99, top=0.85, bottom=0.12)
        
        f1cut *= scale
        
        if self.use_stl:
            p00 = fig.add_subplot(grid[ :2 ,  :6])
            p01 = fig.add_subplot(grid[ :2 , 7: ])
            p1x = fig.add_subplot(grid[3:6 , 1: ])
            h1y = fig.add_subplot(grid[3:6 , 0  ], sharey=p1x)
            h1x = fig.add_subplot(grid[6   , 1: ], sharex=p1x)
            p2x = fig.add_subplot(grid[8:11, 1: ])
            h2y = fig.add_subplot(grid[8:11, 0  ], sharey=p2x)
            h2x = fig.add_subplot(grid[11  , 1: ], sharex=p2x)
            mpl.rcParams['figure.figsize'] = [13.66, 10.24]
        else:
            p00 = fig.add_subplot(grid[ :2,  :6])
            p01 = fig.add_subplot(grid[ :2, 7: ])
            p1x = fig.add_subplot(grid[3:6, 1: ])
            h1y = fig.add_subplot(grid[3:6, 0  ], sharey=p1x)
            h1x = fig.add_subplot(grid[6  , 1: ], sharex=p1x)
        
        p00.scatter(f1cut, f2cut, s=4, color='#37A0CB')
        p00.plot([0, np.nanmax(f1cut)],[0, np.nanmax(f1cut)], 'k-', lw=1.0)
        p00.set_xlabel(r'$F^2_{{{}}}$'.format(_LABEL_1))
        p00.set_ylabel(r'$F^2_{{{}}}$'.format(_LABEL_2))
        p00.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        p00.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        x = np.log10(f1cut)
        y = np.log10(f2cut)
        p01.scatter(x, y, s=4, color='#37A0CB')
        p01.plot([np.nanmin(x), np.nanmax(x)],[np.nanmin(x), np.nanmax(x)], 'k-', lw=1.0)
        p01.set_xlabel(r'$\log\left(F^2_{{{}}}\right)$'.format(_LABEL_1))
        p01.set_ylabel(r'$\log\left(F^2_{{{}}}\right)$'.format(_LABEL_2))
        
        facut = (f1cut + f2cut) / 2.
        x = np.log10(facut)
        y = (f1cut - f2cut)/(facut)
        
        p1x_sc = p1x.scatter(x, y, s=20, alpha=0.5, picker=True, color='#37A0CB')
        p1x.plot([np.min(x), np.max(x)], [0,0], 'k-', lw=1.0)
        p1x.spines['left'].set_visible(False)
        p1x.spines['bottom'].set_visible(False)
        p1x.xaxis.set_visible(False)
        p1x.yaxis.set_visible(False)
        
        if self.use_stl:
            stl = cut['stl'].dropna()
            p2x.scatter(stl, y, s=20, alpha=0.5, picker=True, color='#37A0CB')
            p2x.plot([np.min(stl), np.max(stl)], [0,0], 'k-', lw=1.0)
            p2x.set_ylabel(r'$(F^2_{1}\ -\ F^2_{2})\ /\ \left<F^2_{1,2}\right>$')
            p2x.spines['left'].set_visible(False)
            p2x.spines['bottom'].set_visible(False)
            p2x.yaxis.set_visible(False)
            p2x.xaxis.set_visible(False)
            
            h2y.hist(y[(~np.isnan(y)) & (y<2.) & (y>-2.)], 400, color='#003e5c', histtype='stepfilled', orientation='horizontal')
            h2y.xaxis.set_visible(False)
            h2y.invert_xaxis()
            h2y.spines['top'].set_visible(False)
            h2y.spines['bottom'].set_visible(False)
            h2y.set_ylabel(r'$\left(F^2_{{{0:}}}\ -\ F^2_{{{1:}}}\right)\ /\ \left<F^2_{{{{{0:}}},{{{1:}}}}}\right>$'.format(_LABEL_1, _LABEL_2))
        
            h2x.hist(stl[~np.isnan(stl)], 400, color='#003e5c', histtype='stepfilled', orientation='vertical')
            h2x.yaxis.set_visible(False)
            h2x.spines['left'].set_visible(False)
            h2x.spines['right'].set_visible(False)
            h2x.invert_yaxis()
            h2x.set_xlabel(r'$sin(\theta)/\lambda$')
            
        h1y.hist(y[(~np.isnan(y)) & (y<2.) & (y>-2.)], 400, color='#003e5c', histtype='stepfilled', orientation='horizontal')
        #h1y.set_ylim([-2.0, 2.0])
        h1y.xaxis.set_visible(False)
        h1y.invert_xaxis()
        h1y.spines['top'].set_visible(False)
        h1y.spines['bottom'].set_visible(False)
        #h1y.spines['right'].set_visible(False)
        h1y.set_ylabel(r'$\left(F^2_{{{0:}}}\ -\ F^2_{{{1:}}}\right)\ /\ \left<F^2_{{{{{0:}}},{{{1:}}}}}\right>$'.format(_LABEL_1, _LABEL_2))
        
        h1x.hist(x[~np.isnan(x)], 400, color='#003e5c', histtype='stepfilled', orientation='vertical')
        h1x.yaxis.set_visible(False)
        #h1x.spines['top'].set_visible(False)
        h1x.spines['left'].set_visible(False)
        h1x.spines['right'].set_visible(False)
        h1x.invert_yaxis()
        h1x.set_xlabel(r'$\log\left(\left<F^2_{{{{{0:}}},{{{1:}}}}}\right>\right)$'.format(_LABEL_1, _LABEL_2))
        
        if _SAVE:
            pname = r'{}_{}_vs_{}_c{}_s{}'.format(_PREFIX, _LABEL_1.replace('\\', ''), _LABEL_2.replace('\\', ''), _SIGCUT, scale)
            fig.savefig(os.path.join(self.homedir, f'{pname}.png'), transparent=True)
            #plt.savefig(os.path.join(self.homedir, f'{pname}.png'), transparent=True)
            #plt.savefig(pname + '.png', dpi=600, transparent=True)
        
        fig.show()

    def clear_all(self):
        self.le_data_1.clear()
        self.le_data_2.clear()
        self.la_data_1.setText('Reflections: 0')
        self.la_data_2.setText('Reflections: 0')
        self.la_data_sym.setText('Unique: 0')
        self.ready_data = False
        self.ready_data_1 = False
        self.ready_data_2 = False
        self.data = None
        self.data_1 = None
        self.data_2 = None
        self.tb_plot.setEnabled(False)
        self.le_data_1.setEnabled(True)
        self.le_data_2.setEnabled(True)
        self.tb_data_1.setEnabled(True)
        self.tb_data_2.setEnabled(True)
        
    def on_thread_result(self, r):
        logging.info(self.__class__.__name__)
        data, kwargs = r
        if data is not None:
            if 'parent_widget' in kwargs and kwargs['parent_widget'] == self.le_data_1:
                self.data_1 = data
                self.ready_data_1 = True
                self.la_data_1.setText('Reflections: {}'.format(str(len(data))))
            elif 'parent_widget' in kwargs and kwargs['parent_widget'] == self.le_data_2:
                self.data_2 = data
                self.ready_data_2 = True
                self.la_data_2.setText('Reflections: {}'.format(str(len(data))))
    
    def on_thread_finished(self, kwargs):
        logging.info(self.__class__.__name__)
        if self.threadpool.activeThreadCount() == 0:
            self.statusBar.showMessage('ready.')
            self.setEnabled(True)
        
        if self.ready_data_1 and self.ready_data_2 and not self.ready_data:
            self.thread_run(self.merge_data, flag='ready_data_all')
        
        if 'flag' in kwargs and kwargs['flag'] == 'ready_data_all' and self.threadpool.activeThreadCount() == 0:
            self.tb_plot.setEnabled(True)
            self.cb_sym.setEnabled(True)
    
    def thread_run(self, fn, *args, **kwargs):
        logging.info(self.__class__.__name__)
        self.tb_plot.setEnabled(False)
        w = Worker(fn, *args, **kwargs)
        w.signals.result.connect(self.on_thread_result)
        w.signals.finished.connect(self.on_thread_finished)
        self.threadpool.start(w)
        ## ALWAYS DIASBLE THE SYM SWITCH ##
        self.cb_sym.setEnabled(False)
        self.setEnabled(False)
        ###################################
        self.statusBar.showMessage('I\'m thinking ...')
        if 'parent_widget' in kwargs:
            kwargs['parent_widget'].setEnabled(False)

def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.setWindowTitle('Compare SCXRD Data, {}'.format(_REVISION))
    ui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    # create logger
    logging.basicConfig(level=logging.INFO, format='%(message)20s > %(funcName)s')
    main()
