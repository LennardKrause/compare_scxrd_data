#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#   PILATUS_CompareDataQt.py reads SCXRD data formats .raw, .hkl, .fcf and
#   calculates the diff / mean vs. intensity values of equivalent observation.
#   It currently reads SAINT .raw, XD2006 .fcf and general SHELX .hkl files.
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
_REVISION = 'v2019-01-18'

from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import matplotlib as mpl
try:
    mpl.use('Qt5Agg')
except ImportError:
    os.environ['TCL_LIBRARY'] = '{}/tcl/tcl8.5'.format(sys.prefix)
    pass
import matplotlib.pyplot as plt

import numpy as np
from collections import OrderedDict
import time
import os, sys, traceback, logging

class WorkerSignals(QObject):
    '''
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
    finished = pyqtSignal(object)
    error = pyqtSignal(tuple)
    result = pyqtSignal(tuple)

class Worker(QRunnable):
    '''
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

    @pyqtSlot()
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

class QLineEditDropHandler(QObject):
    def __init__(self, parent = None):
        logging.info(self.__class__.__name__)
        QObject.__init__(self, parent)
    
    def valid_ext(self, ext):
        #logging.info(self.__class__.__name__)
        if ext in ['.raw', '.hkl', '.fco', '.fcf']:
            return True
        else:
            return False
        
    def eventFilter(self, obj, event):
        #logging.info(self.__class__.__name__)
        if event.type() == QEvent.DragEnter:
            md = event.mimeData()
            if md.hasUrls():
                for url in md.urls():
                    filePath = url.toLocalFile()
                    root, ext = os.path.splitext(filePath)
                    if self.valid_ext(ext):
                        event.accept()
        
        if event.type() == QEvent.Drop:
            md = event.mimeData()
            if md.hasUrls():
                for url in md.urls():
                    filePath = url.toLocalFile()
                    root, ext = os.path.splitext(filePath)
                    if self.valid_ext(ext):
                        obj.clear()
                        obj.setText(filePath)
                        obj.returnPressed.emit()
                        return True
            
        return QObject.eventFilter(self, obj, event)

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        logging.info(self.__class__.__name__)
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('PILATUS_CompareDataQt.ui', self)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.WindowCloseButtonHint)
        
        self.le_data_1.installEventFilter(QLineEditDropHandler(self))
        self.le_data_2.installEventFilter(QLineEditDropHandler(self))
        self.le_data_1.returnPressed.connect(lambda: self.prepare_read_data(self.le_data_1.text(), self.le_data_1, self.ready_data_1))
        self.le_data_1.returnPressed.connect(lambda: self.update_last_dir(self.le_data_1.text()))
        self.le_data_2.returnPressed.connect(lambda: self.prepare_read_data(self.le_data_2.text(), self.le_data_2, self.ready_data_2))
        self.le_data_2.returnPressed.connect(lambda: self.update_last_dir(self.le_data_2.text()))
        self.tb_plot.pressed.connect(self.plot_data)
        self.tb_data_1.pressed.connect(lambda: self.open_file_browser(self.le_data_1))
        self.tb_data_2.pressed.connect(lambda: self.open_file_browser(self.le_data_2))
        self.cb_sym.currentTextChanged.connect(self.set_symmetry_operations)
        self.btn_clear.pressed.connect(self.clear_all)
        
        self.HKL_1 = OrderedDict()
        self.HKL_2 = OrderedDict()
        self.threadpool = QThreadPool()
        self.ready_data_1 = False
        self.ready_data_2 = False
        self.data_1 = None
        self.data_2 = None
        self.last_dir = None
        
        self.group_scale = QButtonGroup()
        self.group_scale.addButton(self.rb_scale_1)
        self.group_scale.addButton(self.rb_scale_2)
        
        self.init_custom_styles()
        self.init_symmetry()
        
    def update_last_dir(self, aPath):
        logging.info(self.__class__.__name__)
        self.last_dir = aPath
    
    def open_file_browser(self, aWidget):
        logging.info(self.__class__.__name__)
        aPath = QFileDialog.getOpenFileName(self, 'Open File', self.last_dir, 'SCXRD Data Formats (*.raw, *.fco, *.hkl)', 'hahaha', QFileDialog.DontUseNativeDialog)[0]
        if not os.path.exists(aPath):
            return
        self.last_dir = aPath
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
                                         [[  0,  1,  0],[  1,  0,  0],[  0,  0,  1]]])}
        
        [self.cb_sym.addItem(i) for i in sorted(self.Symmetry.keys())]
        self.cb_sym.setCurrentText('1')
    
    def set_symmetry_operations(self):
        logging.info(self.__class__.__name__)
        self.SymOp = self.Symmetry[self.cb_sym.currentText()]
        self.HKL_1 = OrderedDict()
        self.HKL_2 = OrderedDict()
        self.la_data_sym.setText('-')
        if self.data_1 is not None and self.data_2 is not None:
            self.thread_run(self.dict_symmetry_equivalents, self.data_1, self.HKL_1, 'Io_1', 'Is_1', flag = 'ready_data_1')
            self.thread_run(self.dict_symmetry_equivalents, self.data_2, self.HKL_2, 'Io_2', 'Is_2', flag = 'ready_data_2')
        
    def init_custom_styles(self):
        logging.info(self.__class__.__name__)
        self.tb_style = ('QToolButton          {background-color: rgb(240, 250, 240); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75); border-radius: 5px}'
                         'QToolButton:hover    {background-color: rgb(250, 255, 250); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}'
                         'QToolButton:pressed  {background-color: rgb(255, 255, 255); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}'
                         'QToolButton:checked  {background-color: rgb(220, 220, 220); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}'
                         'QToolButton:disabled {background-color: rgb(220, 200, 200); color: rgb(  0,   0,   0); border: 1px solid rgb( 75,  75,  75)}')
        
        self.tb_plot.setStyleSheet(self.tb_style)

    def prepare_read_data(self, aPath, aWidget, aFlag):
        logging.info(self.__class__.__name__)
        if not os.path.exists(aPath):
            print('invalid path!')
            return
        aFlag = False
        aWidget.setEnabled(False)
        self.thread_run(self.read_data, aPath, parent_widget = aWidget)
        
    def read_data(self, fname, use_columns = None, used_only = True):
        logging.info(self.__class__.__name__)
        '''
        
        '''
        name, ext = os.path.splitext(fname)
        if ext == '.raw':
            if not use_columns:
                use_columns = (0,1,2,3,4)
            data = np.genfromtxt(fname, usecols=use_columns, delimiter=[4,4,4,8,8,4,8,8,8,8,8,8,3,7,7,8,7,7,8,6,5,7,7,7,2,5,9,7,7,4,6,11,3,6,8,8,8,8,4])
        elif ext == '.fco':
            # delimiter=[6,5,5,11,11,11,11,4])
            # skip_header=26
            if not use_columns:
                use_columns = (0,1,2,4,5,6,7)
            data = np.genfromtxt(fname, skip_header=26, usecols=use_columns)
            if used_only:
                data = data[data[::,6] == 0]
            data = data[:,[0,1,2,3,4,5]]
        elif ext == '.sortav':
            if not use_columns:
                use_columns = (0,1,2,3,6)
            data = np.genfromtxt(fname, usecols=use_columns, comments='c')
        elif ext == '.hkl':
            with open(fname) as ofile:
                temp = ofile.readline()
            if len(temp.split()) == 4 and 'NDAT' in temp:
                # XD2006
                # HEADER:XDNAME F^2 NDAT 7
                # delimiter=[4,4,4,2,8,8,8])
                if not use_columns:
                    use_columns = (0,1,2,4,5)
                data = np.genfromtxt(fname, skip_header=1, usecols=use_columns)
            else:
                # SHELX
                # delimiter=[4,4,4,8,8,4]
                # skip_footer=17
                if not use_columns:
                    use_columns = (0,1,2,3,4)
                data = np.genfromtxt(fname, skip_footer=17, usecols=use_columns, delimiter=[4,4,4,8,8,4])
        else:
            data = None
        return data
    
    def dict_symmetry_equivalents(self, data, HKL, key_Io, key_Is):
        logging.info(self.__class__.__name__)
        '''
         TO_CHECK: can loops be merged?
        '''
        use_stl = False
        for r in data:
            h, k, l, Io, Is = r[:5]
            if len(r) == 6:
                use_stl = True
                stl = r[5]
            hkl = tuple(np.unique(np.array([h,k,l]).dot(self.SymOp), axis=0)[0])
            if hkl in HKL:
                if key_Io in HKL[hkl]:
                    HKL[hkl][key_Io].append(Is)
                    HKL[hkl][key_Is].append(Is)
                else:
                    HKL[hkl][key_Io] = [Is]
                    HKL[hkl][key_Is] = [Is]
            else:
                if use_stl:
                    HKL[hkl] = {key_Io:[Is], key_Is:[Is], 'stl':stl}
                else:
                    HKL[hkl] = {key_Io:[Is], key_Is:[Is]}
    
    def calculate_statistics(self):
        logging.info(self.__class__.__name__)
        multi = []
        meaIo = []
        medIo = []
        rIsig = []
        rIstd = []
        hkl   = []
        stl   = []
        for (h,k,l) in self.HKL_1:
            if (h,k,l) in self.HKL_2:
                Io_mean_1 = np.mean(self.HKL_1[(h,k,l)]['Io_1'])
                Io_mean_2 = np.mean(self.HKL_2[(h,k,l)]['Io_2'])
                Io_medi_1 = np.median(self.HKL_1[(h,k,l)]['Io_1'])
                Io_medi_2 = np.median(self.HKL_2[(h,k,l)]['Io_2'])
                Io_std_1  = np.std(self.HKL_1[(h,k,l)]['Io_1'])
                Io_std_2  = np.std(self.HKL_2[(h,k,l)]['Io_2'])
                Is_mean_1 = np.mean(self.HKL_1[(h,k,l)]['Is_1'])
                Is_mean_2 = np.mean(self.HKL_2[(h,k,l)]['Is_2'])
                if 'stl' in self.HKL_1[(h,k,l)]:
                    stl.append(self.HKL_1[(h,k,l)]['stl'])
                multi.append((len(self.HKL_1[(h,k,l)]['Io_1']), len(self.HKL_2[(h,k,l)]['Io_2'])))
                meaIo.append((Io_mean_1, Io_mean_2))
                medIo.append((Io_medi_1, Io_medi_2))
                rIsig.append((Io_mean_1 / Is_mean_1, Io_mean_2 / Is_mean_2))
                rIstd.append((Io_mean_1 / Io_std_1, Io_mean_2 / Io_std_2))
                hkl.append((h,k,l))
            else:
                print('> unmatched: ({:3}{:3}{:3}) {}'.format(int(h), int(k), int(l), self.HKL_1[(h,k,l)]))

        self.multi = np.asarray(multi)
        self.meaIo = np.asarray(meaIo)
        self.medIo = np.asarray(medIo)
        self.rIsig = np.asarray(rIsig)
        self.rIstd = np.asarray(rIstd)
        self.hkl   = np.asarray(hkl)
        self.stl   = np.asarray(stl)
    
    def plot_data(self):
        logging.info(self.__class__.__name__)
        _SIGCUT = round(self.db_sigcut.value(), 1)
        _SCALE  = self.rb_scale_1.isChecked()
        _SAVE   = self.cb_save.isChecked()
        _FILE_1 = self.le_data_1.text()
        _FILE_2 = self.le_data_2.text()
        
        mpl.rcParams['figure.figsize'] = [13.66, 7.68]
        mpl.rcParams['figure.dpi'] = 100
        mpl.rcParams['savefig.dpi'] = 100
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['legend.fontsize'] = 12
        mpl.rcParams['figure.titlesize'] = 12
        
        fig = plt.figure()
        if self.stl.size:
            grid = plt.GridSpec(12, 13, wspace=0.0, hspace=0.0)
        else:
            grid = plt.GridSpec(7, 13, wspace=0.0, hspace=0.0)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.08, wspace=0.0, hspace=0.0)
        
        sigcut = _SIGCUT
        scale = self.ds_scale.value()
        if _SCALE:
            scale = np.nansum(np.prod(self.meaIo, axis=1))/np.nansum(np.square(self.meaIo[:,0]))
        fig.suptitle('Scalefactor: {:6.3f}, cutoff: {}, symmetry: {}\n1: {}\n2: {}'.format(scale, sigcut, self.cb_sym.currentText(), _FILE_1, _FILE_2))
            
        f1 = self.meaIo[:,0]*scale
        f2 = self.meaIo[:,1]
        rIsig = self.rIsig
        
        f1cut = f1[(rIsig[:,0] > sigcut) & (rIsig[:,1] > sigcut)]
        f2cut = f2[(rIsig[:,0] > sigcut) & (rIsig[:,1] > sigcut)]
        
        if self.stl.size:
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
        p00.set_xlabel(r'$I_{1}$')
        p00.set_ylabel(r'$I_{2}$')
        p00.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        p00.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        x = np.log10(f1cut)
        y = np.log10(f2cut)
        p01.scatter(x, y, s=4, color='#37A0CB')
        p01.plot([np.nanmin(x), np.nanmax(x)],[np.nanmin(x), np.nanmax(x)], 'k-', lw=1.0)
        p01.set_xlabel(r'$\log(I_{1})$')
        p01.set_ylabel(r'$\log(I_{2})$')
        
        facut = (f1cut + f2cut) / 2.
        x = np.log10(facut)
        #y = f1[rIsig[:,0] > sigcut] / f2[rIsig[:,0] > sigcut]
        
        y = (f1cut - f2cut)/(facut)
        
        #p1x.set_title(r'$\frac{I_o(\#1)}{I_o(\#2)}\ vs\ \log(I_o(\#1))$')
        p1x.scatter(x, y, s=4, alpha=0.5, picker=4, color='#37A0CB')
        p1x.plot([np.min(x), np.max(x)], [0,0], 'k-', lw=1.0)
        
        p1x.spines['left'].set_visible(False)
        p1x.spines['bottom'].set_visible(False)
        p1x.xaxis.set_visible(False)
        p1x.yaxis.set_visible(False)
        #p1x.axis('off')
        #p1x.set_ylim([-2.0, 2.0])
        
        if self.stl.size:
            stl = self.stl[(rIsig[:,0] > sigcut) & (rIsig[:,1] > sigcut)]
            #p2x.set_title(r'$\frac{I_o(\#1)}{I_o(\#2)}\ vs\ \log(I_o(\#1))$')
            p2x.scatter(stl, y, s=4, alpha=0.5, picker=4, color='#37A0CB')
            p2x.plot([np.min(stl), np.max(stl)], [0,0], 'k-', lw=1.0)
            p2x.set_ylabel(r'$(I_{1}\ -\ I_{2})\ /\ \left<I_{1,2}\right>$')
            p2x.set_xlabel(r'$sin(\left(\theta\right>)/\lambda$')
            p2x.spines['left'].set_visible(False)
            p2x.yaxis.set_visible(False)
            
            h2y.hist(y[(~np.isnan(y)) & (y<2.) & (y>-2.)], 400, color='#003e5c', histtype='stepfilled', orientation='horizontal')
            h2y.xaxis.set_visible(False)
            h2y.invert_xaxis()
            h2y.spines['top'].set_visible(False)
            h2y.spines['bottom'].set_visible(False)
            h2y.set_ylabel(r'$(I_{1}\ -\ I_{2})\ /\ \left<I_{1,2}\right>$')
        
            h2x.hist(stl[~np.isnan(stl)], 400, color='#003e5c', histtype='stepfilled', orientation='vertical')
            h2x.yaxis.set_visible(False)
            h2x.spines['left'].set_visible(False)
            h2x.spines['right'].set_visible(False)
            h2x.invert_yaxis()
            h2x.set_xlabel(r'$\log(\left<I_{1,2}\right>)$')
            
        h1y.hist(y[(~np.isnan(y)) & (y<2.) & (y>-2.)], 400, color='#003e5c', histtype='stepfilled', orientation='horizontal')
        #h1y.set_ylim([-2.0, 2.0])
        h1y.xaxis.set_visible(False)
        h1y.invert_xaxis()
        h1y.spines['top'].set_visible(False)
        h1y.spines['bottom'].set_visible(False)
        #h1y.spines['right'].set_visible(False)
        #h1y.set_ylabel(r'$I_o(\#1)\ /\ I_o(\#2)$')
        h1y.set_ylabel(r'$(I_{1}\ -\ I_{2})\ /\ \left<I_{1,2}\right>$')
        
        h1x.hist(x[~np.isnan(x)], 400, color='#003e5c', histtype='stepfilled', orientation='vertical')
        h1x.yaxis.set_visible(False)
        #h1x.spines['top'].set_visible(False)
        h1x.spines['left'].set_visible(False)
        h1x.spines['right'].set_visible(False)
        h1x.invert_yaxis()
        h1x.set_xlabel(r'$\log(\left<I_{1,2}\right>)$')

        if _SAVE:
            #name_1, ext = os.path.splitext(_FILE_1)
            name_1 = os.path.split(os.path.split(_FILE_1)[0])[1]
            name_2 = os.path.split(os.path.split(_FILE_2)[0])[1]
            #name_2, ext = os.path.splitext(os.path.basename(_FILE_2))
            pname = '1{}_2{}_c{}_compare'.format(name_1, name_2, sigcut)
            plt.savefig(pname + '.pdf', transparent=True)
            plt.savefig(pname + '.png', dpi=600, transparent=True)
        
        self.ann_hkl = None
        def on_pick(event):
            '''
             This will fail if a new plot is shown, the first will index into the wrong self.hkl
             solution: index plots and index into a list of self.hkl lists
            '''
            if event.mouseevent.button == 3 and self.ann_hkl is not None:
                self.ann_hkl.remove()
                self.ann_hkl = None
                plt.draw()
                return
                
            xdata = event.mouseevent.xdata
            ydata = event.mouseevent.ydata
            x = event.mouseevent.x
            y = event.mouseevent.y
            ind = event.ind
            h,k,l = map(int, self.hkl[(rIsig[:,0] > sigcut) & (rIsig[:,1] > sigcut)][ind][0])
            if self.ann_hkl is not None:
                self.ann_hkl.remove()
                self.ann_hkl = None
            self.ann_hkl = plt.annotate('hkl:({:3}{:3}{:3})'.format(h,k,l), xy=(x,y), xytext=(x,y), xycoords='figure pixels', textcoords='figure pixels',)
            plt.draw()
            
        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.show()
            
    def on_thread_result(self, r):
        logging.info(self.__class__.__name__)
        data, kwargs = r
        if data is not None:
            if 'parent_widget' in kwargs and kwargs['parent_widget'] == self.le_data_1:
                self.data_1 = data
                self.la_data_1.setText('Reflections: {}'.format(str(len(data))))
                self.thread_run(self.dict_symmetry_equivalents, self.data_1, self.HKL_1, 'Io_1', 'Is_1', flag = 'ready_data_1')
            elif 'parent_widget' in kwargs and kwargs['parent_widget'] == self.le_data_2:
                self.data_2 = data
                self.la_data_2.setText('Reflections: {}'.format(str(len(data))))
                self.thread_run(self.dict_symmetry_equivalents, self.data_2, self.HKL_2, 'Io_2', 'Is_2', flag = 'ready_data_2')

    def clear_all(self):
        self.le_data_1.setText('')
        self.le_data_2.setText('')
        self.la_data_1.setText('-')
        self.la_data_2.setText('-')
        self.la_data_sym.setText('-')
        self.ready_data_1 = False
        self.ready_data_2 = False
        self.data_1 = None
        self.data_2 = None
        self.last_dir = None
        self.HKL_1.clear()
        self.HKL_2.clear()
        self.hkl = []
        
    def on_thread_finished(self, kwargs):
        logging.info(self.__class__.__name__)
        if self.threadpool.activeThreadCount() == 0:
            self.statusBar.showMessage('ready.')
        if 'flag' in kwargs:
            if kwargs['flag'] == 'ready_data_1':
                self.le_data_1.setEnabled(True)
                self.ready_data_1 = True
                if self.ready_data_1 and self.ready_data_2:
                    self.thread_run(self.calculate_statistics, flag = 'ready_data_all')
            elif kwargs['flag'] == 'ready_data_2':
                self.le_data_2.setEnabled(True)
                self.ready_data_2 = True
                if self.ready_data_1 and self.ready_data_2:
                    self.thread_run(self.calculate_statistics, flag = 'ready_data_all')
            elif kwargs['flag'] == 'ready_data_all' and self.threadpool.activeThreadCount() == 0:
                #####################################
                ## calculations are finished here! ##
                #####################################
                self.tb_plot.setEnabled(True)
                self.cb_sym.setEnabled(True)
                self.la_data_sym.setText(str(len(self.hkl)))
    
    def thread_run(self, fn, *args, **kwargs):
        logging.info(self.__class__.__name__)
        self.tb_plot.setEnabled(False)
        w = Worker(fn, *args, **kwargs)
        w.signals.result.connect(self.on_thread_result)
        w.signals.finished.connect(self.on_thread_finished)
        self.threadpool.start(w)
        ## ALWAYS DIASBLE THE SYM SWITCH ##
        self.cb_sym.setEnabled(False)
        ###################################
        self.statusBar.showMessage('I\'m thinking ...')
        if 'parent_widget' in kwargs:
            kwargs['parent_widget'].setEnabled(False)

def main():
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.setWindowTitle('Compare SCXRD Data, {}'.format(_REVISION))
    ui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    # Remove existing handlers, Python creates a
    # default handler that goes to the console
    # and will ignore further basicConfig calls
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    # create logger
    logging.basicConfig(level=logging.INFO, format='%(message)20s > %(funcName)s')
    main()