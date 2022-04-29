from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from ui_demo import Ui_MainWindow
from PyQt5.QtCore import pyqtSignal,Qt
import os
import sys
def read_dir(path):
    rawlist = os.listdir(path)
    dirlist = list()
    filelist = list()
    for index,file in enumerate(rawlist):
        tmp_path =  os.path.join(path,file)
        if os.path.isdir(tmp_path):
            dirlist.append(file)
        else:
            if len(file.split('.'))==1:
                continue
            filetype = file.split('.')[1]
            if filetype == 'txt':
                filelist.append(file)
    return dirlist,filelist

class MainPageWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(MainPageWindow, self).__init__(parent)
        self.setupUi(self)
        self.path1 = os.path.abspath('../')
        self.path2 = os.path.abspath('../')
        self.update_left()
        self.update_right()
        self.initUI()
    
    def initUI(self):
        self.pushButton_2.clicked.connect(lambda:self.return_upper('left'))
        self.pushButton_3.clicked.connect(lambda:self.return_upper('right'))        

    def update_right(self):        
        self.comboBox_2.clear()        
        dirlist_r, filelist_r = read_dir(self.path2)   
        self.len_dir_r = len(dirlist_r)
        self.label_2.setText(self.path2)
        self.comboBox_2.addItem('--select reference(optional) --')
        self.comboBox_2.addItems(dirlist_r+filelist_r)
        self.comboBox_2.currentIndexChanged.connect(self.selectionChange_summary)


    def update_left(self):
        self.comboBox.clear()
        dirlist_l, filelist_l = read_dir(self.path1)
        self.len_dir_l = len(dirlist_l)
        self.label.setText(self.path1)
        self.comboBox.addItem('--select text--')
        self.comboBox.addItems(dirlist_l+filelist_l)
        self.comboBox.currentIndexChanged.connect(self.selectionChange_text)


    def return_upper(self,sender):
        if sender == 'left':
            self.pushButton_2.clicked.disconnect()
            self.comboBox.currentIndexChanged.disconnect()
            self.path1 = os.path.dirname(os.path.abspath(self.path1))
            self.update_left()
            self.text_buffer.clear()
            self.pushButton_2.clicked.connect(lambda:self.return_upper('left'))
            self.comboBox.currentIndexChanged.connect(self.selectionChange_text)
        else:
            self.pushButton_3.clicked.disconnect()
            self.comboBox_2.currentIndexChanged.disconnect()        
            self.path2 = os.path.dirname(os.path.abspath(self.path2))
            self.update_right()
            self.summ_buffer.clear()
            self.pushButton_3.clicked.connect(lambda:self.return_upper('right'))
            self.comboBox_2.currentIndexChanged.connect(self.selectionChange_summary)


    def selectionChange_text(self,i):
        self.comboBox.currentIndexChanged.disconnect()
        i=i-1
        if i<self.len_dir_l:  #choose path
            self.path1 = os.path.join(self.path1,self.comboBox.currentText())
            self.text_buffer.clear()
            self.update_left()
        else:   #choose file
            textfile = open(os.path.join(self.path1,self.comboBox.currentText()),'r',encoding = 'utf-8')
            text = textfile.read()
            self.text_buffer.setText(text)
            textfile.close()
        self.comboBox.currentIndexChanged.connect(self.selectionChange_text)

    def selectionChange_summary(self,i):
        self.comboBox_2.currentIndexChanged.disconnect()
        i=i-1
        if i<self.len_dir_r:  #choose path
            self.path2 = os.path.join(self.path2,self.comboBox_2.currentText())
            self.summ_buffer.clear()
            self.update_right()
        else:
            textfile = open(os.path.join(self.path2,self.comboBox_2.currentText()),'r',encoding = 'utf-8')
            text = textfile.read()
            self.summ_buffer.setText(text)
            textfile.close()            
        self.comboBox_2.currentIndexChanged.connect(self.selectionChange_summary)
      


if __name__=='__main__':
    app = QApplication(sys.argv)
    mainWindow = MainPageWindow()
    mainWindow.show()
    sys.exit(app.exec_())        

