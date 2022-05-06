from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from ui_demo import Ui_MainWindow
from init_demo import Ui_Init
from PyQt5.QtCore import pyqtSignal,Qt,QThread
import os
import sys
from utils_ui import *
import time
import threading



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

def load():
    global tool,compute_metric,tokenizer_G,tokenizer_B,gptmodel,model_c,model_f
    tool = spacy.load('en_core_web_sm')
    compute_metric = load_metric("./rouge")
    tokenizer_G = GPT2Tokenizer.from_pretrained('gpt2-medium')
    gptmodel = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    tokenizer_B = BartTokenizer.from_pretrained('BART_c')
    model_c = BartForConditionalGeneration.from_pretrained('BART_c')
    model_f = BartForConditionalGeneration.from_pretrained('BART_f')

class MainPageWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(MainPageWindow, self).__init__(parent)
        self.setupUi(self)
        self.path1 = os.path.abspath('../')
        self.path2 = os.path.abspath('../')
        self.update_left()
        self.update_right()
        self.initUI()
        self.enable_generate(False)
    
    def enable_generate(self,flag):
        if flag:
            self.pushButton.setText('generate')
            self.pushButton.setEnabled(True) 
        else:
            self.pushButton.setText('empty input')
            self.pushButton.setEnabled(False)


    def initUI(self):
        self.pushButton.clicked.connect(self.generate)
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

    def generate(self):
        text = self.text_buffer.text()
        self.metric.setText('generating...')
        if len(self.summ_buffer.text()):
            self.call_generator = generator(text,self.summ_buffer.text())
        else:
            self.call_generator = generator(text)
        self.call_generator.result.connect(self.update_summ)
        self.call_generator.score.connect(self.update_info)
        self.call_generator.start()
        self.call_generator.exec()

    def update_summ(self,result):
        self.generated_summary.setText(result)
    
    def update_info(self,score):
        if len(score):
            self.metric.setText(score)
        else:
            self.metric.setText('finished')

    def selectionChange_text(self,i):
        self.comboBox.currentIndexChanged.disconnect()
        i=i-1
        if i<self.len_dir_l:  #choose path
            self.path1 = os.path.join(self.path1,self.comboBox.currentText())
            self.text_buffer.clear()
            self.generated_summary.clear()
            self.metric.clear()
            self.enable_generate(False)
            self.update_left()
        else:   #choose file
            textfile = open(os.path.join(self.path1,self.comboBox.currentText()),'r',encoding = 'utf-8')
            text = textfile.read()
            self.text_buffer.setText(text)
            self.generated_summary.clear()
            self.metric.clear()
            self.enable_generate(True)
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


class generator(QThread):
    result = pyqtSignal(str)
    score = pyqtSignal(str)
    
    def __init__(self,text,reference=None):
        super().__init__()
        self.text = text
        self.reference = reference
    
    def run(self):
        generated_summary,rougescore = summary(raw_text = self.text,tool = tool, compute_metric = compute_metric,tokenizer_G = tokenizer_G,tokenizer_B = tokenizer_B
            ,gpt_model = gptmodel,model_c = model_c,model_f = model_f,reference = self.reference)
        self.result.emit(generated_summary)
        self.score.emit(rougescore)

class Initpage(QMainWindow, Ui_Init):
    def __init__(self,parent=None):
        super(Initpage, self).__init__(parent)
        self.setupUi(self)
        self.initUI()

    def process(self):
        while self.progressBar.value()<99:
            time.sleep(0.54)
            self.progressBar.setValue(self.progressBar.value() + 1)

    def initUI(self):
        self.show()
        load()
        # do_load = threading.Thread(target = load)
        # do_count = threading.Thread(target = self.process)
        # do_load.start()
        # do_count.start()
        # do_count.join()
        # do_load.join()
        self.close()
        mainWindow = MainPageWindow()
        mainWindow.show()

if __name__=='__main__':
    load()
    app = QApplication(sys.argv)
    mainWindow = MainPageWindow()
    mainWindow.show()

    sys.exit(app.exec_())        

