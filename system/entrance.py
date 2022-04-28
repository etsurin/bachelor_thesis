# -*- coding: utf-8 -*-

import os
import time
from utils import summary
def read_dir(path):
    rawlist = os.listdir(path)
    dirlist = list()
    filelist = list()
    for index,file in enumerate(rawlist):
        tmp_path =  os.path.join(path,file)
        if os.path.isdir(tmp_path):
            dirlist.append(file)
        else:
            filetype = file.split('.')[1]
            if filetype == 'txt':
                filelist.append(file)
    return dirlist,filelist

def print_list(list,start_index=0):
    for index, item in enumerate(list,start=start_index):
        print(index,":",item)

def preprocess(file):
    with open(file,'r',encoding='utf-8') as f:
        raw_text = f.read()
        raw_text = raw_text.replace('\n','')
    return raw_text

def select_file(path):
    os.system("cls")
    path = os.path.abspath(path)
    print('current path: %s' %path)
    dirlist,filelist=read_dir(path)
    other = ['return upper folder', 'exit']
    print('***You can generate summaries for these txt files***')
    print_list(filelist)
    print('***You can enter these folders***')
    print_list(dirlist,len(filelist))
    print_list(other,len(filelist)+len(dirlist))
    index = input('***input the file number or folder number: ')
    if not(index.isdecimal()):
        print('***invalid input, will return to main page***')
        time.sleep(1)
        select_file(path)
        return
    index = int(index)
    if index>=(len(filelist)+len(dirlist)+2):
        print('***invalid input, will return to main page***')
        time.sleep(1)
        select_file(path)
        return        
    if index< len(filelist):  ##choose file
        filepath = os.path.join(path,filelist[index])
        text = preprocess(filepath)
        summary(text)
        tmp = input('***finished, input anything and press enter to return main page***')
        select_file(path)
        return 
    elif index>=len(filelist) and index<(len(filelist)+len(dirlist)):
        index = index - len(filelist)
        path = os.path.join(path,dirlist[index])
        select_file(path)
        return
    else:
        index = index-(len(filelist)+len(dirlist))
        if index:  #exit
            return
        else:
            path = os.path.dirname(path)
            select_file(path)
            return
            
def main():
    path = './'
    select_file(path)

if __name__=='__main__':
    main()