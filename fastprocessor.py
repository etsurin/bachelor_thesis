from torch.utils.data import DataLoader,Dataset
from enum import EnumMeta
from text import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import copy
import numpy as np
import string
import matplotlib.pyplot as plt
from divide_text import *

model_id = 'gpt2-medium'
device = 'cuda'
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
text=list()
label=list()
info=list()  #n-th passage has info[n]-2 annotated sentences 
outputlist=list()
counter=0
max_length = 300  #global variable for the padding length

def get_labels(previous_input,sentence):
    tmplist=list()
    max_length=300
    for i in range(len(previous_input)+1): #pass the first word
        tmplist.append(-100)
    tmplist=tmplist+sentence[1:]
    while(1):  # do padding if whole input is shorter than max_length
        if len(tmplist)==max_length:
            break
        tmplist.append(-100)
    return tmplist

def get_inputs(previous_input,sentence):
    if len(previous_input+sentence):
        tmplist=previous_input+sentence
    else:
        tmplist=list()
    while(1):
        if len(tmplist)==max_length:
            break
        tmplist.append(50256)
    return tmplist

def process_one_passage(passage,datasetname):
    if 'ami' in datasetname:
        sentence_list=passage.split('\n')
    else:
        sentence_list=divide_sentences(passage)
    previous_input=list()
    info.append(len(sentence_list)-2)
    for index,sentence in enumerate(sentence_list):
        if index<2:
            previous_input = previous_input+tokenizer.encode(sentence)
            continue
        else:
            sentence=tokenizer.encode(sentence)
            buffer_size=max_length-len(sentence)
            generatesamples(previous_input[-buffer_size:],sentence)
            previous_input=previous_input+tokenizer.encode(sentence)

def generatesamples(previous_input,sentence):
    '''
    previous_input and sentence should be encoded
    '''
    inputs=get_inputs(previous_input,sentence)
    labels=get_labels(previous_input,sentence)
    text.append(np.array(inputs))
    label.append(np.array(labels))

class Annotationset(Dataset):
    def __init__(self):
        self.data = text
        self.labels = label
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

def makeannotationset(datasetname):
    data=processjson(datasetname)
    for index in tqdm(range(len(data))):
        process_one_passage(data[index]['description'],datasetname)
    print('****finish preprocessing****total %d samples****'%len(text))

def main():
    datasetname='ami_test'
    makeannotationset(datasetname)
    annotationset=Annotationset()
    dataloader=torch.utils.data.DataLoader(annotationset, batch_size=1,shuffle=False, num_workers=2)

    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    print('****do running****')
    with torch.no_grad():
        for index,data in tqdm(enumerate(dataloader)):
            input,target = data
            target=target.type(torch.int64).to(device)
            input=input.type(torch.int64).to(device)
            output = model(input,labels=target)
            loss=output.loss.detach().cpu().item()
            # print(loss)
            torch.cuda.empty_cache()
            outputlist.append(loss)
            f = open('losses_%s.pkl' %datasetname, 'wb')
            pickle.dump(outputlist, f)
            f = open('sentenceinfo_%s.pkl' %datasetname, 'wb')
            pickle.dump(info, f)
        print(len(outputlist))

if __name__=='__main__':
    main()
    # data=processjson('testcut')
    # process_one_passage(data[97]['description'],'testcut')
    


