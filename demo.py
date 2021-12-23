from enum import EnumMeta
from test_text import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import re
import string
import matplotlib.pyplot as plt

device = 'cuda'
model_id = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
def divide_sentences(text):
    sentence_list = re.split('([.?])',text)
    sentence_list.append("")
    sentence_list = ["".join(i) for i in zip(sentence_list[0::2],sentence_list[1::2])]
    del sentence_list[-1]
    return len(sentence_list),sentence_list


def get_loss(sentence,encoded=True):
    '''
    step 1: abcde->abcd:sentence, e:target
    step 2: calculate loss
    '''
    if ~encoded:
        sentence=tokenizer.encode(sentence)
    input=sentence[:-1]
    target=sentence[-1:] #list of len=1
    
    final_input=torch.tensor(input).to(device)
    target=torch.tensor(target).to(device)
    #print(final_input,target)
    output=model(final_input,return_dict=True)[0]
    output = output[-1,:]
    loss_fct = torch.nn.CrossEntropyLoss()
    #print(output.view(-1, output.size(-1)).shape, target.shape)
    loss = loss_fct(output.view(-1, output.size(-1)), target)
    return loss.cpu().detach()

def get_loss_for_single_sentence(previous_input,sentence):
    sentence=tokenizer.encode(sentence)
    length=len(sentence)
    tmp_loss_list=[]
    for index in range(len(sentence)):
        if index==0:
            continue
        tmp_word=tokenizer.decode(sentence[index])
        tmp_word=tmp_word.replace(' ','')
        if tmp_word in string.punctuation:
            continue
        else:
            tmp_loss=get_loss(previous_input+sentence[:index+1])
            tmp_loss_list.append(tmp_loss)
    return np.mean(tmp_loss_list)

def get_sentences(sentence_list,start,end):
    '''
    sentences[start:end]
    '''
    if start>=end:
        return ''
    txt=''
    for i in range(start,end):
        txt=txt+sentence_list[i]
    return txt

def process_one_passage(passage):
    _,sentence_list=divide_sentences(passage)
    loss_list=list()
    for index,sentence in enumerate(sentence_list):
        '''
        calculate loss of sentence[index] given first index-1 sentences, pass first 2 sentences.
        loss[i]=p(s[i+2]|s[0,1,...,i+1])
        '''
        if index<2:
            continue
        else:
            previous_input=get_sentences(sentence_list,0,index)
            previous_input=tokenizer.encode(previous_input)
            loss_list.append(get_loss_for_single_sentence(previous_input,sentence))
            print(index)
    return loss_list

def plot_loss(loss_list):
    N_sentences=len(loss_list)
    label_x=range(2,N_sentences+2,1)
    plt.bar(label_x,loss_list)
    plt.xticks(label_x,label_x[::1])
    plt.show()
      

loss_list=process_one_passage(text_04)
plot_loss(loss_list)
    

