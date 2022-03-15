from enum import EnumMeta
from text import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import string
import matplotlib.pyplot as plt
from divide_text import *

device = 'cuda'
model_id = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_id)


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
    '''
    previous_input and sentence should be encoded
    '''
    length=len(sentence)
    tmp_loss_list=[]
    for index in range(len(sentence)):
        if index==0:
            continue
        tmp_word=tokenizer.decode(sentence[index]).strip()
        if tmp_word in string.punctuation:
            continue
        else:
            tmp_loss=get_loss(previous_input+sentence[:index+1])
            tmp_loss_list.append(tmp_loss)
    return np.mean(tmp_loss_list)



def process_one_passage(passage,datasetname):
    if 'ami' in datasetname:
        sentence_list=passage.split('\n')
    else:
        sentence_list=divide_sentences(passage)
    loss_list=list()
    previous_input=list()
    for index,sentence in enumerate(sentence_list):
        '''
        calculate loss of sentence[index] given first index-1 sentences, pass first 2 sentences.
        loss[i]=p(s[i+2]|s[0,1,...,i+1])
        '''
        if index<2:
            previous_input = previous_input+tokenizer.encode(sentence)
            continue
        else:
            sentence=tokenizer.encode(sentence)
            max_length = 300
            buffer_size=max_length-len(sentence)
            loss_list.append(get_loss_for_single_sentence(previous_input[-buffer_size:],sentence))
            print(index,len(previous_input))
            previous_input=previous_input+tokenizer.encode(sentence)
    return loss_list

def plot_loss(loss_list,divide_list=[]):
    N_sentences=len(loss_list)
    label_x=range(2,N_sentences+2,1)
    plt.bar(label_x,loss_list)
    if len(divide_list):
        marklist=list()
        for index in divide_list:
            marklist.append(loss_list[index-2]+0.1)
        plt.scatter(divide_list,marklist,c='r',marker='v')

    plt.xticks(label_x,label_x[::1])
    plt.show()
      

def main():
    datasetname='ami_test'
    data=processjson(datasetname)
    print(process_one_passage(data[8]['description'],datasetname))

if __name__=='__main__':
    main()
    

