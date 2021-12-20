from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import re
import string

device = 'cuda'
model_id = 'gpt2-large'
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
    #step 1: abcde->abcd:sentence, e:target
    #step 2: calculate loss
    if ~encoded:
        sentence=tokenizer.encode(sentence)
    input=sentence[:-1]
    target=sentence[-1:] #list of len=1
    
    final_input=torch.tensor(input).to(device)
    target=torch.tensor(target).to(device)
    print(final_input,target)
    output=model(final_input,return_dict=True)[0]
    output = output[-1,:]
    loss_fct = torch.nn.CrossEntropyLoss()
    print(output.view(-1, output.size(-1)).shape, target.shape)
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
    return tmp_loss_list, np.mean(tmp_loss_list)


print(get_loss_for_single_sentence([],'Your sentences'))

    

