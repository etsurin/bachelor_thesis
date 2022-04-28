print('***loading libraries***')
import nltk
import json
from enum import EnumMeta
from transformers.utils import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import numpy as np
import matplotlib.pyplot as plt
import spacy
from datasets import load_metric


print('***loading parsing model***')
tool = spacy.load('en_core_web_sm')
print('***loading evaluation model***')
compute_metric = load_metric("./rouge")
print('***loading annotation model***')
tokenizer_G = GPT2Tokenizer.from_pretrained('gpt2-medium')
gptmodel = GPT2LMHeadModel.from_pretrained('gpt2-medium')
print('***loading generation models***')
tokenizer_B = BartTokenizer.from_pretrained('BART_c')
model_c = BartForConditionalGeneration.from_pretrained('BART_c')
model_f = BartForConditionalGeneration.from_pretrained('BART_f')

def processjson(filename):
    filesrc='../textfile/'+filename+'.json'
    with open(filesrc,'r') as datafile:
        data = datafile.read()
        data = '['+data.replace('}{','},{')+']'
        data = json.loads(data)
    return data

def rouge(preds, references,rouge_types=None):
    '''
    preds and references can be both str or list[str]
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    '''
    if isinstance(preds,str):
        preds = ["\n".join(nltk.sent_tokenize(preds))]
        references = ["\n".join(nltk.sent_tokenize(references))]
    else:
        preds = [pred.strip() for pred in preds]
        references = [reference.strip() for reference in references]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        references = ["\n".join(nltk.sent_tokenize(reference)) for reference in references]
    result = compute_metric.compute(predictions=preds,references = references,rouge_types=rouge_types,use_stemmer=True)
    result = {key: round(value.mid.fmeasure * 100,2) for key, value in result.items()}
    return result

def divide_sentences(text):
    processed_text = tool(text)
    sentence_list=[]
    for sentence in processed_text.sents:
        sentence_list.append(str(sentence))
    return sentence_list


def divide_text(text,intervals,write_file=False):
    if isinstance(text,str):
        sentence_list=divide_sentences(text)
    else:
        sentence_list=text
    txt=''
    for index,sentence in enumerate(sentence_list):
        if index in intervals:
            txt=txt+'[DIVIDE HERE]'
        txt=txt+sentence
    if write_file:
        txtfile=open('divide_result.txt','w')
        txtfile.write(txt)
        txtfile.close()
        return None
    return txt,txt.split('[DIVIDE HERE]')

def autodivide(lossinfo,sentenceinfo):

    divide_list=list()
    index_skjt=0 #sakujitsu
    index_otti=0 #ototoi
    tight_min=400
    loose_min=750
    max_length=1150
    topnum=int(0.05*len(lossinfo))
    maxvalue=sorted(lossinfo,reverse=True)[:topnum]
    # print(maxvalue)
    index = 1
    safecount=0
    while(index<len(sentenceinfo)):
        # print(index)
        safecount+=1
        if safecount>len(sentenceinfo)+5:
            break
        now_length = sum(sentenceinfo[index_skjt:index])
        if now_length <tight_min:
            if lossinfo[index] in maxvalue and len(divide_list) :
                if lossinfo[index]>=lossinfo[divide_list[-1]]:
                    divide_list[-1]=index
                    # print('case1_1: divide id: %d loss:%f' %(index,lossinfo[index]))
                    index_skjt=index
                    index+=1
                else:
                    index+=1
                continue
            else:
                index+=1
                continue
        
        if now_length >=tight_min and now_length< loose_min:
            if lossinfo[index] in maxvalue:
                divide_list.append(index)
                # print('case2: divide id: %d loss:%f' %(index,lossinfo[index]))
                index_otti=index_skjt
                index_skjt=index
                index+=1
            else:
                index+=1
            continue
        if now_length>=loose_min and now_length<=max_length:
            max_loss=0
            tmp_length=now_length
            tmp_index=index
            while(1):
                if tmp_index>=len(sentenceinfo):
                    break
                if tmp_length>max_length:
                    break                
                if lossinfo[tmp_index]>=max_loss:
                    # print('maxloss=%f, tmp_index=%d, lossinfo[tmp_index]=%f' %(max_loss, tmp_index, lossinfo[tmp_index]))
                    divideindex=tmp_index
                    max_loss=lossinfo[tmp_index]
                tmp_length=tmp_length+sentenceinfo[tmp_index]
                tmp_index+=1

            divide_list.append(divideindex)
            # print('case3: divide id: %d loss:%f' %(divideindex,lossinfo[divideindex]))
            index=divideindex+1
            index_otti=index_skjt
            index_skjt=divideindex
            
        if now_length>=max_length:
            divide_list.append(index)
            index_otti=index_skjt
            index_skjt=index
            index=index+1
    
    if len(divide_list) and len(sentenceinfo[divide_list[-1]:]) < tight_min:
        del divide_list[-1]
    divide_list=[index+2 for index in divide_list]
    return divide_list

def get_labels(previous_input,sentence):
    tmplist=list()
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

def generatesamples(previous_input,sentence,text,label):
    '''
    previous_input and sentence should be encoded
    '''
    inputs=get_inputs(previous_input,sentence)
    labels=get_labels(previous_input,sentence)
    text.append(np.array(inputs))
    label.append(np.array(labels))

def process_one_passage(tokenizer,sentence_list):
    text=list()
    label=list()
    previous_input=list()
    for index,sentence in enumerate(sentence_list):
        if index<2:
            previous_input = previous_input+tokenizer.encode(sentence)
            continue
        else:
            sentence=tokenizer.encode(sentence)[:200]
            buffer_size=max_length-len(sentence)
            generatesamples(previous_input[-buffer_size:],sentence,text,label)
            previous_input=previous_input+tokenizer.encode(sentence)
    return text,label

def get_info_list(tokenizer, sentence_list):
    output = list()
    for index,sentence in enumerate(sentence_list):
        if index<2:
            continue
        output.append(len(tokenizer.encode(sentence)))
    return output

def summary(raw_text,reference=None):
    print('****preprocessing****')
    logging.set_verbosity_error()
    device = 'cuda'
    global tool,compute_metric, tokenizer_G,tokenizer_B, gptmodel,model_c,model_f
    global max_length  #global variable for the padding length
    max_length =  300
    gptmodel = gptmodel.to(device)
    gptmodel.eval()
    sentence_list = divide_sentences(raw_text)
    text,labels = process_one_passage(tokenizer_G,sentence_list)
    infolist = get_info_list(tokenizer_G, sentence_list)
    outputlist = list()
    print('****calculating loss****')
    with torch.no_grad():
        for index,data in enumerate(text):
            input,target = data,labels[index]
            target=torch.tensor(target,dtype=torch.int64).to(device)
            input=torch.tensor(input,dtype=torch.int64).to(device)
            output = gptmodel(input,labels=target)
            loss=output.loss.cpu().item()
            # print(loss)
            torch.cuda.empty_cache()
            outputlist.append(loss)
    gptmodel = gptmodel.cpu()
    print('****dividing text****')
    divide_indexes = autodivide(outputlist,infolist)
    dividedtext,textlist = divide_text(sentence_list,divide_indexes)
    print('****coarse stage****')
    model_c=model_c.to(device)
    input_list = tokenizer_B(textlist,max_length=1024,padding=True,truncation=True,return_tensors='pt')['input_ids'].to(device)
    dataloader=torch.utils.data.DataLoader(input_list, batch_size=5,shuffle=False, num_workers=0)
    output_list_tmp = list()
    with torch.no_grad():
        for item in dataloader:
            summary_ids = model_c.generate(item, num_beams=4, max_length=160, early_stopping=True).cpu()
            output_list_tmp.append(summary_ids)
    output_list_tmp = [b for a in output_list_tmp for b in a]
    # output_list_tmp = torch.tensor(output_list_tmp)
    summary = tokenizer_B.batch_decode(output_list_tmp, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_c = ' '.join(summary)
    model_c = model_c.cpu()
    print('****fine-grained stage****')
    model_f = model_f.to(device)
    input_f = tokenizer_B(output_c,max_length=1024,return_tensors='pt')['input_ids'].to(device)
    summary_ids = model_f.generate(input_f, num_beams=4, max_length=300, early_stopping=True).cpu()
    summary=[tokenizer_B.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    output = summary[0]
    print('****finished generating****')
    print('\nsummary:')
    print(output)
    if reference !=None:
        rougescore = rouge(output,reference)
        print('rougescore:\n',rougescore)

if __name__=='__main__':
    rawdata = processjson('arxiv_val')[47]
    raw_text = rawdata['description']
    reference = rawdata['abstract']
    summary(raw_text, reference)