from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from divide_text import *
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from annotation import get_divide_indexes

device='cuda'
tokenizer = RobertaTokenizer.from_pretrained("roberta-large-mnli")
tokenizer.truncation='only_first'
tokenizerBart = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# print(tokenizer)
'''
tips for Roberta-large-mnli:
format: [PREMISE]</s></s>[HYPHOTHESIS]
output:[contradiction, neutral, entailment]
'''

def tmpinfo(datasetname):
    info=list()
    data=processjson(datasetname)
    for datadict in tqdm(data):
        passage = datadict['description']
        abstract = datadict['abstract']
        if 'ami' in datasetname:
            sentence_list=passage.split('\n')
        else:
            sentence_list=divide_sentences(passage)
        for sentence in sentence_list:
            info.append(len(tokenizerBart.encode(sentence)))
    return info

def preprocess(datasetname):
    text=list()
    data=processjson(datasetname)
    for datadict in tqdm(data):
        passage = datadict['description']
        abstract = datadict['abstract']
        if 'ami' in datasetname:
            sentence_list=passage.split('\n')
        else:
            sentence_list=divide_sentences(passage)
        # info.append(len(sentence_list))
        for sentence in sentence_list:
            len_sentence = len(tokenizer.encode(sentence))
            buffer_size=510-len_sentence
            abstract = tokenizer.encode(abstract, max_length=buffer_size,truncation=True)
            abstract = tokenizer.decode(abstract)
            item = tokenizer(abstract+'</s>'+sentence,max_length=512,padding='max_length',return_tensors="pt").to(device)
            # print(item)
            text.append(item)
    return text

def main():
    outputlist=list()
    datasetname='ami_train'
    text = preprocess(datasetname)
    model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
    model.eval()
    print('****do running****')
    with torch.no_grad():
        for index,data in tqdm(enumerate(text)):
            # data['input_ids']=data['input_ids'][0]
            # data['attention_mask']=data['attention_mask'][0]   #strange bugs... 
            output = model(**data)
            output = F.softmax(output.logits,dim=1)
            score=(output[0][2]-output[0][0]).item()
            torch.cuda.empty_cache()
            outputlist.append(score)
        f = open('entail_%s.pkl' %datasetname, 'wb')
        pickle.dump(outputlist, f)
        print(len(outputlist))

def debug():
    text=''
    output = tokenizer.encode(text,max_length=512,padding='max_length',  truncation=True)
    print(tokenizer.decode(output))
    # model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
    # with torch.no_grad():
    #     inputs = tokenizer("Two men are playing soccer.</s></s>Two men are playing sports.",max_length=512,padding='max_length', return_tensors="pt").to(device)
    #     print(inputs)
    #     output = model(**inputs)
    #     print(output.logits[0][2].cpu().item())

def entailrank():
    
    scorelist = pickle.load(open('entail_ami_train.pkl','rb'))
    datasetname='ami_train'
    infolist = tmpinfo(datasetname)
    c_ratio = 6
    data = processjson(datasetname)
    p = 0
    type='base'
    anno_indexlist = get_divide_indexes(datasetname)
    newdata = list()
    for index,item in tqdm(enumerate(data)):
        text=item['description']
        if 'ami' in datasetname:
            sentence_list=text.split('\n')
        else:
            sentence_list=divide_sentences(text)
        if type =='base':
            textparts,indexlist = get_snippets(sentence_list,'length')
        else:
            indexlist = anno_indexlist[index]
            _,textparts = divide_text(sentence_list, indexlist[:-1])
        indexlist=[p+tmp for tmp in indexlist]
        for inid,textpart in enumerate(textparts):
            tmp=dict()
            if inid==0:
                start,end=p,indexlist[inid]
            else:
                start,end=indexlist[inid-1],indexlist[inid]
            # print(p,start,end)
            tmplist= list(zip(range(start,end), scorelist[start:end]))
            # print(tmplist)
            # tmplen = len(tokenizerBart.encode(textpart))
            
            textlen=sum(infolist[start:end])
            # print(tmplen,textlen)
            if textlen<150:
                continue
            num=1
            while(1):
                if sum(infolist[start:start+num])>=(textlen/c_ratio):
                    break
                num+=1
            tmplist.sort(key=lambda x:x[1])
            res = sorted([tmp[0] for tmp in tmplist[:num]])
            res=[tmp-p for tmp in res]
            # print(res)
            tmp['description']=textpart
            tmp['abstract']=list2txt(sentence_list,res)
            newdata.append(tmp)
        p=indexlist[-1]
    # with open('textfile/%s_seg_eranno.json' %datasetname, 'w') as f:
    #     for item in newdata:
    #         json.dump(item, f)
    print('***finish processing***total %d samples*** ' %len(newdata))
    # print(count)

    

def demo():
    data=processjson('ami_trainseg')
    print(len(data))
    # scorelist = pickle.load(open('entail_ami_train.pkl','rb'))
    # start=0
    # length=120
    # plt.bar(range(length),scorelist[start:start+length])
    # plt.show()

if __name__=='__main__':
    entailrank()