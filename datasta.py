from text import *
from transformers import BartTokenizer
from divide_text import *
from tqdm import tqdm
import numpy as np
device = 'cuda'
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


namelist=['train','valid','test']
infotext = list()
infosumm = list()
for name in namelist:
    data=processjson('ami_%s' %name)
    for index, item in tqdm(enumerate(data)):
        len_text = len(tokenizer.encode(item['description']))
        len_summ = len(tokenizer.encode(item['abstract']))
        infotext.append(len_text)
        infosumm.append(len_summ)
    # len_text_s = len(divide_sentences(item['description']))
    # len_summ_s = len(tokenizer.encode(item['abstract']))
    # txt+=str(len_text)+','+str(len_summ)+','+str(len_text_s)+','+str(len_summ_s)+'\n'
    
print(max(infotext),np.mean(infotext))
print(max(infosumm),np.mean(infosumm))
