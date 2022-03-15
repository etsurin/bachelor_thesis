from divide_text import *
from text import *
from tqdm import tqdm
from tester import rouge
import os
import json



# def list2dict(sentence_list):
#     sentence_dict=dict()
#     for index, sentence in enumerate(sentence_list):
#         sentence_dict[str(index)]=sentence
#     return sentence_dict

# def dict2txt(dict):
#     sortedlist = sorted(dict.items(), key=lambda item:int(item[0]))
#     txt = ''
#     for snippet in sortedlist:
#         txt=txt+snippet[1]+' '
#     return txt
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path)

def match(text,target_list):
    '''
    I used index for the greedy search, it would be easier when concatenating
    '''
    sublist=list()
    tmp_sublist=list()
    type='rouge1'
    while(1):
        tmp_sublist=sublist
        for index,item in enumerate(target_list):
            if index in sublist:
                continue
            tmp_txt = list2txt(target_list,tmp_sublist)
            txt=list2txt(target_list,sublist)+item
            if rouge(text,txt,[type])[type]>rouge(text,tmp_txt,[type])[type]:
                copysublist=sublist
                copysublist.append(index)
                tmp_sublist=copysublist
        if tmp_sublist==sublist:
            break
        else:
            sublist=tmp_sublist
    return list2txt(target_list,sublist)

def generate_new_training_sample(text, summary,duplication=False):
    '''
    '''
    text_list = get_snippets(text,'length')
    result = list()
    summlist=divide_sentences(summary)
    for snippet in text_list:
        if len(tokenizer.encode(snippet))<150:
            continue
        tmp=dict()
        tmp['description']=snippet
        if duplication:
            tmp['abstract']=summary
        else:
            tmp['abstract']=list2txt(match(snippet,summlist))
        if len(tmp['abstract']):
            result.append(tmp)
        else:
            pass
    return result

def debug():        
    data = processjson('my_test')
    testitem = data[331]

    summ=testitem['abstract']+'My favorite singer is Nakamori Akina, who was born in July 13th, 1965. And she led the fashion trend in the 1980s in Japan. '
    # mkdir('./test1')
    # mkdir('./test2')
    # mkdir('./test3')


    # for index,text in enumerate(text_list):
    #     file = open('./test1/summ.'+str(index)+'.txt','w',encoding='utf-8')
    #     file.write(text)
    #     file = open('./test2/summ.'+str(index)+'.txt','w',encoding='utf-8')
    #     file.write(testitem['abstract'])
    #     file = open('./test3/summ.'+str(index)+'.txt','w',encoding='utf-8')
    #     file.write(summ)

    # for text in text_list:
    #     print(rouge(text,testitem['abstract']),rouge(text,summ))    
    # print(generate_new_training_sample(testitem['description'],testitem['abstract']))

def main():
    newdata=list()
    data = processjson('ami_train')
    print(len(data))
    info =list()
    for index,item in tqdm(enumerate(data)):
        text=item['description']
        summ=item['abstract']
        newsamples=generate_new_training_sample(text,summ,duplication=False)
        info.append(len(newsamples))
        for sample in newsamples:
            newdata.append(sample)
    print('***finish processing***total %d samples***' %len(newdata))
    with open('ami_trainseg.json', 'w') as f:
        for item in newdata:
            json.dump(item, f)
    pickle.dump(info, open('ami_info_train.pkl', 'wb'))

if __name__=='__main__':
    main()
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp.add_function(match)
    # lp.add_function(generate_new_training_sample)
    # lp_wrapper = lp(main)
    # lp_wrapper()
    # lp.print_stats()    


        
        

