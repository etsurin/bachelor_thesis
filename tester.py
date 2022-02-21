from tqdm import tqdm
import nltk  
from generator import *
from datasets import load_metric


compute_metric = load_metric("rouge")


def rouge(preds, references,rouge_types=None):
    '''
    preds and references can be both str or list(with strings in)
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
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

def run_and_test(test_only = True):
    text_num = 1914
    preds = list()
    references = list()
    if test_only:
        result_file = open('generated_predictions.txt','r',encoding='utf-8').readlines()
        datasetname='my_test'
        preds = result_file
        print(len(preds))
        for index in range(text_num):
            references.append(get_demo(datasetname,index)['abstract'])
    else:
        result_file = open('generated_results.txt','w', encoding='utf-8')
        datasetname='my_test'
        testdata = processjson(datasetname)
        for index in tqdm(range(text_num)):
            textitem = testdata[index]
            divide_list = get_divide_index(textitem['description'],metric='length')
            pred = generate(textitem['description'],divide_list).strip()
            result_file.write(pred+"\n")
            reference = textitem['abstract']
            preds.append(pred)
            references.append(reference)
    result = rouge(preds,references)
    return result
