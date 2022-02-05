from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from test_text import *
from divide_text import *
import torch
device='cuda' if torch.cuda.is_available() else 'cpu'

def get_model(model_name):
    if model_name=='facebook/bart-large-cnn':
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        return model, tokenizer
    else:
        print('undefined model!')

def generate(text,divide_list=[]):
    model,tokenizer = get_model('facebook/bart-large-cnn')
    if divide_list!=[]:
        text_list=divide_text(text,divide_list)
        summary=''
        for snippet in text_list:
            tmp_summary = generate(snippet)
            summary+=tmp_summary
        return summary
    input=tokenizer(text,max_length=1024,return_tensors='pt')
    input=torch.tensor(input['input_ids']).to(device)
    summary_ids = model.generate(input, num_beams=4, max_length=250, early_stopping=True).cpu()
    summary=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    summary=summary[0]
    return summary


