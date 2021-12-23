from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from test_text import *
from divide_text import *
import torch
device='cuda'
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def generate(text,divide_list=[]):
    if divide_list!=[]:
        _,text_list=divide_text(text,divide_list)
        summary=''
        for snippet in text_list:
            tmp_summary = generate(snippet)
            summary+=tmp_summary
        return summary
    input=tokenizer(text,max_length=1024,return_tensors='pt')
    input=torch.tensor(input['input_ids']).to(device)
    summary_ids = model.generate(input, num_beams=10, max_length=200, early_stopping=True).cpu()
    summary=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    summary=summary[0]
    return summary

summary = generate(text_04,[9,18])
print(summary)
