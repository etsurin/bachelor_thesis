from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
device = 'cuda'
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
input="The capital of Japan is"
target1=' Tokyo'
target2=' Paris'
input_ids = tokenizer.encode(input)
input_ids = torch.tensor(input_ids).to(device)  # [1, seq_len]
print(input_ids)
target1_ids = tokenizer.encode(target1)
target1_ids = torch.tensor(target1_ids).to(device)  # [1, 1]
print(target1_ids)
target2_ids = tokenizer.encode(target2)
target2_ids = torch.tensor(target2_ids).to(device)  # [1, 1]
print(target2_ids)
output = model(input_ids,return_dict=True)[0]
output=output[-1,:]
print(output.shape)
predicted_index = torch.argmax(output)
predicted_text = tokenizer.decode([predicted_index])
print(predicted_index,predicted_text)
loss_fct = torch.nn.CrossEntropyLoss()
print(output.view(-1, output.size(-1)).shape,target1_ids.shape)
loss = loss_fct(output.view(-1, output.size(-1)),target1_ids)

print(loss)