from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")

inputs = tokenizer("S N L T [MASK] R T Q L", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
#mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

#predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
#print(tokenizer.decode(predicted_token_id))
print(inputs['input_ids'])
labels = tokenizer("S N L T T R T Q L", return_tensors="pt")["input_ids"]
print(labels)
# mask labels of non-[MASK] tokens
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
print(labels)
outputs = model(**inputs, labels=labels)
print(round(outputs.loss.item(), 2))
