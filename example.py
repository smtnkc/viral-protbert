from transformers import BertConfig, BertTokenizer, BertForMaskedLM
import torch

config = BertConfig.from_pretrained("Rostlab/prot_bert", output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert", config=config)

masked_sent = "S N L T [MASK] R T Q L"
unmasked_sent = "S N L T T R T Q L"
max_len = len(unmasked_sent) + 3 # an arbitrary max len

inputs = tokenizer(masked_sent, return_tensors="pt", max_length=max_len+2, padding='max_length')
u_inputs = tokenizer(unmasked_sent, return_tensors="pt", max_length=max_len+2, padding='max_length')

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print('Predicted token ID:', tokenizer.decode(predicted_token_id))


input_ids = inputs['input_ids']
attention_mask = inputs["attention_mask"]
u_input_ids = u_inputs["input_ids"]

# mask labels of non-[MASK] tokens
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, u_input_ids, -100)
outputs = model(**inputs, labels=labels)

print(input_ids)
print(u_input_ids)
print(attention_mask)
print(labels)

print(round(outputs.loss.item(), 2))
