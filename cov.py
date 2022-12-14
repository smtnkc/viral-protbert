from utils import *
from anndata import AnnData
from transformers import BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

def get_labels(seqs, target):
    labels = []
    for i in range(len(seqs)):
        label = list(seqs.items())[i][1][0][target]
        labels.append(label)

    unique_labels = list(dict.fromkeys(labels)) # remove duplicates
    return labels, unique_labels


def encode_for_classification(seqs, tokenizer):
    sents = []
    max_seq_len = 0
    for i in range(len(seqs)):
        seq = list(seqs.items())[i][0]
        if len(seq) > max_seq_len:
            max_seq_len = len(seq)
        sent = " ".join(seq)
        sents.append(sent)

    input_ids = []
    attention_mask = []
    for sent in sents:
        tokenized_sent = tokenizer.encode_plus(sent, max_length=max_seq_len+2, padding='max_length', truncation=True)
        input_ids.append(tokenized_sent['input_ids'])
        attention_mask.append(tokenized_sent['attention_mask'])

    input_ids = torch.squeeze(torch.stack(input_ids))
    attention_mask = torch.squeeze(torch.stack(attention_mask))

    return input_ids, attention_mask


def get_batches_for_classification(seqs, target, tokenizer, batch_size):

    input_ids, attention_mask = encode_for_classification(seqs, tokenizer)

    labels, unique_labels = get_labels(seqs, target)
    indices = [unique_labels.index(l) for l in labels] # class_name to class_index
    y = torch.tensor(list(indices), dtype=torch.long) # list to tensor

    tensor_dataset = TensorDataset(input_ids, attention_mask, y)
    tensor_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    return tensor_dataloader


def encode_for_masked_lm(seqs, tokenizer):
    sents = []
    max_seq_len = 0
    for i in range(len(seqs)):
        seq = list(seqs.items())[i][0]
        if len(seq) > max_seq_len:
            max_seq_len = len(seq)
        sent = " ".join(seq)
        sents.append(sent)

    masked_sents = []
    unmasked_sents = []

    tprint('Masking...')
    for s in tqdm(sents):
        words = s.split()
        for i in range(len(words)):
            sent_pre = words[:i]
            sent_post = words[i+1:]
            masked_sent = sent_pre + [tokenizer.mask_token] + sent_post
            masked_sents.append(" ".join(masked_sent))
            unmasked_sents.append(s) # this will be used as true label

    input_ids = []
    attention_mask = []
    labels = []

    tprint('Tokenizing...')
    with tqdm(total=(len(masked_sents))) as pbar:
        for (masked_sent, unmasked_sent) in zip(masked_sents, unmasked_sents):
            t_masked_sent = tokenizer(masked_sent, return_tensors="pt", max_length=max_seq_len+2, padding='max_length')
            t_unmasked_sent = tokenizer(unmasked_sent, return_tensors="pt", max_length=max_seq_len+2, padding='max_length')["input_ids"]
            label = torch.where(t_masked_sent.input_ids == tokenizer.mask_token_id, t_unmasked_sent, -100)
            input_ids.append(t_masked_sent['input_ids'])
            attention_mask.append(t_masked_sent['attention_mask'])
            labels.append(label)
            pbar.update(1)

    input_ids = torch.squeeze(torch.stack(input_ids))
    attention_mask = torch.squeeze(torch.stack(attention_mask))
    labels = torch.squeeze(torch.stack(labels))

    return input_ids, attention_mask, labels



def get_batches_for_masked_lm(seqs, tokenizer, batch_size, use_cache=True):

    fnames = ['cache/input_ids.pt', 'cache/attention_mask.pt', 'cache/labels.pt']

    if use_cache and (os.path.exists(fnames[0]) and 
        os.path.exists(fnames[1]) and os.path.exists(fnames[2])):
        tprint('Loading input_ids, attention_mask, and labels...')
        input_ids = torch.load(fnames[0])
        attention_mask = torch.load(fnames[1])
        labels = torch.load(fnames[2])
    else:    
        input_ids, attention_mask, labels = encode_for_masked_lm(seqs, tokenizer)
        if use_cache:
            tprint('Saving input_ids, attention_mask, and labels...')
            torch.save(input_ids, fnames[0])
            torch.save(attention_mask, fnames[1])
            torch.save(labels, fnames[2])

    tprint('input_ids:        {}     Type:   {}'.format(input_ids.size(), input_ids.type()))
    tprint('attention_mask:   {}     Type:   {}'.format(attention_mask.size(), attention_mask.type()))
    tprint('labels:           {}     Type:   {}'.format(labels.size(), labels.type()))

    tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
    tensor_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    return tensor_dataloader


def embed_seqs(args, model, seqs, target, device, use_cache):

    mkdir_p('cache')
    embed_fname = 'cache/embeddings.npy'

    if use_cache and os.path.exists(embed_fname):
        tprint('Loading X_embed.npy')
        X_embed = np.load(embed_fname, allow_pickle=True)
    else:
        model.eval()
        batches = get_batches_for_classification(seqs, target, tokenizer,
                    batch_size=args.batch_size, use_cache=use_cache)

        X_embed = []
        for batch_id, batch_cpu in enumerate(batches):
            tprint('{}/{}'.format(batch_id*args.batch_size, len(seqs)))
            input_ids, attention_mask, labels = (t.to(device) for t in batch_cpu)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states

                # Extract the embeddings (vector representations) from 
                # the last hidden state as suggested in ProtBert paper:
                token_embeddings = hidden_states[-1][0] 

                tprint(token_embeddings.size())

                # Move each embedding to CPU to avoid GPU memory leak
                token_embeddings = token_embeddings.detach().cpu().numpy() 
                X_embed.append(token_embeddings)

        if use_cache:
            tprint('Saving X_embed.npy')
            np.save(embed_fname, X_embed)

    for seq_idx, seq in enumerate(seqs):
        for meta in seqs[seq]:
            meta['embedding'] = X_embed[seq_idx]

    return seqs


def analyze_embedding(args, model, seqs, target, device, use_cache):

    seqs = embed_seqs(args, model, seqs, target, device, use_cache)

    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    for seq in seqs:
        meta = seqs[seq][0]
        X.append(meta['embedding'].mean(0))
        for key in meta:
            if key == 'embedding':
                continue
            if key not in obs:
                obs[key] = []
            obs[key].append(Counter([
                meta[key] for meta in seqs[seq]
            ]).most_common(1)[0][0])
        obs['n_seq'].append(len(seqs[seq]))
        obs['seq'].append(str(seq))
    X = np.array(X)

    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]

    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)
    sc.tl.umap(adata, min_dist=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata, [ 'host', 'group', 'continent', 'louvain' ])

    interpret_clusters(adata)

    adata_cov2 = adata[(adata.obs['louvain'] == '0') | (adata.obs['louvain'] == '2')]
    plot_umap(adata_cov2, [ 'host', 'group', 'country' ], namespace='cov7')


def evaluate(seqs, tokenizer, model, device, use_cache):

    batches = get_batches_for_masked_lm(seqs, tokenizer, batch_size=1, use_cache=use_cache)
    flat_preds, flat_labels, all_logits, all_labels = [], [], [], []
    total_loss = 0.0

    model.eval()
    tprint('Testing...')
    for batch_cpu in tqdm(batches):
        batch_gpu = (t.to(device) for t in batch_cpu)
        input_ids_gpu, attention_mask, labels = batch_gpu
        with torch.no_grad():
            outputs = model(input_ids=input_ids_gpu, attention_mask=attention_mask, labels=labels)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += outputs.loss.item()

            # To calculate accuracy
            mask_token_index = (input_ids_gpu == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()      
            predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1).item()
            true_token_id = labels[0, mask_token_index].item()
            flat_preds.append(predicted_token_id)
            flat_labels.append(true_token_id)

    all_logits = torch.squeeze(torch.stack(all_logits))
    all_labels = torch.squeeze(torch.stack(all_labels))

    tprint('Logits:   {}'.format(all_logits.size()))
    tprint('Labels:   {}'.format(all_labels.size()))

    cre = F.cross_entropy(all_logits.view(-1, tokenizer.vocab_size), all_labels.view(-1))

    acc = accuracy_score(flat_labels, flat_preds)
    tprint("MLM Loss:        {:.5f}".format(total_loss/len(batches)))
    tprint("Cross Entropy:   {:.5f}".format(cre.item()))
    tprint("Perplexity:      {:.5f}".format(torch.exp(cre).item()))
    tprint("Accuracy:        {:.5f}".format(acc))


if __name__ == '__main__':

    args = parse_args()
    seqs = get_seqs()
    # seqs = get_dummy_seqs()
    # seqs = sample_seqs(seqs, p=1) # random sampling p% of the data

    if args.embed:
        labels, unique_labels = get_labels(seqs, 'group')
        tprint('num_labels = {}'.format(len(unique_labels)))

    train_seqs, test_seqs = split_seqs(seqs)
    tprint('{} train seqs, {} test seqs.'.format(len(train_seqs), len(test_seqs)))

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tprint('Running on {}'.format(device))

    if args.embed:
        config = BertConfig.from_pretrained("Rostlab/prot_bert", output_hidden_states=True, num_labels=len(unique_labels))
        model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", config=config)
        model.to(device)
        analyze_embedding(args, model, seqs, target='group', device=device, use_cache=False)

    if args.test:
        config = BertConfig.from_pretrained("Rostlab/prot_bert", output_hidden_states=True, output_attentions=True)
        model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert", config=config)
        model.to(device)
        evaluate(test_seqs, tokenizer, model, device, use_cache=True)
