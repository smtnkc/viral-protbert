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


def encode_for_masked_lm(seqs, tokenizer, max_seq_len):
    sents = []
    for i in range(len(seqs)):
        seq = list(seqs.items())[i][0]
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
    attention_masks = []
    labels = []

    tprint('Tokenizing...')
    with tqdm(total=(len(masked_sents))) as pbar:
        for (masked_sent, unmasked_sent) in zip(masked_sents, unmasked_sents):
            # Note that [PAD] = 0, [CLS] = 2, [SEP] = 3, [MASK] = 4
            inputs = tokenizer(masked_sent, return_tensors="pt", max_length=max_seq_len+2, padding='max_length')     # [2 9 8 4 6 3 0 0]
            u_inputs = tokenizer(unmasked_sent, return_tensors="pt", max_length=max_seq_len+2, padding='max_length') # [2 9 8 7 6 3 0 0]
            label = torch.where(inputs.input_ids == tokenizer.mask_token_id,
                                u_inputs["input_ids"], -100) # [-100 -100 -100 7 -100 -100 -100 -100]
            input_ids.append(inputs['input_ids']) # [[2 9 8 4 6 3 0 0]]
            attention_masks.append(inputs['attention_mask']) # [[1 1 1 1 1 1 0 0]]
            labels.append(label) # [[-100 -100 -100 7 -100 -100]]
            pbar.update(1)

    input_ids = torch.squeeze(torch.stack(input_ids))
    attention_masks = torch.squeeze(torch.stack(attention_masks))
    labels = torch.squeeze(torch.stack(labels))

    return input_ids, attention_masks, labels


def get_batches_for_masked_lm(seqs, tokenizer, max_seq_len, batch_size, use_cache=True):

    fnames = ['cache/input_ids.pt', 'cache/attention_mask.pt', 'cache/labels.pt']

    if use_cache and (os.path.exists(fnames[0]) and 
        os.path.exists(fnames[1]) and os.path.exists(fnames[2])):
        tprint('Loading input_ids, attention_mask, and labels...')
        input_ids = torch.load(fnames[0])
        attention_mask = torch.load(fnames[1])
        labels = torch.load(fnames[2])
    else:    
        input_ids, attention_mask, labels = encode_for_masked_lm(seqs, tokenizer, max_seq_len)
        if use_cache:
            mkdir_p('cache')
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


def embed_seqs(args, model, seqs, target, device, use_cache=True):

    embed_fname = 'cache/embeddings.npy'

    if use_cache and os.path.exists(embed_fname):
        mkdir_p('cache')
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
            mkdir_p('cache')
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


def get_batch_size_mb(batch):
    return (batch[0].nelement() * batch[0].element_size() + 
            batch[1].nelement() * batch[1].element_size() + 
            batch[2].nelement() * batch[2].element_size()) / 1024**2


def evaluate(seqs, tokenizer, model, device, max_seq_len, use_cache):

    batches = get_batches_for_masked_lm(seqs, tokenizer, max_seq_len, batch_size=1, use_cache=use_cache)
    flat_preds, flat_labels, all_logits, all_labels = [], [], [], []
    total_loss = 0.0

    model.eval()
    tprint('{} minibatches. Each {:.2f} MB'.format(len(batches), get_batch_size_mb(next(iter(batches)))))
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
    perplexity = torch.exp(cre).item()
    cre = cre.item()
    mlm = total_loss/len(batches)
    acc = accuracy_score(flat_labels, flat_preds)

    tprint("MLM Loss:        {:.5f}".format(mlm))
    tprint("Cross Entropy:   {:.5f}".format(cre))
    tprint("Perplexity:      {:.5f}".format(perplexity))
    tprint("Accuracy:        {:.5f}".format(acc))

    return mlm, cre, perplexity, acc


if __name__ == '__main__':

    args = parse_args()
    seqs, max_seq_len = read_seqs()
    # seqs = generate_dummy_seqs() # 20 random seqs with length [100, 110) just for testing
    # seqs = random_sample_seqs(seqs, p=1) # random sampling p% of the sequences just for testing

    if args.embed:
        labels, unique_labels = get_labels(seqs, 'group')
        tprint('num_labels = {}'.format(len(unique_labels)))

    train_seqs, test_seqs = split_seqs(seqs)
    tprint('{} train seqs, {} test seqs.'.format(len(train_seqs), len(test_seqs)))

    if args.inference_batch_size > 0:
        test_seqs_batches = batch_seqs(test_seqs, inference_batch_size=args.inference_batch_size) # test on batches of sequences to save memory
        tprint('Batch lengths of test seqs: {}'.format([len(batch) for batch in test_seqs_batches]))

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tprint('Running on {}'.format(device))

    if args.embed:
        config = BertConfig.from_pretrained("Rostlab/prot_bert", output_hidden_states=True, num_labels=len(unique_labels))
        model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", config=config)
        model.to(device)
        analyze_embedding(args, model, seqs, target='group', device=device, use_cache=args.use_cache)

    if args.test:
        model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
        model.to(device)

        if args.inference_batch_size > 0:
            total_mlm, total_cre, total_perplexity, total_acc = 0.0, 0.0, 0.0, 0.0
            for batch_id, test_seqs_batch in enumerate(test_seqs_batches):
                tprint('Inference batch {} of {}...'.format(batch_id+1, len(test_seqs_batches)))
                mlm, cre, perplexity, acc = evaluate(test_seqs_batch, tokenizer, model, device, max_seq_len, use_cache=args.use_cache)
                total_mlm += mlm
                total_cre += cre
                total_perplexity += perplexity
                total_acc += acc
        
                avg_mlm = total_mlm/(batch_id+1)
                avg_cre = total_cre/(batch_id+1)
                avg_perplexity = total_perplexity/(batch_id+1)
                avg_acc = total_acc/(batch_id+1)
                tprint('Average MLM Loss:        {:.5f}'.format(avg_mlm))
                tprint('Average Cross Entropy:   {:.5f}'.format(avg_cre))
                tprint('Average Perplexity:      {:.5f}'.format(avg_perplexity))
                tprint('Average Accuracy:        {:.5f}'.format(avg_acc))
        else:
            mlm, cre, perplexity, acc = evaluate(test_seqs, tokenizer, model, device, max_seq_len, use_cache=args.use_cache)
