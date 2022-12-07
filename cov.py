from utils import *
from anndata import AnnData
from transformers import BertConfig, BertModel, BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_labels(seqs, target):
    labels = []
    for i in range(len(seqs)):
        label = list(seqs.items())[i][1][0][target]
        labels.append(label)

    unique_labels = list(dict.fromkeys(labels)) # remove duplicates
    return labels, unique_labels


def encode(seqs, tokenizer):
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

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


def get_batches(seqs, target, tokenizer, batch_size):

    input_ids, attention_mask = encode(seqs, tokenizer)

    labels, unique_labels = get_labels(seqs, target)
    indices = [unique_labels.index(l) for l in labels] # class_name to class_index
    y = torch.tensor(list(indices), dtype=torch.long) # list to tensor
    tprint('input_ids:      {}'.format(input_ids.size()))
    tprint('attention_mask: {}'.format(attention_mask.size()))
    tprint('labels:         {}'.format(y.size()))
    tensor_dataset = TensorDataset(input_ids, attention_mask, y)
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
        batches = get_batches(seqs, target, tokenizer, batch_size=args.batch_size)
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


if __name__ == '__main__':
    args = parse_args()

    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }
    seqs = get_seqs()
    _, unique_labels = get_labels(seqs, 'group')
    tprint('num_labels = {}'.format(len(unique_labels)))
    # seqs = sample_seqs(seqs) # random sampling 1% of the data
    # train_seqs, test_seqs = split_seqs(seqs)
    # print('{} train seqs, {} test seqs.'.format(len(train_seqs), len(test_seqs)))

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    config = BertConfig.from_pretrained("Rostlab/prot_bert", output_hidden_states=True, num_labels=len(unique_labels))
    # model = BertModel.from_pretrained("Rostlab/prot_bert", config=config)
    model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", config=config)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    tprint('Running on {}'.format(device))
    model.to(device)
    analyze_embedding(args, model, seqs, target='group', device=device, use_cache=True)
