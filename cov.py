from utils import *
from anndata import AnnData
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from transformers import AdamW
from transformers import get_scheduler

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

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    return input_ids, attention_mask


def get_batches_for_classification(seqs, target, tokenizer, args):

    input_ids, attention_mask = encode_for_classification(seqs, tokenizer)

    labels, unique_labels = get_labels(seqs, target)
    indices = [unique_labels.index(l) for l in labels] # class_name to class_index
    y = torch.tensor(list(indices), dtype=torch.long) # list to tensor

    tensor_dataset = TensorDataset(input_ids, attention_mask, y)
    tensor_dataloader = DataLoader(tensor_dataset, batch_size=args.minibatch_size, shuffle=False)
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
        for i in range(0, len(words), int(1/args.masking_prob)):
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

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)

    return input_ids, attention_masks, labels


def get_batches_for_masked_lm(seqs, tokenizer, max_seq_len, args):

    fnames = ['cache/input_ids.pt', 'cache/attention_mask.pt', 'cache/labels.pt']

    if args.use_cache and (os.path.exists(fnames[0]) and
        os.path.exists(fnames[1]) and os.path.exists(fnames[2])):
        tprint('Loading input_ids, attention_mask, and labels...')
        input_ids = torch.load(fnames[0])
        attention_mask = torch.load(fnames[1])
        labels = torch.load(fnames[2])
    else:
        input_ids, attention_mask, labels = encode_for_masked_lm(seqs, tokenizer, max_seq_len)
        if args.use_cache:
            mkdir_p('cache')
            tprint('Saving input_ids, attention_mask, and labels...')
            torch.save(input_ids, fnames[0])
            torch.save(attention_mask, fnames[1])
            torch.save(labels, fnames[2])

    tprint('input_ids:        {}     Type:   {}'.format(input_ids.size(), input_ids.type()))
    tprint('attention_mask:   {}     Type:   {}'.format(attention_mask.size(), attention_mask.type()))
    tprint('labels:           {}     Type:   {}'.format(labels.size(), labels.type()))

    tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
    tensor_dataloader = DataLoader(tensor_dataset, batch_size=args.minibatch_size, shuffle=False)
    return tensor_dataloader


def embed_seqs(model, seqs, target, device, args):

    embed_fname = 'cache/embeddings.npy'

    if args.use_cache and os.path.exists(embed_fname):
        mkdir_p('cache')
        tprint('Loading X_embed.npy')
        X_embed = np.load(embed_fname, allow_pickle=True)
    else:
        model.eval()
        batches = get_batches_for_classification(seqs, target, tokenizer, args)

        X_embed = []
        for batch_id, batch_cpu in enumerate(batches):
            tprint('{}/{}'.format(batch_id*args.minibatch_size, len(seqs)))
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

        if args.use_cache:
            mkdir_p('cache')
            tprint('Saving X_embed.npy')
            np.save(embed_fname, X_embed)

    for seq_idx, seq in enumerate(seqs):
        for meta in seqs[seq]:
            meta['embedding'] = X_embed[seq_idx]

    return seqs


def analyze_embedding(model, seqs, target, device, args):

    seqs = embed_seqs(model, seqs, target, device, args)

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


def get_batch_memory(batch):
    return (batch[0].nelement() * batch[0].element_size() +
            batch[1].nelement() * batch[1].element_size() +
            batch[2].nelement() * batch[2].element_size()) / 1024**2


def train(seqs, tokenizer, model, device, max_seq_len, optimizer, batch_id, args):

    batches = get_batches_for_masked_lm(seqs, tokenizer, max_seq_len, args)

    batch0 = next(iter(batches))
    tprint('Num of minibatches: {}'.format(len(batches)))
    tprint('Minibatch shape:    {}'.format(torch.stack(batch0).shape))
    tprint('Minibatch memory:   {:.2f} MB'.format(get_batch_memory(batch0)))

    n_steps = args.epochs * len(batches)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=n_steps)
    model.train()

    for epoch in range(1, args.epochs+1):
        flat_preds, flat_labels, epoch_logits, epoch_labels = [], [], [], []
        total_loss = 0.0
        tprint('Epoch: {}'.format(epoch))
        for batch_cpu in tqdm(batches):
            batch_gpu = (t.to(device) for t in batch_cpu)
            input_ids_gpu, attention_mask, labels = batch_gpu

            outputs = model(input_ids=input_ids_gpu, attention_mask=attention_mask, labels=labels)
            epoch_logits.append(outputs.logits.cpu())
            epoch_labels.append(labels.cpu())
            total_loss += outputs.loss.mean().item()

            # To calculate accuracy
            mask_token_index = (input_ids_gpu == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
            predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1).item()
            true_token_id = labels[0, mask_token_index].item()
            flat_preds.append(predicted_token_id)
            flat_labels.append(true_token_id)

            outputs.loss.mean().backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        epoch_logits = torch.cat(epoch_logits, dim=0)
        epoch_labels = torch.cat(epoch_labels, dim=0)

        tprint('Logits:   {}'.format(epoch_logits.size()))
        tprint('Labels:   {}'.format(epoch_labels.size()))

        cre = F.cross_entropy(epoch_logits.view(-1, tokenizer.vocab_size), epoch_labels.view(-1))
        perplexity = torch.exp(cre).item()
        cre = cre.item()
        mlm = total_loss/len(batches)
        acc = accuracy_score(flat_labels, flat_preds)

        tprint("MLM Loss:        {:.5f}".format(mlm))
        tprint("Cross Entropy:   {:.5f}".format(cre))
        tprint("Perplexity:      {:.5f}".format(perplexity))
        tprint("Accuracy:        {:.5f}".format(acc))

        torch.save({
            'epoch': batch_id*epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc,
            'perplexity': perplexity,
            'cre': cre
            }, 'models/checkpoint_{}.pt'.format(batch_id*epoch))

    return mlm, cre, perplexity, acc


def evaluate(seqs, tokenizer, model, device, max_seq_len, args):

    batches = get_batches_for_masked_lm(seqs, tokenizer, max_seq_len, args)
    flat_preds, flat_labels, all_logits, all_labels = [], [], [], []
    total_loss = 0.0

    model.eval()
    batch0 = next(iter(batches))
    tprint('Num of minibatches: {}'.format(len(batches)))
    tprint('Minibatch shape:    {}'.format(torch.stack(batch0).shape))
    tprint('Minibatch memory:   {:.2f} MB'.format(get_batch_memory(batch0)))
    for batch_cpu in tqdm(batches):
        batch_gpu = (t.to(device) for t in batch_cpu)
        input_ids_gpu, attention_mask, labels = batch_gpu
        with torch.no_grad():
            outputs = model(input_ids=input_ids_gpu, attention_mask=attention_mask, labels=labels)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += outputs.loss.mean().item()

            # To calculate accuracy
            mask_token_index = (input_ids_gpu == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
            predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1).item()
            true_token_id = labels[0, mask_token_index].item()
            flat_preds.append(predicted_token_id)
            flat_labels.append(true_token_id)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

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
    if args.dummy > 0:
        seqs, max_seq_len = generate_dummy_seqs(n = args.dummy) # n random seqs with length [100, 110) just for testing
    else:
        seqs, max_seq_len = read_seqs()

    if args.embed:
        labels, unique_labels = get_labels(seqs, 'group')
        tprint('num_labels = {}'.format(len(unique_labels)))

    train_seqs, test_seqs = split_seqs(seqs)
    tprint('{} train seqs, {} test seqs.'.format(len(train_seqs), len(test_seqs)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tprint('Running on {}. Detected {} GPUs.'.format(device, torch.cuda.device_count()))

    if args.train:
        seqs = train_seqs
    elif args.test:
        seqs = test_seqs

    if args.embed:
        config = BertConfig.from_pretrained("Rostlab/prot_bert", output_hidden_states=True, num_labels=len(unique_labels))
        model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", config=config)
        analyze_embedding(model, seqs, 'group', device, args)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
        optimizer = AdamW(model.parameters(), lr=1e-5)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    if args.batch_size > 0:
        batches = batch_seqs(seqs, batch_size=args.batch_size) # run on batches of sequences to save memory
        tprint('Batch lengths: {}'.format([len(batch) for batch in batches]))
        total_mlm, total_cre, total_perplexity, total_acc = 0.0, 0.0, 0.0, 0.0
        for batch_id, batch in enumerate(batches):
            tprint('Batch {} of {}...'.format(batch_id+1, len(batches)))
            if args.train:
                if batch_id > 0:
                    tprint('Loading models/checkpoint_{}.pt to continue training on...'.format(batch_id * args.epochs))
                    checkpoint = torch.load('models/checkpoint_{}.pt'.format(batch_id * args.epochs))
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                    acc = checkpoint['acc']
                    perplexity = checkpoint['perplexity']
                    cre = checkpoint['cre']
                    tprint('Loaded model epoch={}, acc={}, perplexity={}, cre={}'.format(epoch, acc, perplexity, cre))
                mlm, cre, perplexity, acc = train(batch, tokenizer, model, device, max_seq_len, optimizer, batch_id+1, args)
            elif args.test:
                if args.checkpoint != None and os.path.exists(args.checkpoint):
                    tprint('Loading {} to test on...'.format(args.checkpoint))
                    checkpoint = torch.load(args.checkpoint)
                    model.load_state_dict(checkpoint['model_state_dict'])
                mlm, cre, perplexity, acc = evaluate(batch, tokenizer, model, device, max_seq_len, args)
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
        if args.train:
            batch_id = 1
            if args.checkpoint != None and os.path.exists(args.checkpoint):
                tprint('Loading {} to continue training on...'.format(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                acc = checkpoint['acc']
                perplexity = checkpoint['perplexity']
                cre = checkpoint['cre']
                tprint('Loaded model epoch={}, acc={}, perplexity={}, cre={}'.format(epoch, acc, perplexity, cre))
                batch_id = epoch + 1
            mlm, cre, perplexity, acc = train(seqs, tokenizer, model, device, max_seq_len, optimizer, batch_id, args)
        elif args.test:
            if args.checkpoint != None and os.path.exists(args.checkpoint):
                    tprint('Loading {} to test on...'.format(args.checkpoint))
                    checkpoint = torch.load(args.checkpoint)
                    model.load_state_dict(checkpoint['model_state_dict'])
            mlm, cre, perplexity, acc = evaluate(seqs, tokenizer, model, device, max_seq_len, args)
