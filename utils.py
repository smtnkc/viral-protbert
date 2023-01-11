import os
import sys
import errno
import random
import datetime
import warnings
import pickle
from tqdm import tqdm
import numpy as np
import scanpy as sc
from collections import Counter
from dateutil.parser import parse as dparse
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
from Bio import SeqIO

np.random.seed(1)
random.seed(1)
np.set_printoptions(threshold=10000)

AAs = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
    'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
    'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
]

vocab = {
    '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
    'L': 5, 'A': 6, 'G': 7, 'V': 8, 'E': 9, 'S': 10, 'I': 11,
    'K': 12, 'R': 13, 'D': 14, 'T': 15, 'P': 16, 'N': 17, 'Q': 18,
    'F': 19, 'Y': 20, 'M': 21, 'H':22, 'C': 23, 'W': 24, 'X': 25,
    'U': 26, 'B': 27, 'Z':28, 'O': 29
}

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Coronavirus sequence analysis')

    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--minibatch_size', type=int, default=128,
                        help='Batch size for the tokenizer and model.')
    parser.add_argument('--train', action='store_true',
                        help='Train model')
    parser.add_argument('--masking_prob', type=float, default=1,
                        help='Masking probability')
    parser.add_argument('--test', action='store_true',
                        help='Test model')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size for run. Set 0 to disable batching.')
    parser.add_argument('--embed', action='store_true',
                        help='Analyze embeddings')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached data')
    parser.add_argument('--dummy', type=int, default=0,
                        help='Use n random seqs with length [100, 110). Set 0 to disable.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to load')

    args = parser.parse_args()
    return args

def parse_viprbrc(entry):
    fields = entry.split('|')
    if fields[7] == 'NA':
        date = None
    else:
        date = fields[7].split('/')[0]
        date = dparse(date.replace('_', '-'))

    country = fields[9]
    from locations import country2continent
    if country in country2continent:
        continent = country2continent[country]
    else:
        country = 'NA'
        continent = 'NA'

    from mammals import species2group

    meta = {
        'strain': fields[5],
        'host': fields[8],
        'group': species2group[fields[8]],
        'country': country,
        'continent': continent,
        'dataset': 'viprbrc',
    }
    return meta

def parse_nih(entry):
    fields = entry.split('|')

    country = fields[3]
    from locations import country2continent
    if country in country2continent:
        continent = country2continent[country]
    else:
        country = 'NA'
        continent = 'NA'

    meta = {
        'strain': 'SARS-CoV-2',
        'host': 'human',
        'group': 'human',
        'country': country,
        'continent': continent,
        'dataset': 'nih',
    }
    return meta

def parse_gisaid(entry):
    fields = entry.split('|')

    type_id = fields[1].split('/')[1]

    if type_id in { 'bat', 'canine', 'cat', 'env', 'mink',
                    'pangolin', 'tiger' }:
        host = type_id
        country = 'NA'
        continent = 'NA'
    else:
        host = 'human'
        from locations import country2continent
        if type_id in country2continent:
            country = type_id
            continent = country2continent[country]
        else:
            country = 'NA'
            continent = 'NA'

    from mammals import species2group

    meta = {
        'strain': fields[1],
        'host': host,
        'group': species2group[host].lower(),
        'country': country,
        'continent': continent,
        'dataset': 'gisaid',
    }
    return meta

def process(fnames):
    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if len(record.seq) < 1000:
                continue
            if str(record.seq).count('X') > 0:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            if fname == 'data/cov/viprbrc_db.fasta':
                meta = parse_viprbrc(record.description)
            elif fname == 'data/cov/gisaid.fasta':
                meta = parse_gisaid(record.description)
            else:
                meta = parse_nih(record.description)
            meta['accession'] = record.description
            seqs[record.seq].append(meta)

    with open('data/cov/cov_all.fa', 'w') as of:
        for seq in seqs:
            metas = seqs[seq]
            for meta in metas:
                of.write('>{}\n'.format(meta['accession']))
                of.write('{}\n'.format(str(seq)))

    return seqs

def split_seqs(seqs):
    train_seqs, test_seqs = {}, {}

    tprint('Splitting seqs...')
    for idx, seq in enumerate(seqs):
        if idx % 10 < 2:
            test_seqs[seq] = seqs[seq]
        else:
            train_seqs[seq] = seqs[seq]

    return train_seqs, test_seqs

def random_sample_seqs(seqs, p=1):
    sample_seqs = {}

    for idx, seq in enumerate(seqs):
        if idx % (100//p) == 0:
            sample_seqs[seq] = seqs[seq]

    tprint('{} seqs are sampled.'.format(len(sample_seqs)))

    return sample_seqs

def batch_seqs(seqs, batch_size=76):

    n_batches = len(seqs) // batch_size
    if len(seqs) % batch_size > 0:
        n_batches += 1
    batches = [{} for _ in range(n_batches)]

    for idx, seq in enumerate(seqs):
        batch_id = idx // batch_size
        if batch_id == n_batches:
            batch_id = n_batches - 1
        batches[batch_id][seq] = seqs[seq]

    return batches

def read_seqs():
    fnames = [ 'data/cov/sars_cov2_seqs.fa',
               'data/cov/viprbrc_db.fasta',
               'data/cov/gisaid.fasta' ]

    seqs = process(fnames)

    max_seq_len = max([ len(seq) for seq in seqs ])

    tprint('{} unique sequences with the max length of {}.'.format(len(seqs), max_seq_len))
    return seqs, max_seq_len

def generate_dummy_seqs(n):
    dummy = {}

    for i in range(n):
        k = random.randint(100, 110) # k = random length of sequence
        L = random.choices(AAs, k=k) # L = k random AAs
        key = ''.join(L)
        value = 'metata-' + str(i)
        dummy[key] = value

    return dummy, 110

def interpret_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'host', 'country', 'strain' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var])
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

def plot_umap(adata, categories, namespace='cov'):
    for category in categories:
        sc.pl.umap(adata, color=category, save='_{}_{}.png'.format(namespace, category), show=False)
