from utils import *

if __name__ == '__main__':
    args = parse_args()

    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }
    seqs = get_seqs(args)
    train_seqs, test_seqs = split_seqs(seqs)
    print('{} train seqs, {} test seqs.'.format(len(train_seqs), len(test_seqs)))
