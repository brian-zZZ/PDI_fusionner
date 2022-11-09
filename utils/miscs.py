"""A implementary utils for DTI model"""
import random
import torch
import numpy as np

def create_sent(seq_list, seg_len=1):
    # 1-gram
    sent_list = []
    for s in seq_list:
        tmp = []
        for i in range(len(s) - seg_len + 1):
            tmp.append(s[i: i + seg_len])
        sent_list.append(tmp)
    return sent_list

def tokenize(sent_list, vocab, max_seq_len):
    t_seq_list = []
    for sent in sent_list:
        tmp = [vocab['[CLS]']]
        for word in sent:
            tmp.append(vocab[word])
            if len(tmp) == max_seq_len - 1:
                break
        tmp.append(vocab['[SEP]'])
        if len(tmp) < max_seq_len:
            for i in range(max_seq_len - len(tmp)):
                tmp.append(vocab['[PAD]'])
        t_seq_list.append(tmp)
    t_seq_list = np.array(t_seq_list)
    return torch.from_numpy(t_seq_list).long()

def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path) as f:
        for i, line in enumerate(f.readlines()):
            vocab[line.strip()] = i
    return vocab
