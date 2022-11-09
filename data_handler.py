import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

from AttentiveFP import save_smiles_dicts, get_smiles_array
from utils import create_sent, tokenize, load_vocab


class DataHandler:
    def __init__(self, raw_filename, contact_map_file, args):
        self.args = args
        self.data_df, self.smile_feature_dict = self.load_smile(raw_filename)

        def get_cm_dict(contact_map_file):
            cm_dict = {}
            max_num = 0

            data = pickle.load(open(contact_map_file, 'rb'))  # dataframe(PDB-ID, seqs, contact_map)
            for index, row in data.iterrows():
                seq = row['seqs'][:self.max_len]
                cm = row['contact_map'][0][:self.max_len, :self.max_len]
                cm_dict[seq] = cm
                mn = np.max(np.sum(cm, axis=1))
                if max_num < mn:
                    max_num = mn
            return cm_dict, max_num

        # for protein structure
        self.input_size =  self.args.nonspecial_vocab_size
        self.max_len =  self.args.max_seq_len
        self.enc_lib = np.eye(self.input_size)
        # contact_map_dict: {seq: array(amino_num, amino_num), seq: array(amino_num, amino_num), ...}, eq.{seq: array([[True, False, ...],[True, False, ...],...]), ...}
        self.contact_map_dict, self.max_neighbor_num = get_cm_dict(contact_map_file) 
        #print(self.data_df, self.smile_feature_dict)

    def get_init(self, seq_list):
        mat = []
        for seq in seq_list:
            seq = list(map(lambda ch: ord(ch) - ord('A'), seq[:self.max_len]))
            # enc: array, (max_seq_len, no_special_tokens_vocab_size)
            enc = self.enc_lib[seq]
            if enc.shape[0] < self.max_len:
                enc = np.pad(enc, ((0, self.max_len - enc.shape[0]), (0, 0)), 'constant')
            # print(enc.shape)

            mat.append(enc)
        # mat: [array(max_seq_len, no_special_tokens_vocab_size), array(max_seq_len, no_special_tokens_vocab_size), ...]
        mat = np.stack(mat, 0)
        # mat: array(batch_size, max_seq_len, no_special_tokens_vocab_size)
        mat = mat.astype(np.float32)
        return mat

    def get_degree_list(self, seq_list):
        mat = []
        uu = 0
        for seq in seq_list:
            seq = seq[:self.max_len]
            # contact_map_dict: {seq: array(amino_num, amino_num), seq: array(amino_num, amino_num), ...}, eq.{seq: array([[True, False, ...],[True, False, ...],...]), ...}
            if seq in self.contact_map_dict:
                cm = self.contact_map_dict[seq]
                uu += 1
            else:
                # print('Sequence not found, ', seq)
                cm = np.zeros(self.max_len, self.max_neighbor_num)
            ##
            degree_list = []
            for i in range(len(seq)):
                tmp = np.array(np.where(cm[i] > 0.5)[0])
                tmp = np.pad(tmp, (0, self.max_neighbor_num - tmp.shape[0]), 'constant', constant_values=(-1, -1))
                degree_list.append(tmp)
            ##
            degree_list = np.stack(degree_list, 0)
            degree_list = np.pad(degree_list, ((0, self.max_len - degree_list.shape[0]), (0, 0)), 'constant',
                                 constant_values=(-1, -1))
            mat.append(degree_list)
        mat = np.stack(mat, 0)
        return mat

    def get_amino_mask(self, seq_list):
        mat = []
        for seq in seq_list:
            mask = np.ones(min(len(seq), self.max_len), dtype=int)
            mask = np.pad(mask, (0, self.max_len - len(mask)), 'constant')
            mat.append(mask)
        mat = np.stack(mat, 0)
        # print('mask', mat)
        return mat
        
    def get_pro_structure(self, seq_list):
        # f1 = cal_mem()
        amino_list = self.get_init(seq_list)
        # f2 = cal_mem()
        # print('Get Pro Structure Index {}-{} costs: {}MB'.format('f2', 'f1', round(f1-f2, 4)))
        amino_degree_list = self.get_degree_list(seq_list)
        # f3 = cal_mem()
        # print('Get Pro Structure Index {}-{} costs: {}MB'.format('f2', 'f3', round(f2 - f3, 4)))
        amino_mask = self.get_amino_mask(seq_list)
        # f4 = cal_mem()
        # print('Get Pro Structure Index {}-{} costs: {}MB'.format('f3', 'f4', round(f3 - f4, 4)))

        return amino_list, amino_degree_list, amino_mask

    def load_smile(self, raw_filename):
        # raw_filename : "./PPI/drug/tasks/DTI/pdbbind/pafnucy_total_rdkit-smiles-v1.csv"
        feature_filename = raw_filename.replace('.csv', '.pickle')
        filename = raw_filename.replace('.csv', '')
        # smiles_tasks_df : df : ["unnamed", "PDB-ID", "seq", "SMILES", "rdkit_smiles", "Affinity-Value", "set"]
        smiles_tasks_df = pd.read_csv(raw_filename)  # main file
        # smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
        smilesList = smiles_tasks_df[ self.args.SMILES].values
        print("number of all smiles: ", len(smilesList))
        atom_num_dist = []
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                mol = Chem.MolFromSmiles(smiles)  # input : smiles seqs, output : molecule obeject
                atom_num_dist.append(len(mol.GetAtoms()))  # list : get atoms obeject num from molecule obeject
                remained_smiles.append(smiles)  # list : smiles without transformation error
                canonical_smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))  # canonical smiles without transformation error
            except:
                print("the smile \"%s\" has transformation error in the first test" % smiles)
                pass
        print("number of successfully processed smiles after the first test: ", len(remained_smiles))

        "----------------------the first test----------------------"
        smiles_tasks_df = smiles_tasks_df[smiles_tasks_df[ self.args.SMILES].isin(remained_smiles)]  # df(13464) : include smiles without transformation error
        smiles_tasks_df[ self.args.SMILES] = remained_smiles

        # smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
        smilesList = remained_smiles  # update valid smile

        # feature_dicts(dict) : 
        # {smiles_to_atom_info, smiles_to_atom_mask, smiles_to_atom_neighbors, "smiles_to_bond_info", "smiles_to_bond_neighbors", "smiles_to_rdkit_list"}
        if os.path.isfile(feature_filename):  # get smile feature dict
        # if False:
            feature_dicts = pickle.load(open(feature_filename, "rb"))
        else:
            # smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
            # filename : "./PPI/drug/tasks/DTI/pdbbind/pafnucy_total_rdkit-smiles-v1"
            feature_dicts = save_smiles_dicts(smilesList, filename)
        
        "----------------------the second test----------------------"
        # remained_df : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]) : include smiles without transformation error and second test error, 13435
        remained_df = smiles_tasks_df[smiles_tasks_df[ self.args.SMILES].isin(feature_dicts['smiles_to_atom_mask'].keys())]
        print("number of successfully processed smiles after the second test: ", len(remained_df))

        return remained_df, feature_dicts


class ProteinDataset(Dataset):
    def __init__(self, dataset, data_handler, args, shuffle=False):
        super(ProteinDataset, self).__init__()
        if shuffle:
            dataset = dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        self.dataset = dataset
        self.data_handler = data_handler
        self.args = args
        self.vocab = load_vocab(args.vocab_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_entry = self.dataset.iloc[item]
        smiles_list = [data_entry[self.args.SMILES]] #.values
        pro_seqs = [data_entry.seq] #.values
        y_val = [data_entry[self.args.TASK]] #.values
        y_val = torch.tensor(y_val)

        # Generate seq, struc, drug inputs
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(
                                                            smiles_list, self.data_handler.smile_feature_dict)
        amino_list, amino_degree_list, amino_mask = self.data_handler.get_pro_structure(pro_seqs)
        # Tokenize seq sentence
        sents = create_sent(pro_seqs)
        tokenized_sent = tokenize(sents, self.vocab, self.args.max_seq_len)
        
        # print(tokenized_sent.shape, [e.shape for e in [amino_list, amino_degree_list, amino_mask]])
        return y_val, tokenized_sent, (x_atom, x_bonds, x_atom_index, x_bond_index, x_mask), (amino_list, amino_degree_list, amino_mask)

    
def prepare_data(args):
    data_handler = DataHandler(args.input_file, args.contact_map_file, args)
    train_df = data_handler.data_df[data_handler.data_df["set"].str.contains('train')]
    valid_df = data_handler.data_df[data_handler.data_df["set"].str.contains('valid')]
    test_test_df = data_handler.data_df[data_handler.data_df["set"].str.contains('test')]
    test_casf2013_df = data_handler.data_df[data_handler.data_df["set"].str.contains('casf2013')]
    test_astex_df = data_handler.data_df[data_handler.data_df["set"].str.contains('astex')]
    if args.sampling:
        train_df, valid_df, test_test_df, test_casf2013_df, test_astex_df = \
                train_df.iloc[:500], valid_df.iloc[:200], test_test_df.iloc[:200], test_casf2013_df.iloc[:200], test_astex_df.iloc[:200]
    print("train_df_nums: %d, valid_df_nums: %d, core2016_df_nums: %d, casf2013_df_nums: %d, astex_df_nums: %d" 
        % (len(train_df), len(valid_df), len(test_test_df), len(test_casf2013_df), len(test_astex_df)))

    x_atom, x_bonds, _, _, _, _ = get_smiles_array([data_handler.data_df[args.SMILES][1]], data_handler.smile_feature_dict)
    num_atom_features = x_atom.shape[-1]  # 39
    num_bond_features = x_bonds.shape[-1]  # 10

    train_set = ProteinDataset(train_df, data_handler, args, shuffle=True)
    valid_set = ProteinDataset(valid_df, data_handler, args)
    test_test_set = ProteinDataset(test_test_df, data_handler, args)
    test_casf2013_set = ProteinDataset(test_casf2013_df, data_handler, args)
    test_astex_set = ProteinDataset(test_astex_df, data_handler, args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count()//2)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count()//2)
    test_test_loader = DataLoader(test_test_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count()//2)
    test_casf2013_loader = DataLoader(test_casf2013_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count()//2)
    test_astex_loader = DataLoader(test_astex_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count()//2)
    loader_pack = train_loader, valid_loader, test_test_loader, test_casf2013_loader, test_astex_loader
    
    return loader_pack, num_atom_features, num_bond_features

