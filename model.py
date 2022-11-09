import torch
import torch.nn as nn
from ProModels.ProGAT import ProGAT
from ProModels.ProLSTM import BiLSTM
from ProModels.ProBert import BERT
from AttentiveFP import Fingerprint
from ProModels.Fusion import ProDrugCrossFusion


class ProModelFactory(nn.Module):
    def __init__(self, radius, T, input_feature_dim, input_bond_dim,
                 fingerprint_dim, p_dropout, pro_seq_dim, pro_gat_dim, model_type,
                 fusion_n_layers, fusion_n_heads, fusion_dropout, args):
        super(ProModelFactory, self).__init__()
        self.model_type = model_type
        if model_type == "wo_seq":
            pro_seq_dim = 0
        elif model_type == "wo_struc":
            pro_gat_dim = 0
        self.pro_concat_dim = pro_seq_dim + pro_gat_dim
        self.pro_final_dim = int(self.pro_concat_dim / 2) # half of the concatenated dim
        self.Pro = ProNetwork(self.pro_concat_dim, self.pro_final_dim, pro_seq_dim, pro_gat_dim, model_type, 
                              radius=radius, T=T, p_dropout=p_dropout, args=args)

        if model_type != "wo_drug": # with drug molecule
            self.GAT = Fingerprint(radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, p_dropout)
            self.CrossFusion = ProDrugCrossFusion(n_layers=fusion_n_layers, n_heads=fusion_n_heads,
                                            pro_hid_dim=self.pro_final_dim, drug_hid_dim=fingerprint_dim, dropout=fusion_dropout)
            self.predict_n = nn.Sequential(nn.Dropout(p_dropout),
                                           nn.Linear(self.pro_final_dim, 1))
        else:
            self.predict_n_ = nn.Sequential(nn.Dropout(p_dropout),
                                            nn.Linear(self.pro_final_dim, 1))
        
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, tokenized_sent, amino_list,
                amino_degree_list, amino_mask):
        # Protein embedding
        pro_feature = self.Pro(tokenized_sent, amino_list, amino_degree_list, amino_mask)

        # Drug embedding
        if self.model_type != 'wo_drug':
            smile_feature = self.GAT(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
            fusion_pro_feature = self.CrossFusion(pro_feature.unsqueeze(1), smile_feature.unsqueeze(1))
            prediction = self.predict_n(fusion_pro_feature.squeeze(1))
            
            return prediction
        else:
            return self.predict_n_(pro_feature)


class ProModelFactory_fc(nn.Module):
    def __init__(self, radius, T, input_feature_dim, input_bond_dim, \
                 fingerprint_dim, p_dropout, pro_seq_dim, pro_gat_dim, model_type, args):
        super(ProModelFactory_fc, self).__init__()
        self.model_type = model_type
        if model_type == "wo_seq":
            pro_seq_dim = 0
        elif model_type == "wo_struc":
            pro_gat_dim = 0
        self.pro_concat_dim = pro_seq_dim + pro_gat_dim
        self.pro_final_dim = int(self.pro_concat_dim / 2) # half of the concatenated dim
        self.Pro = ProNetwork(self.pro_concat_dim, self.pro_final_dim, pro_seq_dim, pro_gat_dim, model_type, 
                              radius=radius, T=T, p_dropout=p_dropout, args=args)

        if model_type != "wo_drug": # with drug molecule
            self.GAT = Fingerprint(radius, T, input_feature_dim, input_bond_dim, fingerprint_dim, p_dropout)
            self.predict_n = nn.Sequential(nn.Dropout(p_dropout),
                                           nn.Linear(fingerprint_dim + self.pro_final_dim, 1))
        else:
            self.predict_n_ = nn.Sequential(nn.Dropout(p_dropout),
                                            nn.Linear(self.pro_final_dim, 1))
        
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, tokenized_sent, amino_list,
                amino_degree_list, amino_mask):
        # Protein embedding
        pro_feature = self.Pro(tokenized_sent, amino_list, amino_degree_list, amino_mask)

        # Drug embedding
        if self.model_type != 'wo_drug':
            smile_feature = self.GAT(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
            con_feature = torch.cat((smile_feature, pro_feature), dim=1)
            prediction = self.predict_n(con_feature)
            return prediction
        else:
            return self.predict_n_(pro_feature)


class ProNetwork(nn.Module):
    def __init__(self, pro_input_size, pro_out_size, seq_size, gat_size, model_type, radius=2, T=1, p_dropout=0.3, args=None):
        super(ProNetwork, self).__init__()
        if model_type != 'wo_seq': # with seq
            if model_type == "bilstm":
                self.SeqEmb = BiLSTM(rnn_layers=1, keep_dropout=p_dropout) # UserWarning: dropout>0 works when num_layers>1
            elif model_type == "wo_pt":
                self.SeqEmb = BERT(args, pt=False)
            else:
                self.SeqEmb = BERT(args, pt=True)
            self.seq_layernorm = nn.LayerNorm(seq_size)
        if model_type != 'wo_struc': # with struc
            self.GatEmb = ProGAT(embedding_size=gat_size, radius=radius, T=T, p_dropout=p_dropout)
            self.gat_layernorm = nn.LayerNorm(gat_size)
        
        self.model_type = model_type
        self.predict_n = nn.Sequential(nn.Dropout(p_dropout),
                                       nn.Linear(pro_input_size, pro_out_size))

    def forward(self, tokenized_sent, amino_list, amino_degree_list, amino_mask):
        # Forword pass to generate seq / struc embedding
        if self.model_type != 'wo_seq': # with seq
            if self.model_type == 'bilstm':
                seq_bert_feature = self.SeqEmb(tokenized_sent)
            elif self.model_type == 'wo_ft':
                self.SeqEmb.eval()
                with torch.no_grad():
                    seq_bert_feature = self.SeqEmb(tokenized_sent)[1]
            else:
                seq_bert_feature = self.SeqEmb(tokenized_sent)[1]
        if self.model_type != 'wo_struc': # with struc
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)

        # Concate corresponding embeddings
        if self.model_type == 'wo_struc': 
            seq_feature = self.seq_layernorm(seq_bert_feature)
        elif self.model_type == "wo_seq":
            seq_feature = self.gat_layernorm(seq_gat_feature)
        else:
            seq_bert_feature = self.seq_layernorm(seq_bert_feature)
            seq_gat_feature = self.gat_layernorm(seq_gat_feature)
            seq_feature = torch.cat([seq_bert_feature, seq_gat_feature], dim=1)
        
        # Linear projection into specific dim
        seq_feature = self.predict_n(seq_feature)

        return seq_feature
    