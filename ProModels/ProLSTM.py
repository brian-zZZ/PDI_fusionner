import os
import pickle
import torch
import torch.nn as nn

EMB_SIZE = 512
emb_path = "./PPI/drug/GAT/save/embeding_512d_10m.model"

class BiLSTM(nn.Module):
    def __init__(self, vocab_size=28, char_embedding_size=EMB_SIZE, hidden_dims=512, 
                 num_classes=1, rnn_layers=1, keep_dropout=0.2):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.char_embedding_size = char_embedding_size
        self.num_classes = num_classes
        self.keep_dropout = keep_dropout
        self.hidden_dims = hidden_dims
        self.rnn_layers = rnn_layers
        # 初始化字向量
        if os.path.exists(emb_path):
            embedding_pretrained = torch.tensor(pickle.load(open(emb_path, 'rb')).astype('float32'))
            self.char_embeddings = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
            print("using pretrained embedding:")
        else:
            self.char_embeddings = nn.Embedding(self.vocab_size, self.char_embedding_size)
            print("using newly-built embedding")
        # 字向量参与更新
        self.char_embeddings.weight.requires_grad = True
        # attention层
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # 双层lstm
        self.lstm_net = nn.LSTM(self.char_embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.keep_dropout,
                                bidirectional=True, batch_first=True)
        # FC层
        self.fc_out = nn.Sequential(
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, char_id):
        if not hasattr(self, '_flattened'):
            self.lstm_net.flatten_parameters()
            setattr(self, '_flattened', True)

        # input : [batch_size, len_seq, embedding_dim]
        sen_char_input = self.char_embeddings(char_id)
        # output : [batch_size, len_seq, n_hidden * 2]
        # output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm_net(sen_char_input)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        lstm_hidden = torch.mean(final_hidden_state, dim=1)
        final_output = self.fc_out(lstm_hidden)
        return final_output
