# transformers == 2.7.0
import torch
from transformers import AutoModel, AutoConfig

def BERT(args, pt=False):
    config = AutoConfig.from_pretrained(args.bert_config)
    BertModel = AutoModel.from_config(config)
    if pt: # with pre-training
        # Load the pretrained model weights
        pt_model_weights = torch.load(args.pretrained_model_pth)
        BertModel.load_state_dict(pt_model_weights, strict=True)
 
    return BertModel
