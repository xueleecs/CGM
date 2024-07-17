from transformers import BertModel, BertTokenizer
import re, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception

def pandding_J(padded_protein,maxlen):

    L= len(padded_protein)
    if L<maxlen:
        for j in range(L,maxlen):
            padded_protein+='J'
    else:
        padded_protein=padded_protein[0:maxlen]
                
    return padded_protein  

def get_bert(protein):

    tokenizer=BertTokenizer.from_pretrained('/prot_bert_bfd', do_lower_case=False)
    Bert_model= BertModel.from_pretrained("/prot_bert_bfd")
    

    protein_seq = dict()
    block1=nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()

    )
    i=0

    for p_key in protein.keys():
        i=i+1
        # print(i)
        maxlen=1400
        input_seq=protein[p_key]
        input_seq=pandding_J(input_seq,maxlen)
        input_seq = ' '.join(input_seq)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        protein_seq =  tokenizer(input_seq, return_tensors='pt')
        proteinfeature = Bert_model(**protein_seq)
        proteinfeature1 = proteinfeature[0]
        proteinfeature2 = proteinfeature1.view(-1, 1024)
        proteinfeature3 = block1(proteinfeature2)

        proteinfeature4 = proteinfeature3.flatten()
        protein[p_key] = proteinfeature4.detach().numpy()

    return protein
