import torch
from torch.utils.data import Dataset,DataLoader,Subset
from utils import load_pickle_file
from utils import _read_knowledge_graph_dialkg,_make_dgl_graph,_find_entity_path,_find_relation_entity_path
from collections import defaultdict
import dgl
from dgl.sampling import sample_neighbors
from copy import deepcopy

import random
import pandas as pd
# from association_rels import predict_rel
import itertools
import time
from association_rels_exact import predict_rel

from tqdm import tqdm

class MultiReDataset(Dataset):
    def __init__(self, opt,transform):
        self.transform = transform
        self.dataset = load_pickle_file(opt['dialog_samples'])

        self.relation2relationID = load_pickle_file(opt['relation2relationID'])

        self.relation_embeddings = load_pickle_file(opt["relation_embeddings"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        process_samples = []
        if self.transform:
            dialog_samples = self.dataset[idx]

            dialogID = dialog_samples['dialogID']
            samples_flag = dialog_samples['samples_flag']

            flag = samples_flag['flag']
            samples = samples_flag['samples']
            state = samples_flag['state']

            for sample in samples:
                sample['flag'],sample['state'] = flag,state
                process_samples.append(self.transform(sample))

        return dialogID,process_samples,flag,state


class ToTensor(object):
    def __init__(self, opt):
        self.relation2relationID = load_pickle_file(opt['relation2relationID'])




    def __call__(self, sample):

        encoder_utterances = sample['utterances']
        true_paths = sample['paths']

        paths_history = sample['paths history'] if 'paths history' in sample.keys() else []

        extra_dialogue_representation = torch.unsqueeze(sample['extra_utterances'],0) if 'extra_utterances' in sample.keys() else None

        dialogue_representation = torch.unsqueeze(encoder_utterances,0)
  
        try:
            rels_history = [self.relation2relationID[path[1]] for path in paths_history[-1]] if len(paths_history) else []
            true_rel = [self.relation2relationID[path[1]] for path in true_paths]
        except:
            return {}

 
        return {
                'dialogue_representation': dialogue_representation,
                'extra_dialogue_representation':extra_dialogue_representation,
                'rels_history':rels_history,
                'true_rel':true_rel
            }


def MultiRe_collate(batch): #取数据时进行堆叠
    #print('collate',batch)

    batch_datas =[]

    for data in batch:
        
        dialogID,process_samples,flag,state = data[0],data[1],data[2],data[3]

        one_dialogue_representation = {}
        one_extra_dialogue_representation = {}
        one_rels_history = {}
        one_true_rel = {}


        count = 0
        for i,sample in enumerate(process_samples):        
            if not len(sample):
                continue 
            
            one_dialogue_representation[count] = {'dialogue_representation':sample['dialogue_representation']}
            one_extra_dialogue_representation[count] = {'extra_dialogue_representation':sample['extra_dialogue_representation']}
            one_rels_history[count] = {'rels_history':sample['rels_history']}
            one_true_rel[count] = {'true_rel':sample['true_rel']}

            count += 1

        one_data = {
            'one_dialogue_representation': one_dialogue_representation,
            'one_extra_dialogue_representation':one_extra_dialogue_representation,
            'one_rels_history':one_rels_history,
            'one_true_rel':one_true_rel,

            'flag': flag,
            'state': state,
            'sample_num':count,
        }


        batch_datas.append(one_data)
        
    return batch_datas



