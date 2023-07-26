from __future__ import print_function, division
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch
from skimage import io, transform
import numpy as np
import pickle
import csv
import glob
from tqdm import tqdm
from functools import reduce
import argparse

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataset_attnio import AttnIODataset,ToTensor,AttnIO_collate
from AttnIO_build import AttnIOModel
from torch.utils.data import Subset
from dgl.sampling import sample_neighbors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device=torch.device("cuda")

def filter_paths(paths):
    last_entities = set()
    final_paths = []
    for path in paths:
        if path[-1] not in last_entities:
            last_entities.add(path[-1])
            final_paths.append(path)
    return final_paths

def remove_incorrect_paths(paths):
    final_paths = []
    for path in paths:
        if 0 not in path:
            final_paths.append(path)
    return final_paths

def get_actions(last_entity, graph):
    actions = sample_neighbors(g=graph, nodes = last_entity, fanout=-1, edge_dir="out")
    actions = actions.edges()[1]
    return actions


def calculate_metrics(paths, true_path, edges,recall_entity,recall_path):
    K = [1, 3, 5, 10, 25]
    x = edges
    filtered_paths = filter_paths(paths)
    
    entities = [path[-1] for path in filtered_paths]
    recall_entity["counts"] += 1
    recall_path["counts"] += 1
    #print(filtered_paths [0:25])
    x = 0
    for k in K:
        if true_path in filtered_paths[:k]:
            recall_path["counts@"+str(k)] += 1
        if true_path[-1] in entities[:k]:
            x = 1
            recall_entity["counts@"+str(k)] += 1
        if x==0 and k==25:
            p=10

def relation_accuracy(new_relation_path_pool, new_probs_pool, true_relation_path):
    new_probs_pool = [reduce(lambda x, y: x*y, probs) for probs in new_probs_pool]
    probs_relation_entity = zip(new_probs_pool, new_relation_path_pool)
    probs_relation_entity = sorted(probs_relation_entity, key=lambda x:x[0], reverse=True)
    probs_relation_entity = zip(*probs_relation_entity)
    probs_relation_entity = [list(a) for a in probs_relation_entity]
    probs_pool , path_pool = probs_relation_entity[0], probs_relation_entity[1]

    if true_relation_path in path_pool[:1]:
        recall_relation["counts@1"] += 1
    recall_relation["counts"] += 1


def batch_beam_search(model, batch_data, device, topk,recall_entity,recall_path):

    with torch.no_grad():
        model.eval()
        # 读取数据
        for one_batch_data in batch_data:

            state,flag,sample_num = one_batch_data[5],one_batch_data[6],one_batch_data[7]

            for i in range(sample_num):
                
                dialogue_representation, seed_entities, subgraph, paths,sample_mask = one_batch_data[0][i],one_batch_data[1][i],one_batch_data[2][i],one_batch_data[3][i],one_batch_data[4][i]

                if len(seed_entities) == 0:
                    continue
                elif (state == 'test' and flag == 0) and sample_mask == 0:
 
                    continue
                else:
                    dialogue_representation = dialogue_representation.to(device) # [1,768]
                    seed_entities = seed_entities.to(device)
                    subgraph = subgraph.to(device)
                    true_path = paths.to(device)
             
                    updated_subgraph, expilcit_entity_rep = model(dialogue_representation, seed_entities, subgraph, state, flag)
    
                    edges = updated_subgraph.edges()
        
                    path_pool = []  # list of list, size=bs
                    probs_pool = []

                    for hop in range(3):
                        # 两跳内实体
                        if hop==0:
                            #print('seed',seed_entities)
                            k = min(topk[hop], len(seed_entities))

                            probs = updated_subgraph.ndata["a_0"].to("cpu")
                            #print('probs',probs)
                            topk_probs, topk_actions = torch.topk(probs, k=k)
                            for j in range(k):
                                path_pool.append([topk_actions[j].item()])
                                probs_pool.append([topk_probs[j].item()])

                        else:
                            new_path_pool = []
                            new_probs_pool = []

                            for i in range(len(path_pool)):
                                path = path_pool[i]
                                prob = probs_pool[i]
                                last_entity = path[-1]

                                neighbors = get_actions(last_entity, updated_subgraph).tolist()
                                neighbor_scores = []
                                #print(updated_subgraph.ndata)
                                probs = updated_subgraph.ndata["a_"+str(hop)].to("cpu")
                                x = torch.sum(probs)
                                for neighbor in neighbors:
                                    neighbor_prob = probs[neighbor]
                                    neighbor_scores.append(neighbor_prob.item())
                                neighbor_score = zip(neighbors, neighbor_scores)
                                neighbor_score = sorted(neighbor_score, key=lambda x:x[1], reverse=True)
                                neighbor_score = zip(*neighbor_score)
                                neighbor_score = [list(a) for a in neighbor_score]
                                neighbors, neighbor_scores = neighbor_score[0], neighbor_score[1]
                                k = min(topk[hop], len(set(neighbors)))

                                cnt = 0
                                for j in range(len(probs)):
                                    node = neighbors[j]
                                    prob_node = neighbor_scores[j]
                                    if node in neighbors:
                                        new_path_pool.append(path+[node])
                                        new_probs_pool.append(prob+[prob_node])
                                        cnt += 1
                                        if cnt==k:
                                            break
                            path_pool = new_path_pool
                            probs_pool = new_probs_pool

                    # relation_accuracy(new_relation_path_pool, new_probs_pool, true_relation_path[0])
                    # new_probs_pool = [[-np.log(p+1e-60) for p in prob] for prob in new_probs_pool]
                    new_probs_pool = [reduce(lambda x, y: x*y, probs) for probs in new_probs_pool]
                    probs_relation_entity = zip(new_probs_pool, path_pool)
                    probs_relation_entity = sorted(probs_relation_entity, key=lambda x:x[0], reverse=True)
                    probs_relation_entity = zip(*probs_relation_entity)
                    probs_relation_entity = [list(a) for a in probs_relation_entity]
                    probs_pool , path_pool = probs_relation_entity[0], probs_relation_entity[1]
                    #print('path_pool',path_pool,'probs_pool',probs_pool,'path',path,'edge',edges)
                    calculate_metrics(path_pool ,true_path.tolist(), edges,recall_entity,recall_path)
                    p=10

def predict_paths(policy_file, ConvKGDatasetLoaderTest, opt):
    print('Predicting paths...')
    pretrain_sd = torch.load(policy_file)
    model = AttnIOModel(opt).to(opt["device"])
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    model = model.to(opt["device"])

    K = [[2, 10, 1], [2, 10, 2], [2, 10, 5], [2, 10, 10], [2, 15, 1], [2, 15, 2], [2, 15, 5], [2, 15, 10],
        [2, 20, 1], [2, 20, 2], [2, 20, 5], [2, 20, 10], [2, 25, 1], [2, 25, 2], [2, 25, 5], [2, 25, 10], [2, 50, 50]]
    K = [[2, 15, 15]]


    recall_entity = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
    recall_path = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
    recall_relation = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}

    with torch.no_grad():
        for ks in K:
            for batch in tqdm(ConvKGDatasetLoaderTest):
                # i+=1
                # if i==500:
                #     break
                batch_beam_search(model, batch, opt["device"],  ks,recall_entity,recall_path)
            
            for k, v in recall_entity.items():
                if "@" in k:
                    recall_entity[k] /= recall_entity["counts"]

            for k, v in recall_path.items():
                if "@" in k:
                    recall_path[k] /= recall_path["counts"]
            
            # for k, v in recall_relation.items():
            #     if "@" in k:
            #         recall_relation[k] /= recall_relation["counts"]

            #print(ks)
            path_res = str(recall_path["counts@1"]*100) + "\t" + str(recall_path["counts@3"]*100) + "\t" + str(recall_path["counts@5"]*100) + "\t" + str(recall_path["counts@10"]*100) + "\t" + str(recall_path["counts@25"]*100) + "\t" + str(recall_path["counts"])
            entity_res = str(recall_entity["counts@1"]*100) + "\t" + str(recall_entity["counts@3"]*100) + "\t" + str(recall_entity["counts@5"]*100) + "\t" + str(recall_entity["counts@10"]*100) + "\t" + str(recall_entity["counts@25"]*100) + "\t" + str(recall_path["counts"])
            path_res = "\t".join([f"{round(recall_path['counts@1']*100, 2)}", f"{round(recall_path['counts@3']*100, 2)}", f"{round(recall_path['counts@5']*100, 2)}", f"{round(recall_path['counts@10']*100, 2)}", f"{round(recall_path['counts@25']*100, 2)}", str(recall_path['counts'])])
            entity_res = "\t".join([f"{round(recall_entity['counts@1']*100, 2)}", f"{round(recall_entity['counts@3']*100, 2)}", f"{round(recall_entity['counts@5']*100, 2)}", f"{round(recall_entity['counts@10']*100, 2)}", f"{round(recall_entity['counts@25']*100, 2)}", str(recall_path['counts'])])

            print('path_res',path_res)
            print('entity_res',entity_res)

            for k in recall_entity.keys():
                recall_entity[k] = 0
            for k in recall_path.keys():
                recall_path[k] = 0
    return path_res,entity_res

if __name__ == '__main__':

    # 参数设置
    data_directory = './dataset/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    # 
    opt_dataset = {
                    "entity2entityID_seen": data_directory+"entity2entityID_seen.pkl", 
                    "entity2entityID_unseen": data_directory+"entity2entityID_unseen.pkl", 
                    "relation2relationID": data_directory+"relation2relationID.pkl",
                    "entity_embeddings_seen": data_directory+"entity_embeddings_seen.pkl",
                    "entity_embeddings_unseen": data_directory+"entity_embeddings_unseen.pkl", 
                    "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "dialog_samples": data_directory + "dialog_samples_list.pkl", 
                    "knowledge_graph": data_directory+"opendialkg_triples.txt",
                    'kg_seen':data_directory+'kg_seen.pkl',
                    'kg_unseen':data_directory+'kg_unseen.pkl',
                    "device": device,
                    "n_hop": 2, "n_max": 100, "max_dialogue_history": 3,'batch':batch_size,'seen_percentage':0.8}
    
    data = AttnIODataset(opt=opt_dataset,transform=transforms.Compose([ToTensor(opt_dataset)]))

    opt_model = {
                "n_entity_seen": len(data.entity2entityID_seen), 
                'n_entity_unseen':len(data.entity2entityID_unseen), 
                "n_relation": len(data.relation2relationID),
                "entity_embeddings_seen": data.entity_embeddings_seen, 
                'entity_embeddings_unseen':data.entity_embeddings_unseen,
                "relation_embeddings": data.relation_embeddings,
                "out_dim":80, "in_dim": 768, "batch_size":batch_size, "device": device, "lr": 5e-4, "lr_reduction_factor":0.1, "attn_heads": 5, "beam_size": 5,
                "epoch": 20, "n_hop": 2, "n_max": 100,"model_directory": "./models/", "model_name": f'MultiRe_', "clip": 5, "self_loop_id": data.relation2relationID["self loop"]}
    
    model = AttnIOModel(opt_model)


    # 定义划分的索引范围
    train_indices = list(range(10583))
    valid_indices = list(range(10583, 10583+1200))
    test_seen_indices = list(range(len(data)-1200, len(data)-600))
    test_unseen_indices = list(range(len(data)-600,len(data)))

    # 使用 Subset 创建划分的子数据集
    train_data = Subset(data, train_indices)
    valid_data = Subset(data, valid_indices)
    test_seen_data = Subset(data, test_seen_indices)
    test_unseen_data = Subset(data,test_unseen_indices)

    # 使用Dataloader加载
    # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,collate_fn=MultiRe_collate)
    # valid_dataloader = DataLoader(valid_data, batch_size=batch_size,collate_fn=MultiRe_collate)
    test_seen_dataloader = DataLoader(test_seen_data, batch_size=batch_size,collate_fn=AttnIO_collate)
    test_unseen_dataloader = DataLoader(test_unseen_data, batch_size=batch_size,collate_fn=AttnIO_collate)


    result_file = './result/result.csv'
    fieldnames = ["model_path", "test_type", "path_res", "entity_res"]

    with open(result_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)

    for i in range(2,22,2):
        model_path = f'./models/adam/batch-4_nhop-2_nmax-100/'
        model_name = f'AttnIO_epoch-{i}'
        model_path += model_name

        print(model_path)
        print('test seen')
        path_res_seen,entity_res_seen = predict_paths(model_path, test_seen_dataloader, opt_model)

        with open(result_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([model_name, "seen", path_res_seen, entity_res_seen])

        print('test unseen')
        path_res_unseen,entity_res_unseen = predict_paths(model_path, test_unseen_dataloader, opt_model)

        with open(result_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([model_name, "unseen", path_res_unseen, entity_res_unseen])


