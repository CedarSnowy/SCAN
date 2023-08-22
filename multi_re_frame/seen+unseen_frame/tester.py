from __future__ import print_function, division
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch
from skimage import io, transform
import numpy as np

from tqdm import tqdm
from functools import reduce

import logging
import csv

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter


from MulitRe_build import MultiReModel
from dataset_dense import ToTensor,MultiRe_collate,MultiReDataset
from torch.utils.data import Subset
from dgl.sampling import sample_neighbors
from utils import load_pickle_file
from main import opt_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

entityID2entity_seen = load_pickle_file('./dataset/entityID2entity_seen.pkl')
entityID2entity_unseen = load_pickle_file('./dataset/entityID2entity_unseen.pkl')

def path2node(path,seen):
    start = path[0]
    middle = path[1]
    tail = path[2]

    try:
        if seen:
            print(f'{entityID2entity_seen[start]} --> {entityID2entity_seen[middle]} --> {entityID2entity_seen[tail]}')
        else:
            print(f'{entityID2entity_unseen[start]} --> {entityID2entity_unseen[middle]} --> {entityID2entity_unseen[tail]}')
    except:
        pass



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


def calculate_metrics(paths, true_path, edges,recall_entity,recall_path,no_graph):
    recall_entity_one_jump,recall_entity_two_jump,recall_entity_all = recall_entity[0],recall_entity[1],recall_entity[2]
    recall_path_one_jump,recall_path_two_jump,recall_path_all = recall_path[0],recall_path[1],recall_path[2]

    recall_entity_all["counts"] += 1
    recall_path_all["counts"] += 1

    jump = 1 if true_path[1] == true_path[2] else 2

    if jump == 1:
        recall_entity_one_jump["counts"] += 1
        recall_path_one_jump["counts"] += 1
    else:
        recall_entity_two_jump["counts"] += 1
        recall_path_two_jump["counts"] += 1  

    if no_graph:
        return

    K = [1, 3, 5, 10, 25]
    x = edges
    filtered_paths = filter_paths(paths)
    
    entities = [path[-1] for path in filtered_paths]

    x = 0
    for k in K:
        if true_path in filtered_paths[:k]:
            recall_path_all["counts@"+str(k)] += 1

            if jump == 1:
                recall_path_one_jump["counts@"+str(k)] += 1
            else:
                recall_path_two_jump["counts@"+str(k)] += 1

        if true_path[-1] in entities[:k]:
            x = 1
            recall_entity_all["counts@"+str(k)] += 1

            if jump == 1:
                recall_entity_one_jump["counts@"+str(k)] += 1
            else:
                recall_entity_two_jump["counts@"+str(k)] += 1

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


def batch_beam_search(model, batch_data, device, topk,recall_entity,recall_path,predict_entity_list,true_entity_list,seen):
    with torch.no_grad():
        model.eval()
        # 读取数据
        for one_data in batch_data:
            state,sample_num = one_data['state'],one_data['sample_num']
            one_dialogue_representation = one_data['one_dialogue_representation']
            one_extra_dialogue_representation = one_data['one_extra_dialogue_representation']
            one_seed_Entities = one_data['one_seed_Entities']
            one_subgraph = one_data['one_subgraph']
            one_entity_paths = one_data['one_entity_paths']
            one_sub_embedding = one_data['one_sub_embedding']
            one_entity_state_1_jump = one_data['one_entity_state_1_jump']
            one_entity_state_2_jump = one_data['one_entity_state_2_jump']
            one_head_entities_1_jump = one_data['one_head_entities_1_jump']
            one_head_entities_2_jump = one_data['one_head_entities_2_jump']
            one_tail_entities_1_jump = one_data['one_tail_entities_1_jump']
            one_tail_entities_2_jump = one_data['one_tail_entities_2_jump']
            one_edge_relations_1_jump = one_data['one_edge_relations_1_jump']
            one_edge_relations_2_jump = one_data['one_edge_relations_2_jump']
            one_unseen_rel = one_data['one_unseen_rel']
            one_node2nodeID = one_data['one_node2nodeID']
            one_nodeID2node = one_data['one_nodeID2node']
            one_startEntity_in_seenKG = one_data['one_startEntity_in_seenKG']

            for i in range(sample_num):
                last_graph = None if i == 0 else last_graph
                last_entity_rep = None if i == 0 else last_entity_rep

                last_graph = None if i == 0 else last_graph
                last_entity_rep = None if i == 0 else last_entity_rep

                dialogue_representation = one_dialogue_representation[i]['dialogue_representation']
                extra_dialogue_representation = one_extra_dialogue_representation[i]['extra_dialogue_representation']
                seed_entities = one_seed_Entities[i]['seed_entities']
                subgraph = one_subgraph[i]['subgraph']
                paths = one_entity_paths[i]['entity_paths']
                sub_embedding = one_sub_embedding[i]['sub_embedding']
                entity_state_1_jump = one_entity_state_1_jump[i]['entity_state_1_jump']
                entity_state_2_jump = one_entity_state_2_jump[i]['entity_state_2_jump']
                head_entities_1_jump = one_head_entities_1_jump[i]['head_entities_1_jump']
                head_entities_2_jump = one_head_entities_2_jump[i]['head_entities_2_jump']
                tail_entities_1_jump = one_tail_entities_1_jump[i]['tail_entities_1_jump']
                tail_entities_2_jump = one_tail_entities_2_jump[i]['tail_entities_2_jump']
                edge_relations_1_jump = one_edge_relations_1_jump[i]['edge_relations_1_jump']
                edge_relations_2_jump = one_edge_relations_2_jump[i]['edge_relations_2_jump']
                unseen_rel = one_unseen_rel[i]['unseen_rel']
                node2nodeID = one_node2nodeID[i]['node2nodeID']
                nodeID2node = one_nodeID2node[i]['nodeID2node']
                startEntity_in_seenKG = one_startEntity_in_seenKG[i]['startEntity_in_seenKG']


                if len(seed_entities) == 0:
                    continue
                elif ((state == 'test' or state == 'valid') and not startEntity_in_seenKG) and len(sub_embedding) == 0:
                    continue
                elif not subgraph.ndata['nodeId'].numel():
                    path_pool = edges = None
                    true_path = paths.to(device)
                    calculate_metrics(path_pool ,true_path.tolist(), edges,recall_entity,recall_path,True)
                else:
                    dialogue_representation = dialogue_representation.to(device) # [1,768]
                    seed_entities = seed_entities.to(device)
                    subgraph = subgraph.to(device)
                    true_path = paths.to(device)
             
                    updated_subgraph, expilcit_entity_rep = model(dialogue_representation, extra_dialogue_representation,seed_entities, subgraph, sub_embedding,entity_state_1_jump,
                                                                 entity_state_2_jump,head_entities_1_jump,head_entities_2_jump,tail_entities_1_jump,tail_entities_2_jump,edge_relations_1_jump,
                                                                 edge_relations_2_jump,unseen_rel,node2nodeID,nodeID2node,last_graph = last_graph,last_entity_rep = last_entity_rep)
                    edges = updated_subgraph.edges()

                    last_graph,last_entity_rep = updated_subgraph,expilcit_entity_rep 
        
                    path_pool = []  # list of list, size=bs
                    probs_pool = []

                    for hop in range(3):
                        # 两跳内实体
                        if hop==0:
                            #print('seed',seed_entities)
                            k = min(topk[0], len(seed_entities))

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
                    calculate_metrics(path_pool ,true_path.tolist(), edges,recall_entity,recall_path,False)

                    filtered_paths = filter_paths(path_pool)
                    pred_entity = filtered_paths[0][-1]

                    predict_entity_list.append(pred_entity)
                    true_entity_list.append(true_path.tolist()[-1])

                    # print('true:')
                    # path2node(true_path.tolist(),seen)
                    # print('pred:')
                    # for idx,path in enumerate(path_pool):
                    #     path2node(path,seen)
                    #     if idx == 10:
                    #         break
                    # print()


def predict_paths(policy_file, ConvKGDatasetLoaderTest, opt,seen):
    print('Predicting paths...')
    pretrain_sd = torch.load(policy_file)
    model = MultiReModel(opt).to(opt["device"])
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    model = model.to(opt["device"])

    K = [[2, 10, 1], [2, 10, 2], [2, 10, 5], [2, 10, 10], [2, 15, 1], [2, 15, 2], [2, 15, 5], [2, 15, 10],
        [2, 20, 1], [2, 20, 2], [2, 20, 5], [2, 20, 10], [2, 25, 1], [2, 25, 2], [2, 25, 5], [2, 25, 10], [2, 50, 50]]
    K = [[2, 15, 15]]


    recall_entity_one_jump = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
    recall_path_one_jump = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
    
    recall_entity_two_jump = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
    recall_path_two_jump = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}

    recall_entity_all = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}
    recall_path_all = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}

    recall_relation = {"counts@1":0, "counts@3":0, "counts@5":0, "counts@10":0, "counts@25":0, "counts": 0}

    with torch.no_grad():
        for ks in K:
            predict_entity_list,true_entity_list = [],[]
            for batch in tqdm(ConvKGDatasetLoaderTest):
                # i+=1
                # if i==500:
                #     break
                
                batch_beam_search(model, batch, opt["device"], ks,[recall_entity_one_jump,recall_entity_two_jump,recall_entity_all],[recall_path_one_jump,recall_path_two_jump,recall_path_all],predict_entity_list,true_entity_list,seen)

            
            for k, v in recall_entity_all.items():
                if "@" in k:
                    recall_entity_all[k] /= recall_entity_all["counts"]

            for k, v in recall_path_all.items():
                if "@" in k:
                    recall_path_all[k] /= recall_path_all["counts"]

            for k, v in recall_entity_one_jump.items():
                if "@" in k:
                    recall_entity_one_jump[k] /= recall_entity_one_jump["counts"]

            for k, v in recall_path_one_jump.items():
                if "@" in k:
                    recall_path_one_jump[k] /= recall_path_one_jump["counts"]

            for k, v in recall_entity_two_jump.items():
                if "@" in k:
                    recall_entity_two_jump[k] /= recall_entity_two_jump["counts"]

            for k, v in recall_path_two_jump.items():
                if "@" in k:
                    recall_path_two_jump[k] /= recall_path_two_jump["counts"]
            
            # for k, v in recall_relation.items():
            #     if "@" in k:
            #         recall_relation[k] /= recall_relation["counts"]

            #print(ks)
                # 计算每个样本的预测是否成功
           
            # 计算准确率
            precision = accuracy_score(true_entity_list, predict_entity_list)

            # 计算召回率
            recall_macro = recall_score(true_entity_list, predict_entity_list, average='macro')  # 设置average参数为'macro'，计算宏平均召回率

            # 计算F1得分
            f1_macro = f1_score(true_entity_list, predict_entity_list, average='macro')  # 设置average参数为'macro'，计算宏平均F1得分

            # 计算召回率
            recall_micro = recall_score(true_entity_list, predict_entity_list, average='micro')  # 设置average参数为'macro'，计算宏平均召回率

            # 计算F1得分
            f1_micro = f1_score(true_entity_list, predict_entity_list, average='micro')  # 设置average参数为'macro'，计算宏平均F1得分

            
            # path_res_all = str(recall_path_all["counts@1"]*100) + "\t" + str(recall_path_all["counts@3"]*100) + "\t" + str(recall_path_all["counts@5"]*100) + "\t" + str(recall_path_all["counts@10"]*100) + "\t" + str(recall_path_all["counts@25"]*100) + "\t" + str(recall_path_all["counts"])
            # entity_res_all = str(recall_entity_all["counts@1"]*100) + "\t" + str(recall_entity_all["counts@3"]*100) + "\t" + str(recall_entity_all["counts@5"]*100) + "\t" + str(recall_entity_all["counts@10"]*100) + "\t" + str(recall_entity_all["counts@25"]*100) + "\t" + str(recall_entity_all["counts"])
            # path_res_all = "\t".join([f"{round(recall_path_all['counts@1']*100, 2)}", f"{round(recall_path_all['counts@3']*100, 2)}", f"{round(recall_path_all['counts@5']*100, 2)}", f"{round(recall_path_all['counts@10']*100, 2)}", f"{round(recall_path_all['counts@25']*100, 2)}", str(recall_path_all['counts'])])
            # entity_res_all = "\t".join([f"{round(recall_entity_all['counts@1']*100, 2)}", f"{round(recall_entity_all['counts@3']*100, 2)}", f"{round(recall_entity_all['counts@5']*100, 2)}", f"{round(recall_entity_all['counts@10']*100, 2)}", f"{round(recall_entity_all['counts@25']*100, 2)}", str(recall_entity_all['counts'])])

            path_res_all = "\t".join([f"{round(recall_path_all['counts@1']*100, 2)}", 
                                    f"{round(recall_path_all['counts@3']*100, 2)}", 
                                    f"{round(recall_path_all['counts@5']*100, 2)}", 
                                    f"{round(recall_path_all['counts@10']*100, 2)}", 
                                    f"{round(recall_path_all['counts@25']*100, 2)}", 
                                    str(recall_path_all['counts'])])

            entity_res_all = "\t".join([f"{round(recall_entity_all['counts@1']*100, 2)}", 
                                        f"{round(recall_entity_all['counts@3']*100, 2)}", 
                                        f"{round(recall_entity_all['counts@5']*100, 2)}", 
                                        f"{round(recall_entity_all['counts@10']*100, 2)}", 
                                        f"{round(recall_entity_all['counts@25']*100, 2)}", 
                                        str(recall_entity_all['counts'])])

            path_res_one_jump = "\t".join([f"{round(recall_path_one_jump['counts@1']*100, 2)}", 
                               f"{round(recall_path_one_jump['counts@3']*100, 2)}", 
                               f"{round(recall_path_one_jump['counts@5']*100, 2)}", 
                               f"{round(recall_path_one_jump['counts@10']*100, 2)}", 
                               f"{round(recall_path_one_jump['counts@25']*100, 2)}", 
                               str(recall_path_one_jump['counts'])])

            entity_res_one_jump = "\t".join([f"{round(recall_entity_one_jump['counts@1']*100, 2)}", 
                                 f"{round(recall_entity_one_jump['counts@3']*100, 2)}", 
                                 f"{round(recall_entity_one_jump['counts@5']*100, 2)}", 
                                 f"{round(recall_entity_one_jump['counts@10']*100, 2)}", 
                                 f"{round(recall_entity_one_jump['counts@25']*100, 2)}", 
                                 str(recall_entity_one_jump['counts'])])
            
            path_res_two_jump = "\t".join([f"{round(recall_path_two_jump['counts@1']*100, 2)}", 
                               f"{round(recall_path_two_jump['counts@3']*100, 2)}", 
                               f"{round(recall_path_two_jump['counts@5']*100, 2)}", 
                               f"{round(recall_path_two_jump['counts@10']*100, 2)}", 
                               f"{round(recall_path_two_jump['counts@25']*100, 2)}", 
                               str(recall_path_two_jump['counts'])])

            entity_res_two_jump = "\t".join([f"{round(recall_entity_two_jump['counts@1']*100, 2)}", 
                                 f"{round(recall_entity_two_jump['counts@3']*100, 2)}", 
                                 f"{round(recall_entity_two_jump['counts@5']*100, 2)}", 
                                 f"{round(recall_entity_two_jump['counts@10']*100, 2)}", 
                                 f"{round(recall_entity_two_jump['counts@25']*100, 2)}", 
                                 str(recall_entity_two_jump['counts'])])

            

            logging.info(f"path_res_all: {path_res_all}")
            logging.info(f"entity_res_all: {entity_res_all}")

            logging.info(f"path_res_one_jump: {path_res_one_jump}")
            logging.info(f"entity_res_one_jump: {entity_res_one_jump}")

            logging.info(f"path_res_two_jump: {path_res_two_jump}")
            logging.info(f"entity_res_two_jump: {entity_res_two_jump}")

            logging.info(f"precision: {precision}, recall_macro: {recall_macro}, f1_macro: {f1_macro}, recall_micro: {recall_micro}, f1_micro: {f1_micro}")

            for k in recall_entity_all.keys():
                recall_entity_all[k] = 0

            for k in recall_path_all.keys():
                recall_path_all[k] = 0

            for k in recall_entity_one_jump.keys():
                recall_entity_one_jump[k] = 0

            for k in recall_path_one_jump.keys():
                recall_path_one_jump[k] = 0

            for k in recall_entity_two_jump.keys():
                recall_entity_two_jump[k] = 0

            for k in recall_path_two_jump.keys():
                recall_path_two_jump[k] = 0

        
    return path_res_all,entity_res_all,precision,recall_macro,f1_macro,recall_micro,f1_micro

if __name__ == '__main__':

    # 参数设置
    data_directory = './dataset/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    n_hop = 2
    n_max = 100
    save_path = f'./models/dense/batch-{batch_size}_nhop-{n_hop}_nmax-{n_max}/'

    # 
    opt_dataset = {
                        'entity2entityID_all':data_directory + 'entity2entityID_all.pkl',
                        "entity2entityID_seen": data_directory+"entity2entityID_seen.pkl", 
                        "entity2entityID_unseen": data_directory+"entity2entityID_unseen.pkl", 
                        'entityID2entity_all':data_directory + 'entityID2entity_all.pkl',
                        "relation2relationID": data_directory+"relation2relationID.pkl",
                        "entity_embeddings_seen": data_directory+"entity_embeddings_seen.pkl",
                        "entity_embeddings_unseen": data_directory+"entity_embeddings_unseen.pkl", 
                        "relation_embeddings": data_directory+"relation_embeddings.pkl",
                        "dialog_samples": data_directory+ "dialog_samples_list.pkl", 
                        "knowledge_graph": data_directory+"opendialkg_triples.txt",
                        'kg_whole':data_directory + 'kg_whole.pkl',
                        'kg_seen':data_directory+'kg_seen.pkl',
                        'kg_unseen':data_directory+'kg_unseen.pkl',
                        "device": device,
                        "n_hop": n_hop, "n_max":n_max, "max_dialogue_history": 3,'batch':batch_size,'seen_percentage':0.95,'train_unseen_rate':0.95,'max_edge':10000,'prefix_size':1}
    
    data = MultiReDataset(opt=opt_dataset,transform=transforms.Compose([ToTensor(opt_dataset)]))

    # multi_apper_dataset = {
    #                 "entity2entityID_seen": data_directory+"entity2entityID_seen.pkl", 
    #                 "entity2entityID_unseen": data_directory+"entity2entityID_unseen.pkl", 
    #                 "relation2relationID": data_directory+"relation2relationID.pkl",
    #                 "entity_embeddings_seen": data_directory+"entity_embeddings_seen.pkl",
    #                 "entity_embeddings_unseen": data_directory+"entity_embeddings_unseen.pkl", 
    #                 "relation_embeddings": data_directory+"relation_embeddings.pkl",
    #                 "dialog_samples": data_directory + "multi_apper_test_list.pkl", 
    #                 "knowledge_graph": data_directory+"opendialkg_triples.txt",
    #                 'kg_seen':data_directory+'kg_seen.pkl',
    #                 'kg_unseen':data_directory+'kg_unseen.pkl',
    #                 "device": device,
    #                 "n_hop": 2, "n_max": 100, "max_dialogue_history": 3,'batch':batch_size,'seen_percentage':0.8}
    
    # multi_apper_data = MultiReDataset(opt = multi_apper_dataset,transform=transforms.Compose([ToTensor(multi_apper_dataset)]))


    # opt_model = {"n_entity_seen": len(data.entity2entityID_seen), 
    #              'n_entity_unseen':len(data.entity2entityID_unseen), 
    #              "n_relation": len(data.relation2relationID),
    #             "entity_embeddings_seen": data.entity_embeddings_seen, 
    #             "relation_embeddings": data.relation_embeddings,
    #             "out_dim":80, "in_dim": 768, "batch_size":batch_size, "device": device, "lr": 5e-4, "lr_reduction_factor":0.1, "attn_heads": 5, "beam_size": 5,
    #             "epoch": 20, "n_hop": 2, "n_max": 100,"model_directory":save_path, "model_name": f'MultiRe', "clip": 5, "self_loop_id": data.relation2relationID["self loop"]}
    
    model = MultiReModel(opt_model)


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
    test_seen_dataloader = DataLoader(test_seen_data, batch_size=batch_size,collate_fn=MultiRe_collate)
    test_unseen_dataloader = DataLoader(test_unseen_data, batch_size=batch_size,collate_fn=MultiRe_collate)

    #test_multi_apper_dataloader = DataLoader(multi_apper_data,batch_size=batch_size,collate_fn=MultiRe_collate)

    result_name = './result/as_no-predict'
    result_file = result_name+'.csv'

    logging.basicConfig(filename=result_name +'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    fieldnames = ["model_path", "test_type", "path_res", "entity_res",'precision','recall_macro','f1_macro','recall_micro','f1_micro']

    with open(result_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)

    for i in range(2,22,2):
        model_path = f'./models/as/batch-4_nhop-2_nmax-100/'
        model_name = f'MultiRe_epoch-{i}'
        model_path += model_name

        print(model_path)
        logging.info(model_path)
        print('test seen')
        logging.info('test seen')
        path_res_seen,entity_res_seen,precision,recall_macro,f1_macro,recall_micro,f1_micro = predict_paths(model_path, test_seen_dataloader, opt_model,seen = True)

        with open(result_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([model_name, "seen", path_res_seen, entity_res_seen,precision,recall_macro,f1_macro,recall_micro,f1_micro])

        print('test unseen')
        logging.info('test unseen')
        path_res_unseen,entity_res_unseen,precision,recall_macro,f1_macro,recall_micro,f1_micro = predict_paths(model_path, test_unseen_dataloader, opt_model,seen = False)

        logging.info("")

        with open(result_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([model_name, "unseen", path_res_unseen, entity_res_unseen,precision,recall_macro,f1_macro,recall_micro,f1_micro])

        # path_res_unseen,entity_res_unseen,precision,recall_macro,f1_macro,recall_micro,f1_micro = predict_paths(model_path, test_multi_apper_dataloader, opt_model,seen = False)

        # with open(result_file, mode="a", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerow([model_name, "unseen", path_res_unseen, entity_res_unseen,precision,recall_macro,f1_macro,recall_micro,f1_micro])



