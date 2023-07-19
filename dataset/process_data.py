import json
import linecache
from collections import defaultdict
import pickle
from tqdm import tqdm
import copy
import re
import torch
from transformers import AlbertTokenizer, AlbertModel

file_path = "./data.json"
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", cache_dir="/root/autodl-tmp/cache")
model = AlbertModel.from_pretrained("albert-base-v2", cache_dir="/root/autodl-tmp/cache")
encoder = AlbertModel.from_pretrained('albert-base-v2',cache_dir="/root/autodl-tmp/cache")
set_size = 600

def get_albert_word_representations(sentence_list):
    sentence = ''
    for sen in sentence_list:
        sentence += sen + ' '
    sentence_tokenized = tokenizer.tokenize(sentence)  # 对句子进行分词

    last_sentence = sentence_list[-1]
    mask_count = last_sentence.count("[MASK]")

    mask_indices = [index for index, element in enumerate(sentence_tokenized) if element == '[MASK]'][-mask_count:]

    mask_indices = mask_indices[-mask_count:]

    sentence_ids = tokenizer.convert_tokens_to_ids(sentence_tokenized)  # 将分词转换为对应的词id

    # 添加特殊标记和生成批处理张量
    input_ids = [tokenizer.cls_token_id] + sentence_ids + [tokenizer.sep_token_id]
    inputs = torch.tensor(input_ids).unsqueeze(0)  # 添加批处理维度

    with torch.no_grad():
        outputs = model(inputs)  # 获取ALBERT模型的输出
    

    word_embeddings = outputs.last_hidden_state.squeeze(0)  # 获取最后一层的隐藏状态
    word_representations = word_embeddings[1:-1]  # 去除特殊标记并保留每个单词的编码
    
    return word_representations,mask_indices

def get_albert_representations(sentence):
    sentence_tokenized = tokenizer(sentence, return_tensors="pt")

    sentence_encoding = encoder(**sentence_tokenized)[0].detach().to("cpu")
    sentence_encoding = sentence_encoding[0][0]

    return sentence_encoding

def read_json_file_skip_first_line(file_path):
    with open(file_path, "r") as file:
        next(file)  # 跳过第一行
        lines = file.readlines()  # 从第二行开始读取
    return lines


def get_unseen_entity(file_path):
    line_count = sum(1 for line in open(file_path))  # 获取文件总行数

    start_line = max(1, line_count - 600 + 1)  # 计算起始行数 (最后六百行的起始行数)
    end_line = line_count  # 结束行数 (文件总行数)

    unseen_entity = []
    for line_number in range(start_line, end_line + 1):
        line = linecache.getline(file_path, line_number)  # 逐行读取
        line = json.loads(line)  # 解析 JSON
        kg = line["kg"]
        for key in kg.keys():
            unseen_entity.append(key)
    unseen_entity = list(set(unseen_entity))
    return unseen_entity


def get_whole_kg(file_path):
    with open(file_path, "r") as file:
        first_line = file.readline().strip()
        data = json.loads(first_line)
        return data


def save_kg(file_path):
    unseen_entity = get_unseen_entity(file_path)

    kg_whole = get_whole_kg(file_path)

    # 10048个entity
    item_count = 0  # 初始项总数为0

    # 173572个triple
    for value_list in kg_whole.values():
        item_count += len(value_list)
    
    print(item_count)

    kg_seen = {
        key: value for key, value in kg_whole.items() if key not in unseen_entity
    }
    kg_unseen = {key: value for key, value in kg_whole.items() if key in unseen_entity}

    seen_entity = [key for key in kg_seen.keys()]

    with open("kg_seen.pkl", "wb") as file:
        pickle.dump(kg_seen, file)

    with open("kg_unseen.pkl", "wb") as file:
        pickle.dump(kg_unseen, file)

    return seen_entity, unseen_entity


def index_and_embedding_entity(seen_entity, unseen_entity):
    # index
    idx = 1
    entity2entityID_seen = defaultdict(int)
    for i in range(len(seen_entity)):
        entity = seen_entity[i]
        entity2entityID_seen[entity] = idx
        idx += 1

    entityID2entity_seen = {v: k for k, v in entity2entityID_seen.items()}

    for key,value in entity2entityID_seen.items():
        print(key,value)

    idx = 1
    entity2entityID_unseen = defaultdict(int)
    for i in range(len(unseen_entity)):
        entity = unseen_entity[i]
        entity2entityID_unseen[entity] = idx
        idx += 1

    entityID2entity_unseen = {v: k for k, v in entity2entityID_unseen.items()}

    with open("entity2entityID_seen.pkl", "wb") as file:
        print(f'save{file}')
        pickle.dump(entity2entityID_seen, file)

    with open("entityID2entity_seen.pkl", "wb") as file:
        pickle.dump(entityID2entity_seen, file)

    with open("entity2entityID_unseen.pkl", "wb") as file:
        pickle.dump(entity2entityID_unseen, file)

    with open("entityID2entity_unseen.pkl", "wb") as file:
        pickle.dump(entityID2entity_unseen, file)


    # embedding
    entity_embeddings_seen,entity_embeddings_unseen = [],[]
    for entityId in tqdm(sorted(entityID2entity_seen.keys())):
        entity_embeddings_seen.append(get_albert_representations(entityID2entity_seen[entityId]))
    entity_embeddings_seen = [torch.zeros(768, dtype=torch.float32)] + entity_embeddings_seen

    print(len(entity_embeddings_seen))

    entity_embeddings_seen = torch.stack(entity_embeddings_seen)

    for entityId in tqdm(sorted(entityID2entity_unseen.keys())):
        entity_embeddings_unseen.append(get_albert_representations(entityID2entity_unseen[entityId]))
    entity_embeddings_unseen = [torch.zeros(768, dtype=torch.float32)] + entity_embeddings_unseen
    entity_embeddings_unseen = torch.stack(entity_embeddings_unseen)

    with open("entity_embeddings_seen.pkl", "wb") as file:
        pickle.dump(entity_embeddings_seen, file)

    with open("entity_embeddings_unseen.pkl", "wb") as file:
        pickle.dump(entity_embeddings_unseen, file)

def index_and_embedding_relation(file):
    relation2relationID = defaultdict(int)
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 逐行读取文件内容并将其存储在列表中

    idx = 1
    for line in lines:
        line = line.strip()  # 去除每行开头和结尾的空白字符
        relation2relationID[line] = idx
        idx +=1

    relation2relationID["self loop"] = idx
    relationID2relation = {v:k for k, v in relation2relationID.items()}
    relation_embeddings = []

    for relationId in tqdm(sorted(relationID2relation)):
        relation = re.sub("\~", "reverse ", relationID2relation[relationId])
        relation_embedding = get_albert_representations(relation)
        relation_embeddings.append(relation_embedding)

    relation_embeddings = [torch.zeros(768, dtype=torch.float32)] + relation_embeddings
    relation_embeddings = torch.stack(relation_embeddings)


    with open("relation2relationID.pkl", "wb") as f:
        pickle.dump(relation2relationID, f)
    
    with open("relationID2relation.pkl", "wb") as f:
        pickle.dump(relationID2relation, f)

    with open("relation_embeddings.pkl", "wb") as f:
        pickle.dump(relation_embeddings, f)

def process_dialog(file,seen_percentage,max_history_num):
    # 读取data.json，跳过第一行
    lines = read_json_file_skip_first_line(file)

    # 划分数据集
    train,valid,test_seen,test_unseen = 10583,1200,600,600
    train_seen = [0,int(train * seen_percentage)-1]
    train_unseen = [int(train * seen_percentage) ,10583 -1]
    valid_seen = [10583,10583 + int(valid * seen_percentage)-1]
    valid_unseen = [10583 + int(valid * seen_percentage),12983 - 1200]
    print('train_seen,train_unseen,valid_seen,valid_unseen',train_seen,train_unseen,valid_seen,valid_unseen)

    # 处理unseen部分
    dialog_mask = {}
    for i,line in enumerate(lines):
        if train_unseen[0] <= i <= train_unseen[1] or valid_unseen[0] <= i <= valid_unseen[1] or i>=valid_unseen[1] +600:
            line = json.loads(line)
            dialogs = line['dialog']
            mask_entity = {}
            for j in range(len(dialogs)): 
                dialog = dialogs[j]
                context,path = dialog[0],dialog[1]             
                path_num = len(path)
                if path_num > 0:
                    # 记录出现过的entity
                    existing_entity = []
                    for k in range(path_num):
                        existing_entity.append(path[k][0]) # head出现
                        # existing_entity.append(path[k][2]) # tail出现
                    existing_entity = list(set(existing_entity))
                    
                    # 将原先句子中出现的entity替换为MASK，并且记录
                    masked_sentence = context
                    masked_indices = []

                    # 遍历词组列表中的每个词组
                    for phrase in existing_entity:
                        phrase_pos = masked_sentence.lower().find(phrase.lower())  # 查找词组在句子中的位置
                        if phrase_pos != -1:
                            masked_sentence = masked_sentence[:phrase_pos] + "[MASK]" + masked_sentence[phrase_pos+len(phrase):]
                            masked_indices.append((phrase_pos, phrase_pos+len(phrase)))

                    # 获取每个 MASK 原先对应的实体
                    original_entity = []
                    for pos in masked_indices:
                        original_entity.append(context[pos[0]:pos[1]])

                    # if len(original_entity) > 0:
                    #     mask_entity[masked_sentence] = original_entity[::-1] # 倒置

                    dialog[0] = masked_sentence
                    dialog.append(original_entity[::-1])
                    dialogs[j] = dialog

            # 写回lines
            line['dialog'] = dialogs
            #line['mask embeddings'] = mask_entity
            lines[i] = json.dumps(line)
            #dialog_mask[i+1] = mask_entity 

    # 处理每次对话
    dialogID = 1
    dialog_samples = {}  # 每个dialog里面可能有多个训练样本
    for line in lines:
        # 每一行是关于一个主题的多轮对话
        utterances = []
        samples_flag = {}
        samples = []
        line = json.loads(line)
        dialogs = line['dialog']
        for i in range(len(dialogs)):
            dialog = dialogs[i]
            context,path = dialog[0],dialog[1]
            original_entity = dialog[2] if len(dialog) == 3 else None
            utterances.append(context)

            if len(utterances)>max_history_num:
                utterances.pop(0)
            path_num = len(path)
            if path_num > 0:
                # 是一个训练样本
                one_sample = {} # {'utterances':...,'seeds':...,'paths':..,'origins:(mask)'...}
                one_sample['utterances'] = copy.deepcopy(utterances)
                start_seed = []
                jump_path = []
                for i in range(path_num):
                    start_seed.append(path[i][0])
                    path[i][1] = path[i][1][:-4]
                    jump_path.append(path[i])

                one_sample['seeds'] = start_seed
                one_sample['paths'] = jump_path
                if original_entity is not None and len(original_entity) > 0:
                    one_sample['origins'] = original_entity

                samples.append(one_sample)

        samples_flag['samples'] = samples
        if  train_unseen[0] <= dialogID -1  <= train_unseen[1] or valid_unseen[0] <= dialogID -1 <= valid_unseen[1] or dialogID -1>=valid_unseen[1] +600:
            samples_flag['flag'] = 0
        else:
            samples_flag['flag'] = 1

        dialog_samples[dialogID] = samples_flag
        dialogID += 1
    print(dialog_samples[11750])
    #print(dialog_mask[10000])
    #print(dialog_samples[10000])
    return dialog_samples


def encoder_dialog(dialog_samples): 
    dialog_samples_list = []

    for dialogID,samples_flag in tqdm(dialog_samples.items()):
        if samples_flag['flag'] == 1:
            samples = samples_flag['samples']

            for i in range(len(samples)):
                sample = samples[i]
                utterances = sample['utterances']
                all_context = ''
                for utterance in utterances:
                    all_context += utterance + ' '
                encoder_utterances = get_albert_representations(all_context)
                sample['utterances'] = encoder_utterances
                samples[i] = sample
            
            samples_flag['samples'] = samples


        else:
            samples = samples_flag['samples']

            for i in range(len(samples)):
                sample = samples[i]
                utterances = sample['utterances']
                # 得到单个词的encoder
                encoder_output, mask_indices = get_albert_word_representations(utterances)
                mask_embeddings = [encoder_output[mask] for mask in mask_indices]
                # 得到整个句子的encoder
                all_context = ''
                for utterance in utterances:
                    all_context += utterance + ' '
                encoder_utterances = get_albert_representations(all_context)
                # 写回
                sample['utterances'] = encoder_utterances
                sample['MASK embedding'] = mask_embeddings
                samples[i] = sample

            samples_flag['samples'] = samples
        
        dialog_samples_list.append({'dialogID':dialogID,'samples_flag':samples_flag})


        dialog_samples[dialogID] = samples_flag


    #print(dialog_samples[10000])

    with open('dialog_samples.pkl','wb') as f:
        pickle.dump(dialog_samples, f)

    with open('dialog_samples_list.pkl','wb') as f:
        pickle.dump(dialog_samples_list, f)



if __name__ == "__main__":
    #seen_entity, unseen_entity = save_kg(file_path)

    #index_and_embedding_entity(seen_entity, unseen_entity)

    #index_and_embedding_relation('./opendialkg_relations.txt')

    dialog_samples = process_dialog(file_path,0.8,3)
    
    encoder_dialog(dialog_samples)

