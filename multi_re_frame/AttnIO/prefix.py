import json

from prefixspan import PrefixSpan
from utils import load_pickle_file

file_path = '.../data/data.json'

relation2relationID = load_pickle_file('../dataset/relation2relationID.pkl')

def read_json_file_skip_first_line(file_path):
    with open(file_path, "r") as file:
        next(file)  # 跳过第一行
        lines = file.readlines()  # 从第二行开始读取
    return lines

def find_rel():
    lines = read_json_file_skip_first_line('../../data/data.json')
    data = []
    for i,line in enumerate(lines):
        line = json.loads(line)

        rels = []

        kg = line['kg']
        dialogs = line['dialog']
        
        for dialog in dialogs:
            paths = dialog[1]

            if len(paths):
                now_rel = ()
                for path in paths:
                    if path[1][:-4] in relation2relationID.keys():
                        #rels.append(relation2relationID[path[1][:-4]])
                        rels.append(path[1][:-4])

        if len(rels) > 1:
            data.append(rels)

        if i == 10582:
            break
    return data

if __name__ =='__main__':
    data = find_rel()

    ps = PrefixSpan(data)
    #print(ps.frequent(1))
    print(ps.topk(20,closed = True))
    print()
    #print(ps.frequent(20))
    #print(ps.frequent(1))