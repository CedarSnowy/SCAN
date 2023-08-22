import json
import linecache
from collections import defaultdict
import pickle
import pandas as pd
import logging

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

file_path = "../data/data.json"


def read_json_file_skip_first_line(file_path):
    with open(file_path, "r") as file:
        next(file)  # 跳过第一行
        lines = file.readlines()  # 从第二行开始读取
    return lines

def find_rel(rules_save_path):
    lines = read_json_file_skip_first_line('../data/data.json')
    data = []
    for i,line in enumerate(lines):
        line = json.loads(line)

        rels = []

        kg = line['kg']
        dialogs = line['dialog']
        
        for dialog in dialogs:
            paths = dialog[1]

            if len(paths):
                now_rel = []
                for path in paths:
                    now_rel.append(path[1][:-4])

                    rels.append(path[1][:-4])

        data.append(rels)

        if i == 10582:
            break


    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # 将数据转换为事务格式
    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)


    # 使用Apriori算法查找频繁项集
    frequent_itemsets = apriori(df, min_support=0.002, use_colnames=True)

    # 使用关联规则算法查找关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.002)
    rules = rules.sort_values(by=['confidence'], ascending=False)

    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

    # # 将完整的DataFrame转换为字符串
    # full_print_info = rules.to_string()


    # 将完整的打印信息保存到文件中
    rules.to_csv(rules_save_path,index=False)

    #print(rules)

    return rules

def predict_rel(key, rules):
    # 创建布尔索引
    #print(key)
    filtered_df = rules[rules['antecedents'].apply(lambda x: x == key)]

    sorted_df = filtered_df.sort_values(by=['confidence'], ascending=False).head(5)
    #print('find:',sorted_df['consequents'].tolist())
    return sorted_df['consequents'].tolist()


def predict(data_path,rules):
    line_count = sum(1 for line in open(data_path))  # 获取文件总行数

    start_line = max(1, line_count - 1200 + 1)  # 计算起始行数 (最后六百行的起始行数)
    end_line = line_count  # 结束行数 (文件总行数)

    one_test_num,two_test_num,one_search_success,two_search_success = 0,0,0,0
    all_test_num = 0

    prefix_size = 1

    for line_number in range(start_line, end_line + 1):
        line = linecache.getline(data_path, line_number)  # 逐行读取
        line = json.loads(line)  # 解析 JSON

        prefix_rels = []

        kg = line['kg']
        dialogs = line['dialog']
        
        for dialog in dialogs:
            paths = dialog[1]

            if not len(paths):
                continue
            
            prefix_rels = prefix_rels[-prefix_size:]
            search_rels = []
            if len(prefix_rels):
                search_rels = predict_rel(','.join(prefix_rels),rules)

            true_rel = []

            if len(paths) == 1:
                all_test_num += 1
                for path in paths:
                    rel = path[1][:-4]
                    true_rel.append(rel)
                    
                if len(search_rels):
                    one_test_num += 1
                    if true_rel[0] in search_rels:
                        one_search_success += 1
            else:
                all_test_num += 1
                for path in paths:
                    rel = path[1][:-4]
                    true_rel.append(rel)

                print('true:',true_rel,'search:',search_rels)
                if len(search_rels):
                    two_test_num += 1
                    if true_rel[0] in search_rels and true_rel[1] in search_rels:
                        two_search_success += 1                   
            
            
            
            # print('prefix_rel:',prefix_rels)
            # print('true_rel:',true_rel)

            prefix_rels.extend(true_rel)


    print(one_search_success,one_test_num,two_search_success,two_test_num,all_test_num)




if __name__ == "__main__":

    rules_save_path = '../data/rules.csv'

    find_rel(rules_save_path)

    rules = pd.read_csv(rules_save_path)


    predict('../data/data.json',rules)
