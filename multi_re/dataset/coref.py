import spacy
import neuralcoref
import json
from tqdm import tqdm
import re

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)


def read_json_file_skip_first_line(file_path):
    with open(file_path, "r") as file:
        next(file)  # 跳过第一行
        lines = file.readlines()  # 从第二行开始读取
    return lines


def resolve_anaphora(sentences, current_context):
    # 加载英文模型
    nlp = spacy.load('en_core_web_sm')

    # 添加neuralcoref组件
    neuralcoref.add_to_pipe(nlp)

    # 将句子拼接成一整个文本
    text = ' '.join(sentences) + '##' + current_context

    # 处理文本
    doc = nlp(text)

    # 进行指代消解
    resolved_text = doc._.coref_resolved

    # 使用正则表达式从解析后的文本中提取句子
    resolved_current_sentence = resolved_text.split('##')[-1]

    return resolved_current_sentence

def process_dialog(file):
    # 读取data.json，跳过第一行
    lines = read_json_file_skip_first_line(file)
    new_lines = []
    for i, line in tqdm(enumerate(lines)):
        line = json.loads(line)
        dialogs = line['dialog']

        origin_context = []
        resolved_context = []

        for dialog in dialogs:
            current_context = dialog[0]
            current_resolved_context = resolve_anaphora(origin_context, current_context)

            resolved_context.append(current_resolved_context)
            origin_context.append(current_context)

        print(len(origin_context),len(resolved_context))
        if len(origin_context) != len(resolved_context):
            print(origin_context)
            print(resolved_context)

        for i in range(len(dialogs)):
            dialog = dialogs[i]
            dialog[0] = resolved_context[i]

        line['dialog'] = dialogs

        new_lines.append(json.dumps(line)+'\n')

        # print(line)

        # lines[i] = json.dumps(line)
    
    with open('./coref_data.json', 'w') as file:
        file.writelines(new_lines)


if __name__ == '__main__':
    process_dialog('./data.json')
