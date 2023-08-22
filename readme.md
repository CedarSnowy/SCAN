# 1. /data/
1. data.json：EARL整理的数据。第一行是一个小的知识图谱（**这个小图谱不知道怎么整理出来的，不使用**）。
2. coref_data.json：对全部数据都进行了指代消解
3. coref_data_notest.json：没有对测试集数据进行指代消解
4. opendialkg_relations.txt：官方数据集，关系的集合
5. opendialkg_triples.txt：官方数据集，所有的三元组，可以构建一个大的知识图谱
6. rules.csv/rules_exact.csv:使用关联规则挖掘得到的规则


# 2. /multi_re_frame/
以同一主题多轮对话为框架
## 1. /dataset/
模型需要使用到的预处理后的数据。此时使用的知识图谱是EARL整理出来的小知识图谱
## 2. /dataset_whole/
模型需要使用到的预处理后的数据。此时使用的知识图谱是原始的大知识图谱
## 3./AttnIO/
当前多轮对话框架下对AttnIO的复现
## 4./seen+unseen_frame/
当前多轮对话框架下，考虑知识图谱中混杂了seen与unseen的实体

# 3. 代码逻辑
1. process_data.py：数据预处理
    1. 使用Albert对实体，关系，对话进行预编码。如果是seen的实体，embeddings会在训练过程中不断地调整。如果unseen，那么就认为始终是与预编码得到的了

2. main.py：训练的程序
3. attnio_build.py:构建模型用的
4. dataset_attnio.py:对数据进行transform的部分
4. attnio_model_new.py：图神经网络计算的部分，基本不用动
5. tester.py:测试模型的程序
6. utils.py：定义了一些辅助函数