import pandas as pd
import os
import sys
from javalang.ast import Node
import javalang
import random
sys.setrecursionlimit(10000)
import time
class Pipeline:
    def __init__(self, root):
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def parse_source(self):
        path = self.root
        
        import javalang
        def parse_program(func):
            tree = javalang.parse.parse(func)
            return tree

        source = pd.read_pickle(path)
        source['code'] = source['code'].apply(parse_program)

        self.sources = source

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, size):
        self.size = size
        trees = self.sources
        trees['code'] = trees['code'].apply(parse_ast)
        sequences = trees['code'].values.tolist()

        corpus = []
        n = len(sequences)
        for i in range(n):
            seq = str(sequences[i])
            line = seq.split()
            corpus.append(line)
        # print(corpus.shape) 
        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(sentences=corpus, size=size, workers=16, sg=1, min_count=1)
        if not os.path.exists('data/embedding'):
            os.mkdir('data/embedding')
        w2v.save('data/embedding/node_w2v_' + str(size)) 


        # # code 将序列中的词转换为向量
        # from gensim.models.word2vec import Word2Vec
        # w2v = Word2Vec.load(self.root+'train/embedding/node_w2v_'+str(size))
        max_token = len(w2v.wv.index2word)
        vocab = w2v.wv.vocab   # {'词': gensim对象表示的是词的向量}

        import numpy as np
        def trans2feature(sequence):
            
            result = []
            sequence = str(sequence).split()
            for seq in sequence:
                # 获得词对应的词向量
                if seq not in vocab:
                    result.append(w2v.wv.__getitem__(vocab[max_token]))
                else:
                    vector = w2v.wv.__getitem__(seq)
                    result.append(vector)
            return result

        trees['code'] = trees['code'].apply(trans2feature)
        trees.to_pickle('data/vectors.pkl')

    # split data for training, developing and testing
    def split_data(self):

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        def save_flod_data(part, data):  # i: 第几轮， part：训练、测试、验证 ， data：数据
            data_folder = 'data/'+part+'/'
            check_or_create(data_folder)
            data_path = data_folder + 'features.pkl'
            data.to_pickle(data_path)

        data = pd.read_pickle('data/vectors.pkl')
        data = data.sample(frac=1, random_state=666+random.randint(1,1000))

        data_num = len(data)
        ratio = int(data_num *0.8)
        test = data.iloc[ratio:]
        train = data.iloc[:ratio]
  
        # save_flod_data('test', test)
        # save_flod_data('train', train)
        return train, test       

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source()
        # print('train word embedding...')
        self.dictionary_and_embedding(128)
        print('split data...')
        train, test = self.split_data()
        print('end')
        return train, test

def parse_ast(tree):
    res = []
    for path, node in tree:
        # res.append(node)
        pattern = javalang.tree.ReferenceType
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('ReferenceType_' + node.name)
        pattern = javalang.tree.MethodInvocation
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('MethodInvocation_' + node.member)
        pattern = javalang.tree.MethodDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('MethodDeclaration_' + node.name)
        pattern = javalang.tree.TypeDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('TypeDeclaration_' + node.name)
        pattern = javalang.tree.ClassDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('ClassDeclaration_' + node.name)
        pattern = javalang.tree.EnumDeclaration
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append('EnumDeclaration_' + node.name)
        pattern = javalang.tree.IfStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("ifstatement")
        pattern = javalang.tree.WhileStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("whilestatement")
        pattern = javalang.tree.DoStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("dostatement")
        pattern = javalang.tree.ForStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("forstatement")
        pattern = javalang.tree.AssertStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("assertstatement")
        pattern = javalang.tree.BreakStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("breakstatement")
        pattern = javalang.tree.ContinueStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("continuestatement")
        pattern = javalang.tree.ReturnStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("returnstatement")
        pattern = javalang.tree.ThrowStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("throwstatement")
        pattern = javalang.tree.SynchronizedStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("synchronizedstatement")
        pattern = javalang.tree.TryStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("trystatement")
        pattern = javalang.tree.SwitchStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switchstatement")
        pattern = javalang.tree.BlockStatement
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("blockstatement")
        pattern = javalang.tree.StatementExpression
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("statementexpression")
        pattern = javalang.tree.TryResource
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("tryresource")
        pattern = javalang.tree.CatchClause
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catchclause")
        pattern = javalang.tree.CatchClauseParameter
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("catchclauseparameter")
        pattern = javalang.tree.SwitchStatementCase
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("switchstatementcase")
        pattern = javalang.tree.ForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("forcontrol")
        pattern = javalang.tree.EnhancedForControl
        if ((isinstance(pattern, type) and isinstance(node, pattern))):
            res.append("enhancedforcontrol")

    return ' '.join(res)


# projects = ['jmeter_upsample42']

# for project in projects:
#     for i in range(1):

        # print("-------------",project,"   ", i,"--------------------")
        # ppl = Pipeline('data/programs_'+project+'.pkl')
        # # ppl = Pipeline('data-old/'+project+'/')    # 原始只有一个版本的数据
        # ppl.run()

# source = pd.read_pickle('data/programs_lucene3.pkl')

        


