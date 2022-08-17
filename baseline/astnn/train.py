import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import sys
sys.setrecursionlimit(99999)
# import xlwt

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        # labels.append(item[2]-1)
        labels.append(item[2])
    return data, torch.LongTensor(labels)

# def write_excel_xls(path, sheet_name, value):
#     index = len(value)  # 获取需要写入数据的行数
#     workbook = xlwt.Workbook()  # 新建一个工作簿
#     sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
#     for i in range(0, index):
#         for j in range(0, len(value[i])):
#             sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
#     workbook.save(path)  # 保存工作簿
#     print("xls格式表格写入数据成功！")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
res = [] 
projects=['lucene']    #'lucene',
data = []
data_avg = []

# count = 30
count = 1
for project in projects:
    start_time2=time.time()
    data.append([project, '-', '-'])
    data_avg.append([project, '-', '-'])
    acc_sum = 0
    auc_sum = 0
    f1_sum = 0
    p_sum = 0
    r_sum = 0
    for h in range(count):
        root = 'data/' + project +'/'
        train_data = pd.read_pickle(root+'train/blocks.pkl')
        val_data = pd.read_pickle(root + 'dev/blocks.pkl')
        test_data = pd.read_pickle(root+'test/blocks.pkl')

        word2vec = Word2Vec.load(root+"train/embedding/node_w2v_100").wv
        embeddings = np.zeros((word2vec.wv.vectors.shape[0] + 1, word2vec.wv.vectors.shape[1]), dtype="float32")
        embeddings[:word2vec.wv.vectors.shape[0]] = word2vec.wv.vectors

        HIDDEN_DIM = 100
        ENCODE_DIM = 100
        # LABELS = 104
        LABELS = 2
        EPOCHS = 15
        # EPOCHS = 10
        BATCH_SIZE = 2
        # BATCH_SIZE = 32
        USE_GPU = True
        # USE_GPU = False
        MAX_TOKENS = word2vec.wv.vectors.shape[0]
        EMBEDDING_DIM = word2vec.wv.vectors.shape[1]

        model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
        if USE_GPU:
            model.cuda()

        parameters = model.parameters()
        optimizer = torch.optim.Adamax(parameters)
        # print(parameters)
        # print(list(parameters))
        loss_function = torch.nn.CrossEntropyLoss()

        train_loss_ = []
        val_loss_ = []
        train_acc_ = []
        val_acc_ = []
        best_acc = 0.0
        best_loss = 1.0  # 根据验证集的损失选择最优模型
        print('Start training...')
        # training procedure
        best_model = model
        for epoch in range(EPOCHS):
            start_time = time.time()

            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(train_data):
                batch = get_batch(train_data, i, BATCH_SIZE)
                i += BATCH_SIZE
                train_inputs, train_labels = batch
                if USE_GPU:
                    train_inputs, train_labels = train_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                output = model(train_inputs)
                # print("output = ", output)
                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()

                # calc training acc
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == train_labels).sum()
                total += len(train_labels)
                total_loss += loss.item()*len(train_inputs)

            train_loss_.append(total_loss / total)
            train_acc_.append(total_acc.item() / total)
            # validation epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(val_data):
                batch = get_batch(val_data, i, BATCH_SIZE)
                i += BATCH_SIZE
                val_inputs, val_labels = batch
                if USE_GPU:
                    val_inputs, val_labels = val_inputs, val_labels.cuda()

                model.batch_size = len(val_labels)
                model.hidden = model.init_hidden()
                output = model(val_inputs)

                loss = loss_function(output, Variable(val_labels))

            # calc valing acc
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == val_labels).sum()
                total += len(val_labels)
                total_loss += loss.item()*len(val_inputs)
            val_loss_.append(total_loss / total)
            val_acc_.append(total_acc.item() / total)
            end_time = time.time()
            if total_acc/total > best_acc:
                best_model = model
                
            # if val_loss_[epoch]<best_loss:
            #     best_loss = val_loss_[epoch]
            #     best_model = model
            print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            model = best_model
            ## testing epoch
            results=[]
            y_pred = []
            y_true = []
        
            print(project,'--------------------')
            while i < len(test_data):
                batch = get_batch(test_data, i, BATCH_SIZE)
                i += BATCH_SIZE
                test_inputs, test_labels = batch
                if USE_GPU:
                    test_inputs, test_labels = test_inputs, test_labels.cuda()
                y_true+=test_labels.tolist()
                model.batch_size = len(test_labels)
                model.hidden = model.init_hidden()
                output = model(test_inputs)
                results+=output.tolist()

                loss = loss_function(output, Variable(test_labels))

                _, predicted = torch.max(output.data, 1)
                y_pred+=predicted.tolist()

                total_acc += (predicted == test_labels).sum()
                total += len(test_labels)
                total_loss += loss.item() * len(test_inputs)
        
            print("Testing results(Acc):", total_acc.item() / total)
            # 保持results , y_true
            testsaves = pd.DataFrame({'predict_result':results,'label':y_true}, columns=['predict_result','label'])
            
            path = 'result/predict_result/time/{}'.format(project)
            if not os.path.exists(path):
                os.mkdir(path)
            testsaves.to_csv(path+'/{}_count_{}_epoch_{}.csv'.format(project, str(h+1), str(epoch+1)))

    end_time2=time.time()
    timesaves=pd.DataFrame([[project, end_time2-start_time2]], columns=['task','traintime'])
    timesaves.to_csv(path+'/'+project+'_time.csv')

            
            
