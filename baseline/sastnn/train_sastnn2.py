import os
import torch
import copy
from torch.utils.data import DataLoader
import torch.utils.data as Data
import rnn as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
# import openpyxl
import pipeline

# def write_excel_xls_hypoth(path, sheet_name, value):
#     if os.path.exists(path):
#         data = openpyxl.load_workbook(path)
#         table = data.create_sheet(sheet_name)
#         table.title = sheet_name
#         index = len(value)  # 获取需要写入数据的行数
#         for i in range(0, index):
#             for j in range(0, len(value[i])):
#                 table.cell(row=1 + i, column=j + 1, value=str(value[i][j]))  # 像表格中写入数据（对应的行和列）
#         data.save(path)  # 保存工作簿
#     else:
#         index = len(value)
#         workbook = openpyxl.Workbook()  # 新建工作簿（默认有一个sheet？）
#         sheet = workbook.active  # 获得当前活跃的工作页，默认为第一个工作页
#         sheet.title = sheet_name  # 给sheet页的title赋值
#         for i in range(0, index):
#             for j in range(0, len(value[i])):
#                 sheet.cell(row=i + 1, column=j + 1, value=str(value[i][j]))  # 行，列，值 这里是从1开始计数的
#         workbook.save(path)  # 一定要保存
#     print("xlsx格式表格写入数据成功！")

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def handel_data(train_x):
    X = []
    train_x = train_x.tolist()
    n_x = len(train_x)
    max_lens = 0
    for i in range(n_x):
        length = len(train_x[i])
        if length>max_lens:
            max_lens = length

    for i in range(n_x):
        row = []
        for _ in range(max_lens - len(train_x[i])):
            row.append([0.0]*128)
        for vc in train_x[i]:
            row.append(vc)
        X.append(row)
    X= np.array(X)
    X = torch.from_numpy(X)
    
    return X, max_lens

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

## parameter setting
epochs = 10
# epochs = 5
batch_size = 32
use_gpu = torch.cuda.is_available()
learning_rate = 0.01

# projects=['lucene3_upsample33']
projects=['ambari_upsample37','ant1_upsample39', 'argouml', 'hibernate_upsample50', 'jenkins_upsample35', 'jmeter_upsample42', 'lucene3_upsample33', 'poi_upsample41']

# projects = ["ant1"]   
data = []
data_avg = []
# count = 3
count = 1
alltimes=[]
for project in projects:
    print('*********************', project, '*********************')
    start_time=time.time()
    for h in range(count):
        print('*********************', '第',str(h), '轮', '*********************')

        embedding_dim = 128
        hidden_dim = 100
        nlabel = 2

        model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim,hidden_dim=hidden_dim,label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
        if use_gpu:
            model = model.cuda()

        ppl = pipeline.Pipeline('data/programs_'+project+'.pkl')
        train,test=ppl.run()

        train_y = train['label'].values.astype(int)
        train_x, train_y = train['code'].values, torch.from_numpy(train_y)
        train_x, train_max_lens = handel_data(train_x)
        train_data = Data.TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        test_y = test['label'].values.astype(int)
        test_x, test_y = test['code'].values, torch.from_numpy(test_y)
        test_x, max_lens = handel_data(test_x)
        test_data = Data.TensorDataset(test_x, test_y)
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # parameters = model.parameters()
        # optimizer = torch.optim.Adamax(parameters)
        loss_function = nn.CrossEntropyLoss()

        print('Start training...')
        # training procedure
        best_model = model
        for epoch in range(epochs):

            total_acc = 0.0
            total_loss = 0.0
            total = 0.0

            optimizer = adjust_learning_rate(optimizer, epoch)

            for iter, traindata in enumerate(train_loader):
                train_inputs, train_labels = traindata
                
                if use_gpu:
                    train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
                else: 
                    train_inputs = Variable(train_inputs)

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                train_inputs = train_inputs.float()
                output = model(train_inputs)
                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()

                # calc training acc
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == train_labels).sum()
                total += len(train_labels)
                total_loss += loss.item()*len(train_inputs)
        
            ## testing epoch
            results=[]
            y_pred=[]
            y_true=[]
            print(project,'--------------------')
            for iter, testdata in enumerate(test_loader):
                test_inputs, test_labels = testdata
                if use_gpu:
                    test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
                else: 
                    test_inputs = Variable(test_inputs)
                y_true+=test_labels.tolist()
                # y_true.append(test_labels.tolist())
                model.batch_size = len(test_labels)
                model.hidden = model.init_hidden()
                test_inputs = test_inputs.float()
                output = model(test_inputs)
                results+=output.tolist()
                # results.append(output.tolist())
                
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
    end_time=time.time()
    alltimes.append([project, end_time-start_time])

timesaves=pd.DataFrame(alltimes, columns=['task','alltime'])
timesaves.to_csv('result/predict_result/time/times.csv')

            
            

            
    


