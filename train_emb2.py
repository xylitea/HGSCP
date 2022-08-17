import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import os
import argparse
from tqdm import tqdm, trange
import pycparser
# from createclone_java import createast,createggnndata,createseparategraph
from createsimplify_java_emb import createast,createggnndata,createseparategraph
import models
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import sys
sys.setrecursionlimit(9999)
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default='gcj')
parser.add_argument("--graphmode", default='astonly') # astandnext
parser.add_argument("--simplify", default=True)
parser.add_argument("--nextsib", default=False)
parser.add_argument("--blockedge", default=False)

parser.add_argument("--nexttoken", default=False)

parser.add_argument("--ifedge", default=False)
parser.add_argument("--whileedge", default=False)
parser.add_argument("--foredge", default=False)

parser.add_argument("--nextuse", default=False)

parser.add_argument("--data_setting", default='0')
parser.add_argument("--batch_size", default=32)   # 32
parser.add_argument("--num_layers", default=4)  #8
parser.add_argument("--num_epochs", default=15)
parser.add_argument("--counts", default=3)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()

def create_batches(data):
    #random.shuffle(data)
    batches = [data[graph:graph+args.batch_size] for graph in range(0, len(data), args.batch_size)]
    return batches

def test(dataset):
    #model.eval()
    count=0
    correct=0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results=[]
    y_pred=[]
    y_true=[]
    for data,label in dataset:
        label=torch.tensor(label, dtype=torch.long, device=device)
        x,edge_index, edge_attr=data
        x=torch.tensor(x, dtype=torch.float32, device=device)
        edge_index=torch.tensor(edge_index, dtype=torch.long, device=device)
        if edge_attr!=None:
            edge_attr=torch.tensor(edge_attr, dtype=torch.long, device=device)
        data=[x, edge_index, edge_attr]
        output=model(data)
        results.append(output.tolist())
        # prediction = torch.sign(output).item()
        prediction = torch.argmax(output, dim=1).item()
        y_pred.append(prediction)
        y_true.append(label.item())
        if prediction==label.item() and label.item()==1:
            tp+=1
            #print('tp')
        if prediction==label.item() and label.item()==0:
            tn+=1
            #print('tn')
        if prediction!=label.item() and label.item()==0:
            fp+=1
            #print('fp')
        if prediction!=label.item() and label.item()==1:
            fn+=1
            #print('fn')
    print(tp,tn,fp,fn)
    p=0.0
    r=0.0
    f1=0.0
    if tp+fp==0:
        print('precision is none')
        return
    p=tp/(tp+fp)
    if tp+fn==0:
        print('recall is none')
        return
    r=tp/(tp+fn)
    f1=2*p*r/(p+r)
    print('precision',p)
    print('recall',r)
    print('F1',f1)
    acc=(tp+tn)/(tp+fp+tn+fn)
    print('accuracy',acc)
    mcc=(tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
    print('mcc',mcc)
    auc=roc_auc_score(y_true,y_pred)
    print('auc',auc)
    return results, y_true
    
device=torch.device('cuda:0')
# device=torch.device('cpu')
# projects=['argouml','hibernate_upsample50', 'jenkins_upsample35', 'jmeter_upsample42', 'lucene3_upsample33', 'poi_upsample41'] #'ambari_upsample37' 'ant1 'argouml2', 'hibernate_upsample50', 'jenkins_upsample35', 'jmeter_upsample_42', 'lucene3', 'poi'
projects=['ambari_upsample37','ant1_upsample39']
# 效果差'ambari_upsample37','ant1_upsample39', 学习率 0.01不行
for project in projects:
    for count in range(args.counts):
        astdict, vocablen, vocabdict, labeldict, corpus=createast(project)

        id_vector=dict()
        # 训练词向量
        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(sentences=corpus, size=100, workers=16, sg=1, min_count=0)
        for word, id in vocabdict.items():
            vector = w2v.wv.__getitem__(word)
            id_vector[id]=vector

        sample=len(astdict)
        print('sample = ', sample)
        treedict=createseparategraph(astdict, vocablen, vocabdict,id_vector,device,mode=args.graphmode,simplify=args.simplify,nextsib=args.nextsib,ifedge=args.ifedge,whileedge=args.whileedge,foredge=args.foredge,blockedge=args.blockedge,nexttoken=args.nexttoken,nextuse=args.nextuse)
        traindata,testdata=createggnndata(treedict,labeldict,device)
        

        num_layers=int(args.num_layers)
        model=models.GGNN2(vocablen,embedding_dim=100,num_layers=num_layers,device=device).to(device)
        # model=models.GCN(vocablen,embedding_dim=100,num_class=2,device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # weight = torch.tensor([0.1, 1], dtype=torch.float, device=device)
        # criterion=nn.CrossEntropyLoss(weight=weight)
        criterion=nn.CrossEntropyLoss()
        
        epochs = trange(args.num_epochs, leave=True, desc = "Epoch")
        for epoch in epochs:# without batching
            print(epoch)
            batches=create_batches(traindata)
            totalloss=0.0
            main_index=0.0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
                # print('batch is ', batch)
                optimizer.zero_grad()
                batchloss= 0
                for data,label in batch:
                    label=torch.tensor([label], dtype=torch.long, device=device)
                    # print('label ',label)
                    #print(len(data))
                    #for i in range(len(data)):
                        #print(i)
                        #data[i]=torch.tensor(data[i], dtype=torch.long, device=device)
                    x, edge_index, edge_attr=data
                    x=torch.tensor(x, dtype=torch.float32, device=device)
                    edge_index=torch.tensor(edge_index, dtype=torch.long, device=device)
                    if edge_attr!=None:
                        edge_attr=torch.tensor(edge_attr, dtype=torch.long, device=device)
                    data=[x, edge_index, edge_attr]
                    output=model(data)
                    batchloss=batchloss+criterion(output,Variable(label))
                batchloss.backward(retain_graph=True)
                optimizer.step()
                loss = batchloss.item()
                totalloss+=loss
                main_index = main_index + len(batch)
                loss=totalloss/main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss,5))
             
            # devresults,_=test(validdata)
            # devfile=open('result/'+args.graphmode+'_dev_epoch_'+str(epoch+1),mode='w')
            # for res in devresults:
            #     devfile.write(str(res)+'\n')
            # devfile.close()

            # model.eval()
            testresults,testlabels=test(testdata)
            testsaves = pd.DataFrame({'predict_result':testresults,'label':testlabels}, columns=['predict_result','label'])
            ##

            path = 'result/predict_result/embed/simplify/{}'.format(project)
            if not os.path.exists(path):
                os.mkdir(path)
            testsaves.to_csv(path+'/{}_count_{}_epoch_{}.csv'.format(project, str(count+1), str(epoch+1)))
            
            #torch.save(model,'models/gmngcj'+str(epoch+1))
            #for start in range(0, len(traindata), args.batch_size):
                #batch = traindata[start:start+args.batch_size]
                #epochs.set_description("Epoch (Loss=%g)" % round(loss,5))




