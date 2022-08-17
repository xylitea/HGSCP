from itertools import count
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
import os

# projects=["ant1"]
# projects=["lucene_upsample50"]
# projects=['lucene3', 'lucene3_upsample43']
# projects=['ambari_upsample37']
projects=['ambari','ant', 'argouml', 'hibernate', 'jenkins', 'jmeter', 'lucene','poi']
# projects=['hibernate_upsample50', 'jenkins_upsample35', 'jmeter_upsample42', 'lucene3', 'poi']

def compute_prf(data):
    precision = precision_score(y_true=data["label"], y_pred=data["predict_result"])
    recall = recall_score(y_true=data["label"], y_pred=data["predict_result"])
    f1 = f1_score(y_true=data["label"], y_pred=data["predict_result"])
    acc=accuracy_score(y_true=data["label"], y_pred=data["predict_result"])
    auc=roc_auc_score(y_true=data["label"], y_score=data["predict_result"])
    mcc=matthews_corrcoef(y_true=data["label"], y_pred=data["predict_result"])
    return precision, recall, f1, acc, auc, mcc
count=30
if __name__ == '__main__':
    for project in projects:
        # metric_results=open('baseline/train_ast/result/{}.csv'.format(project), 'w', encoding='utf-8')
        # metric_results=open('result/metric_result/{}.csv'.format(project), 'w', encoding='utf-8')
        # metric_results=open('result/metric_result/edgeweight/{}.csv'.format(project), 'w', encoding='utf-8')
        # metric_results=open('result/metric_result/simplify/{}.csv'.format(project), 'w', encoding='utf-8')
        metric_results=open('result/metric_result/{}.csv'.format(project), 'w', encoding='utf-8')
        metric_results.write('task,Precision,Recall,F1,Acc,AUC,MCC\n')
        list_prediction=[]
        list_recall=[]
        list_f1=[]
        list_acc=[]
        list_auc=[]
        list_mcc=[]
        for i in range(count):
            max_precision,max_recall,max_f1,max_acc,max_auc,max_mcc = 0,0,0,0,0,0  # 取15轮中最好的结果
            for j in range(15):
                # file_name = 'baseline/train_ast/{}/{}_count_{}_epoch_{}.csv'.format(project, project, str(i+1), str(j+1))
                # file_name = 'result/predict_result/{}/{}_count_{}_epoch_{}.csv'.format(project, project, str(i+1), str(j+1))
                # file_name = 'result/predict_result/edgeweight/{}/{}_count_{}_epoch_{}.csv'.format(project, project, str(i+1), str(j+1))
                # file_name = 'result/predict_result/simplify/{}/{}_count_{}_epoch_{}.csv'.format(project, project, str(i+1), str(j+1))
                file_name = 'result/predict_result/{}/{}_count_{}_epoch_{}.csv'.format(project, project, str(i+1), str(j+1))
                predict_data = pd.read_csv(file_name)
                
                def output(output):
                    output = eval(output)
                    prediction = np.argmax(output) #, axis=1
                    return prediction

                predict_data['predict_result'] = predict_data['predict_result'].apply(output)
                precision, recall, f1, acc, auc, mcc=compute_prf(predict_data)
                print(project,precision, recall, f1, acc, auc, mcc)
                # if f1>max_f1:
                if j==14:
                    max_f1=f1
                    max_precision=precision
                    max_recall=recall
                    max_acc=acc
                    max_auc=auc
                    max_mcc=mcc
            
            metric_results.write("{},{},{},{},{},{},{}\n".format(project,max_precision,max_recall,max_f1,max_acc,max_auc,max_mcc))
            # print(project,max_precision,max_recall,max_f1,max_acc,max_auc,max_mcc)
            list_prediction.append(max_precision)
            list_recall.append(max_recall)
            list_f1.append(max_f1)
            list_acc.append(max_acc)
            list_auc.append(max_auc)
            list_mcc.append(max_mcc)
            metric_results.flush()
        metric_results.write("{},{},{},{},{},{},{}\n".format("average",
                                                            sum(list_prediction)/count,
                                                            sum(list_recall)/count,
                                                            sum(list_f1)/count,
                                                            sum(list_acc)/count,
                                                            sum(list_auc)/count,
                                                            sum(list_mcc)/count))
        metric_results.close()

