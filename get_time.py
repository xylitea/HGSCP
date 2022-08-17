import pandas as pd

def getexcelresult():
    projects=['ambari_upsample37','ant1_upsample39','argouml', 'hibernate_upsample50', 'jenkins_upsample35', 'jmeter_upsample42', 'lucene3_upsample33', 'poi_upsample41']
    # projects=['ambari_upsample37']
    metrics=['parsetime','graphbuildtime','datasettime','traintime','testtime']

    metric_results=open('result/predict_result/simplifyedge/time/time.csv', 'w', encoding='utf-8')
    # metric_results=open('result/predict_result/simplify/time/time.csv', 'w', encoding='utf-8')
    # metric_results=open('result/predict_result/ast-ggnn/multi/time.csv', 'w', encoding='utf-8')
    # metric_results=open('origast/time.csv', 'w', encoding='utf-8')
    metric_results.write('task,parsetime,graphbuildtime,datasettime,traintime,testtime\n')
    
    for project in projects:
        file_name='result/predict_result/simplifyedge/time/{}/times.csv'.format(project)
        # file_name='result/predict_result/simplify/time/{}/times.csv'.format(project)
        # file_name='result/predict_result/ast-ggnn/multi/{}/times.csv'.format(project)
        # file_name='origast/time/{}/time.csv'.format(project)
        data = pd.read_csv(file_name)
        metric_value=[]
        for metric in metrics:
            metric_value.append(data.iloc[-1][metric])
        metric_results.write("{},{},{},{},{},{}\n".format(str(project),metric_value[0],metric_value[1],metric_value[2],metric_value[3],metric_value[4]))
    metric_results.close()


getexcelresult()