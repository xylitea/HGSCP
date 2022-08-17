from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import random

from torch.utils import data

def subsampling(project):
    data = pd.read_pickle("programs_{}.pkl".format(project))
    print(data.shape)
    positive_data = pd.DataFrame(columns=['id', 'code', 'label'])
    for i in range(len(data)):
        row = data.iloc[i]
        if row['label'] == 1:
            positive_data = positive_data.append(
                pd.DataFrame({"id": [row['id']], "code": [row['code']], "label": [row['label']]}),
                ignore_index=True)
    # print(positive_data.shape)
    labels = data['label'].tolist()
    print('before upsample ratio: ', sum(labels)/len(labels))
    positive_num = sum(labels)
    nums = len(labels)
    change_ratio = 0.33
    sample_num = len(positive_data)
    while positive_num/nums < change_ratio:
        i = int(random.uniform(0, sample_num-1))
        data = data.append(positive_data.iloc[i])
        positive_num+=1
        nums+=1
    data = data.sample(frac=1, random_state=random.randint(1, 100))
    # print(data)
    labels = data['label'].tolist()
    print('after upsample ratio: ', sum(labels)/len(labels))
    data.to_pickle("programs_{}_upsample{}.pkl".format(project, str(int(change_ratio*100))))

project = "lucene3"
subsampling(project)

