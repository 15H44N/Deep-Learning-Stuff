import os
import pandas as pd
data = os.listdir('D:/Datasets/dogs-vs-cats/train')

info_list = []
for idx,fname in enumerate(data):
    target = data[idx].split(".")[0]
    target_num = 1 if target == 'dog' else 0
    info = { "filename":fname, 'target':target_num }
    info_list.append(info)
    
pd.DataFrame(info_list).to_csv('D:/Datasets/dogs-vs-cats/dogs-vs-cats.csv',index=False)