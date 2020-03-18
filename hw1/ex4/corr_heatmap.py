import numpy as np
import pandas as pd

data=pd.read_csv('train.csv')
data.drop(['a','b'],axis=1,inplace=True)
column=data['c'].unique()
data_new=pd.DataFrame(np.zeros([24*240,18]),columns=column)
for i in column:
    aa=data[data['c']==i]
    aa.drop(['c'],axis=1,inplace=True)
    aa=np.array(aa)
    aa[aa=='NR']='0'
    aa=aa.astype('float32')
    aa=aa.reshape(1,5760)
    aa=aa.T
    data_new[i]=aa
label=np.array(data_new['PM2.5'][9:],dtype='float32')

import matplotlib.pyplot as plt
import seaborn as sns
# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(abs(data_new.corr()), fmt="d", linewidths=.5, ax=ax, cmap='Blues')
f.savefig('heatmap.png')

"""————————————————
版权声明：本文为CSDN博主「yinfang1252」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/yinfang1252/article/details/79630222"""

