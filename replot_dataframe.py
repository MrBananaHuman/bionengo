import ipdb
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

df=pd.read_pickle('/home/pduggins/bionengo/data/SMPKKWJ16/loss_vs_synapse_type_dataframe.pkl')
# ipdb.set_trace()
param='synapse_type'
n_avg=5
sns.set(context='poster')
figure1, (ax1,ax2) = plt.subplots(2, 1)
# sns.regplot(x='param',y='loss',data=df,x_jitter=0.05)
# sns.tsplot(time='param',value='loss',unit='trial',data=df,ax=ax1)
# sns.tsplot(time='param',value='runtime',unit='trial',data=df,ax=ax2)
sns.boxplot(x='param',y='loss',data=df,ax=ax1)
sns.boxplot(x='param',y='runtime',data=df,ax=ax2)
ax1.set(ylabel='mean loss',title='N=%s'%n_avg)
ax2.set(xlabel=param,ylabel='mean runtime (s)')
plt.savefig('loss_vs_%s.png'%param)