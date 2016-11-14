'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''

from model import simulate
import ipdb
from initialize import ch_dir, make_signal, make_spikes_in, add_search_space
from run_hyperopt import run_hyperopt
from analyze import plot_final_tuning_curves
from pathos.multiprocessing import ProcessingPool as Pool
import copy
import timeit
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

P=eval(open('parameters.txt').read())
upper_datadir=ch_dir()
n_avg=5
param='bias_max'
sweep_param=[-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5]
columns=('trial','param','loss','runtime')
df=pd.DataFrame(columns=columns)

for v in range(len(sweep_param)):
	print '%s=%s'%(param,sweep_param[v])
	P[param]=sweep_param[v]
	for n in range(n_avg):
		start=timeit.default_timer()
		os.chdir(upper_datadir)
		print 'n=%s'%n
		lower_datadir=ch_dir()
		P['directory']=lower_datadir
		raw_signal=make_signal(P)
		make_spikes_in(P,raw_signal)
		P_list=[]
		for bio_idx in range(P['n_bio']):
			P_idx=add_search_space(P,bio_idx)
			# run_hyperopt(P_idx)
			P_list.append(copy.copy(P_idx))
		pool = Pool(nodes=P['n_processes'])
		filenames=pool.map(run_hyperopt, P_list) #multithread
		# filenames=[run_hyperopt(P_idx) for P_idx in P_list] #single thread
		with open('filenames.txt','wb') as outfile:
			json.dump(filenames,outfile)
		# with open('params.txt','wb') as param_outfile:
		# 	json.dump(P,param_outfile)
		loss=plot_final_tuning_curves(P,filenames)
		stop=timeit.default_timer()
		runtime=stop-start
		print 'runtime=%s'%runtime
		df=df.append(pd.DataFrame([[n,sweep_param[v],loss,runtime]],columns=columns),ignore_index=True)

os.chdir(upper_datadir)
df.to_pickle('loss_vs_%s_dataframe.pkl'%param)
# sns.set(context='poster')
figure1, (ax1,ax2) = plt.subplots(2, 1,sharex=True)
# sns.regplot(x='param',y='loss',data=df,x_jitter=0.05)
sns.tsplot(time='param',value='loss',unit='trial',data=df,ax=ax1)
sns.tsplot(time='param',value='runtime',unit='trial',data=df,ax=ax2)
# sns.boxplot(x='param',y='loss',data=df,ax=ax1)
# sns.boxplot(x='param',y='runtime',data=df,ax=ax2)
ax1.set(ylabel='mean loss',title='N=%s'%n_avg)
ax2.set(xlabel=param,ylabel='mean runtime (s)')
plt.savefig('loss_vs_%s.png'%param)