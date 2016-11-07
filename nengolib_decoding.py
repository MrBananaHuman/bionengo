'''Adapted from Aaron Voelker's 2d_oscillator_adaptation.ipynb
https://github.com/arvoelke/nengolib/tree/osc-adapt/doc/notebooks/research
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nengo
import nengolib
import json
import os
import ipdb
from neuron_methods import make_bioneuron, connect_bioneuron, run_bioneuron
from analyze import get_rates, make_tuning_curves, tuning_curve_loss

'''load previous optimized weights/locations into new bioneurons'''
root=os.getcwd()
datadir='data/8RCOQKGYQ/data/XUVLD1EGK/'
directory='/home/pduggins/bionengo/'+datadir
os.chdir(directory)
lifdata=np.load(directory+'lifdata.npz')
signal_in=lifdata['signal_in']
spikes_in=lifdata['spikes_in']
lif_eval_points=lifdata['lif_eval_points'].ravel()
lif_activities=lifdata['lif_activities']
f=open('filenames.txt','r')
filenames=json.load(f)
p_in=open('fake_params.txt','r')
P=json.load(p_in)
os.chdir(root)

biopop=[]
for bio_idx in range(P['n_bio']):
	#load weights, locations, biases
	with open(filenames[bio_idx],'r') as data_file: 
		bioneuron_info=json.load(data_file)
	bias=bioneuron_info['bias']
	weights=np.array(bioneuron_info['weights'])
	locations=np.array(bioneuron_info['locations'])
	#create neurons
	bioneuron = make_bioneuron(P,weights,locations,bias)
	connect_bioneuron(P,spikes_in,bioneuron)
	biopop.append(bioneuron)

'''test if this loading works by rerunning and replotting the tuning curves'''
run_bioneuron(P)
losses=[]
sns.set(context='poster')
figure1, ax1 = plt.subplots(1, 1)
ax1.set(xlabel='x',ylabel='firing rate (Hz)')
for bio_idx in range(P['n_bio']):
	spike_times=np.round(np.array(biopop[bio_idx].spikes),decimals=3)
	biospikes, biorates=get_rates(P,spike_times)
	bio_eval_points, bio_activities = make_tuning_curves(P,signal_in,biorates)
	X,f_bio_rate,f_lif_rate,loss=tuning_curve_loss(
			P,lif_eval_points,lif_activities[:,bio_idx],
			bio_eval_points,bio_activities)
	lifplot=ax1.plot(X,f_bio_rate(X),linestyle='-')
	bioplot=ax1.plot(X,f_lif_rate(X),linestyle='--',color=lifplot[0].get_color())
	losses.append(loss)
ax1.set(ylim=(0,60),title='total loss = %s'%np.sum(losses))
figure1.savefig('remade_biopop_tuning_curves.png')