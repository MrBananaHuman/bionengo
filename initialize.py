def make_addon(N):
	import string
	import random
	addon=str(''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(N)))
	return addon

def ch_dir():
	#change directory for data and plot outputs
	import os
	import sys
	root=os.getcwd()
	addon=make_addon(9)
	datadir=''
	if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
		datadir=root+'/data/'+addon+'/' #linux or mac
	elif sys.platform == "win32":
		datadir=root+'\\data\\'+addon+'\\' #windows
	os.makedirs(datadir)
	os.chdir(datadir) 
	return datadir

def make_signal(P):
	""" Returns: array indexed by t when called from a nengo Node"""
	import signals
	sP=P['signal']
	dt=P['dt']
	t_final=P['t_sample']+dt #why is this extra step necessary?
	raw_signal=None
	if sP['type']=='constant':
		raw_signal=signals.constant(dt,t_final,sP['value'])
	if sP['type']=='white':
		raw_signal=signals.white(dt,t_final)
	elif sP['type']=='white_binary':
		raw_signal=signals.white_binary(dt,t_final,sP['mean'],sP['std'])
	elif sP['type']=='switch':
		raw_signal=signals.switch(dt,t_final,sP['max_freq'],)
	elif sP['type']=='equalpower':
		raw_signal=signals.equalpower(
			dt,t_final,sP['max_freq'],sP['mean'],sP['std'])
	elif sP['type']=='poisson_binary':
		raw_signal=signals.poisson_binary(
			dt,t_final,sP['mean_freq'],sP['max_freq'],sP['low'],sP['high'])
	elif sP['type']=='poisson':
		raw_signal=signals.poisson(
			dt,t_final,sP['mean_freq'],sP['max_freq'])
	elif sP['type']=='pink_noise':
		raw_signal=signals.pink_noise(
			dt,t_final,sP['mean'],sP['std'])
	assert raw_signal is not None, "signal type not specified"
	return raw_signal

def make_spikes_in(P,raw_signal):
	import nengo
	import numpy as np
	import pandas as pd
	import json
	spikes_in=[]
	lifdata={}
	while np.sum(spikes_in)==0: #rerun nengo spike generator until it returns something
		with nengo.Network() as model:
			signal = nengo.Node(
					output=lambda t: raw_signal[int(t/P['dt'])])
			ideal = nengo.Ensemble(1,
					dimensions=1,
					max_rates=nengo.dists.Uniform(80,120)) #ideal tuning curve has limited rate
			ens_in = nengo.Ensemble(P['n_lif'],
					dimensions=1,
					max_rates=nengo.dists.Uniform(P['min_lif_rate'],P['max_lif_rate']))		
			nengo.Connection(signal,ens_in)
			probe_signal = nengo.Probe(signal)
			probe_in = nengo.Probe(ens_in.neurons,'spikes')
		with nengo.Simulator(model,dt=P['dt']) as sim:
			sim.run(P['t_sample'])
			eval_points, activities = nengo.utils.ensemble.tuning_curves(ideal,sim)
		signal_in=sim.data[probe_signal]
		spikes_in=sim.data[probe_in]
	lifdata['signal_in']=signal_in.ravel()
	lifdata['spikes_in']=spikes_in
	lifdata['lif_eval_points']=eval_points.ravel()
	lifdata['lif_activities']=activities.ravel()
	out_data=pd.DataFrame([lifdata])
	out_data.reset_index().to_json(P['directory']+'lifdata.json',orient='records')
	return lifdata

def find_w_max(P,lifdata):
	import numpy as np
	from analyze import get_rates
	spike_train, summed_rates=get_rates(P,lifdata['spikes_in'])
	rate_max=np.amax(summed_rates)
	w_max=175.0/rate_max*0.01 #175 hz input from 0.01 max weight in response curve
	return w_max

def add_search_space(P,w_max):
	#adds a hyperopt-distributed weight, location, bias for each synapse
	import numpy as np
	import hyperopt
	P['bias']=hyperopt.hp.uniform('b',P['bias_min'],P['bias_max'])
	P['weights']={}
	P['locations']={}
	for n in range(P['n_lif']):
		for i in range(P['n_syn']): 
			P['weights']['%s_%s'%(n,i)]=hyperopt.hp.uniform('w_%s_%s'%(n,i),-1.0*w_max,1.0*w_max)
			if P['synapse_dist'] == 'optimized': 
				P['locations']['%s_%s'%(n,i)]=hyperopt.hp.uniform('l_%s_%s'%(n,i),0,1)	
			elif P['synapse_dist'] == 'soma': 
				P['locations']['%s_%s'%(n,i)]=0.5
	return P