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

def make_spikes_in(P,raw_signal,datadir):
	import nengo
	import numpy as np
	import pandas as pd
	import json
	spikes_in=[]
	LIFdata={}
	while np.sum(spikes_in)==0: #rerun nengo spike generator until it returns something
		with nengo.Network() as model:
			signal = nengo.Node(
					output=lambda t: raw_signal[int(t/P['dt'])])
			ideal = nengo.Ensemble(1,
					dimensions=1,
					max_rates=nengo.dists.Uniform(P['min_LIF_rate'],P['max_LIF_rate']))
			ens_in = nengo.Ensemble(P['n_LIF'],
					dimensions=1,
					max_rates=nengo.dists.Uniform(P['min_LIF_rate'],P['max_LIF_rate']))		
			nengo.Connection(signal,ens_in)
			probe_signal = nengo.Probe(signal)
			probe_in = nengo.Probe(ens_in.neurons,'spikes')
		with nengo.Simulator(model,dt=P['dt']) as sim:
			sim.run(P['t_sample'])
			eval_points, activities = nengo.utils.ensemble.tuning_curves(ideal,sim)
		signal_in=sim.data[probe_signal]
		spikes_in=sim.data[probe_in]
	LIFdata['signal_in']=signal_in.ravel()
	LIFdata['spikes_in']=spikes_in
	LIFdata['lif_eval_points']=eval_points.ravel()
	LIFdata['lif_activities']=activities.ravel()
	out_data=pd.DataFrame([LIFdata])
	out_data.reset_index().to_json(datadir+'LIFdata.json',orient='records')
	return LIFdata

def find_w_max(P,LIFdata):
	import numpy as np
	from analyze import get_rates
	summed_rates=get_rates(P,LIFdata['spikes_in'])
	rate_max=np.amax(summed_rates)
	w_max=175.0/rate_max*0.01*200 #175 hz input from 0.01 max weight in response curve
	print summed_rates,w_max
	return w_max