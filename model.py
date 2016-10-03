'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''

import nengo
import numpy as np
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hyperopt
import json
import ipdb

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
		datadir=root+'/data/'+addon #linux or mac
	elif sys.platform == "win32":
		datadir=root+'\\data\\'+addon #windows
	os.makedirs(datadir)
	os.chdir(datadir) 
	return datadir

def make_search_space(P):
	space={'P':P,'weights':{},'locations':{},}
	n_syn=P['synapses_per_connection']
	for i in range(n_syn): #adds a hyperopt-distributed weight and location for each synapse
		space['weights'][i]=hyperopt.hp.quniform('w'+str(i),P['weight_min'],P['weight_max'],0.000001)
		space['locations'][i]=hyperopt.hp.uniform('l'+str(i),0,1)
	return space

def make_signal(P):
	""" Returns: array indexed by t when called from a nengo Node"""
	import signals
	sP=P['signal']
	dt=P['dt']
	t_final=P['t_sample']+dt #why is this extra step necessary?
	raw_signal=None
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
	with nengo.Network() as model:
		signal = nengo.Node(
				output=lambda t: raw_signal[int(t/P['dt'])])
		ideal = nengo.Ensemble(1,
				dimensions=1,
				max_rates=nengo.dists.Uniform(P['min_LIF_rate'],P['max_LIF_rate']))
		ens_in = nengo.Ensemble(1,
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
	return spikes_in.ravel(), signal_in.ravel(), eval_points.ravel(), activities.ravel()

def make_synapses(P,bioneuron,weights,loc):
	locations=None
	if P['synapse_dist'] == 'random':
		locations=np.random.uniform(0,1,size=len(bioneuron.synapses))
	elif P['synapse_dist'] == 'linear':
		locations=np.arange(0,1,1.0/len(bioneuron.synapses))
	elif P['synapse_dist'] == 'optimized':
		locations=loc
	for i in range(len(locations)):
		if P['biosynapse']['type'] == 'ExpSyn':
			bioneuron.add_ExpSyn(
				bioneuron.cell.apical(locations[i]),weights[i],P['biosynapse']['tau'])
		elif P['biosynapse']['type'] == 'AlphaSyn':
			bioneuron.add_AlphaSyn(
				bioneuron.cell.apical(locations[i]),weights[i],P['biosynapse']['tau'])

def make_bioneuron(P,weights,locations):
	from neurons import Bahl
	bioneuron=Bahl()
	make_synapses(P,bioneuron,weights,locations)
	bioneuron.start_recording()
	return bioneuron

def run_neuron(P,LIFdata,bioneuron):
	import sys
	neuron.init()
	for t in P['timesteps']: 
		sys.stdout.write("\r%d%%" %(1+100*t/(P['t_sample'])))
		sys.stdout.flush()
		if np.any(np.asfarray(np.where(LIFdata['spikes_in'])[0]) == t/P['dt']):
			# print 'input spike at t=%s, idx=%s' %(t*1000, t/P['dt'])
			for syn in bioneuron.synapses:
				syn.conn.event(t*1000)
		neuron.run(t*1000)

def get_rates(P,bioneuron,LIFdata,addon):
	timesteps=P['timesteps']
	spike_times=np.round(np.array(bioneuron.spikes),decimals=3)
	spike_train=np.zeros_like(timesteps)
	if spike_times.shape[0] >= 20000: ipdb.set_trace()
	for idx in spike_times/P['dt']/1000: spike_train[idx]=1.0
	rates=np.zeros_like(spike_train)
	if P['kernel']['type'] == 'exp':
		kernel = np.exp(-timesteps/P['kernel']['tau'])
	elif P['kernel']['type'] == 'gauss':
		kernel = np.exp(-timesteps**2/(2*P['kernel']['sigma']**2))
	elif P['kernel']['type'] == 'alpha':  
		kernel = (timesteps / P['kernel']['tau']) * np.exp(-timesteps / P['kernel']['tau'])
	# kernel /= kernel.sum()
	rates = np.convolve(kernel, spike_train, mode='full')[:len(timesteps)]

	sns.set(context='poster')
	figure, (ax1,ax2) = plt.subplots(2,1)
	# figure, (ax1,ax2,ax3) = plt.subplots(3,1)
	ax1.plot(np.array(bioneuron.t_record)/1000, np.array(bioneuron.v_record))
	ax1.set(xlabel='time', ylabel='voltage (mV)')
	ax2.plot(timesteps,LIFdata['signal_in'],label='input signal')
	ax2.plot(timesteps,LIFdata['spikes_in'],label='input spikes')
	# ax2.plot(timesteps,spike_train,label='output spikes')
	ax2.plot(timesteps,rates,label='output rate')
	ax2.set(xlabel='time (s)')
	# ax3.plot(timesteps,kernel,label='kernel')
	# ax3.set(xlabel='time (s)', ylabel='filter value')
	plt.legend()
	figure.savefig('spikes_' + addon +'.png')
	return rates

def make_tuning_curves(P,LIFdata,rates):
	X=np.arange(np.min(LIFdata['signal_in']),np.max(LIFdata['signal_in']),P['dx']) #eval points in X
	Hz=np.zeros_like(X) #firing rate for each eval point
	for xi in range(len(X)-1):
		ts=[] #find the time indices where the signal is between this and the next evalpoint
		for ti in range(len(P['timesteps'])):
			if X[xi] < LIFdata['signal_in'][ti] < X[xi+1]:
				ts.append(ti)
		if len(ts)>0:
			#average the firing rate at each of these time indices
			Hz[xi]=np.average([rates[ti] for ti in ts])
			#convert units to Hz by dividing by the time window
			Hz[xi]=Hz[xi]/len(ts)/P['dt']
	return X, Hz

def calculate_loss(P,LIFdata,X_NEURON,Hz_NEURON,space,addon):
	#shape of activities and Hz is mismatched, so interpolate and slice activities for comparison
	from scipy.interpolate import interp1d
	f_NEURON_rate = interp1d(X_NEURON,Hz_NEURON)
	f_LIF_rate = interp1d(np.array(LIFdata['X_LIF']),np.array(LIFdata['Hz_LIF']))
	x_min=np.maximum(LIFdata['X_LIF'][0],X_NEURON[0])
	x_max=np.minimum(LIFdata['X_LIF'][-1],X_NEURON[-1])
	X=np.arange(x_min,x_max,P['dx'])
	loss=np.sqrt(np.average((f_NEURON_rate(X)-f_LIF_rate(X))**2))
	sns.set(context='poster')

	figure, ax1 = plt.subplots(1,1)
	ax1.plot(X,f_NEURON_rate(X),label='bioneuron firing rate (Hz)')
	ax1.plot(X,f_LIF_rate(X),label='LIF firing rate (Hz)')
	ax1.set(xlabel='x',ylabel='firing rate (Hz)',title='loss=%0.3f' %loss)
	plt.legend()
	figure.savefig('%0.3f_'%loss + addon +'.png')
	plt.close(figure)
	my_params=pd.DataFrame([space])
	my_params.reset_index().to_json('parameters_' + addon + '%0.3f_'%loss + '.json',orient='records')

	return loss

def plot_loss(trials):
	sns.set(context='poster')
	figure1, ax1 = plt.subplots(1, 1)
	X=[t['tid'] for t in trials]
	Y=[t['result']['loss'] for t in trials]
	ax1.scatter(X,Y)
	ax1.set(xlabel='$t$',ylabel='loss')
	figure1.savefig('hyperopt_result.png')

'''main---------------------------------------------------'''


def simulate(space):
	P=space['P']
	P['timesteps']=np.arange(0,P['t_sample'],P['dt'])
	n_syn=P['synapses_per_connection']
	with open("LIFdata.json") as data_file:    
		LIFdata = json.load(data_file)[0]
	weights,locations=np.zeros(n_syn),np.zeros(n_syn)
	for i in range(n_syn):
		weights[i]=space['weights'][i]
		locations[i]=space['locations'][i]
	bioneuron = make_bioneuron(P,weights,locations)
	print '\nRunning NEURON'
	run_neuron(P,LIFdata,bioneuron)
	addon=make_addon(6)
	rates=get_rates(P,bioneuron,LIFdata,addon)
	X_NEURON, Hz_NEURON = make_tuning_curves(P,LIFdata,rates)
	loss=calculate_loss(P,LIFdata,X_NEURON,Hz_NEURON,space,addon)
	return {'loss': loss, 'status': hyperopt.STATUS_OK}

def main():
	P=eval(open('parameters.txt').read())
	datadir=ch_dir()
	print 'Generating input spikes ...'
	raw_signal=make_signal(P)
	spikes_in=[]
	LIFdata={}
	while np.sum(spikes_in)==0: #rerun nengo spike generator until it returns something
		spikes_in, signal_in, X_LIF, Hz_LIF = make_spikes_in(P,raw_signal)
	LIFdata['signal_in']=signal_in
	LIFdata['spikes_in']=spikes_in*P['dt']
	LIFdata['X_LIF']=X_LIF
	LIFdata['Hz_LIF']=Hz_LIF
	out_data=pd.DataFrame([LIFdata])
	out_data.reset_index().to_json('LIFdata.json',orient='records')

	if P['optimization']=='hyperopt':
		search_space=make_search_space(P)
		trials=hyperopt.Trials()
		best=hyperopt.fmin(simulate,space=search_space,algo=hyperopt.tpe.suggest,
							max_evals=P['max_evals'],trials=trials)
		print best
		plot_loss(trials)

if __name__=='__main__':
	main()