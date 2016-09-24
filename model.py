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

def make_search_space(P):
	space={'P':P,}
	for i in range(P['n_synapses']): #add a quniform weight for each synapse
		space[i]=hyperopt.hp.quniform('weights[%s]'%i,P['weight_min'],P['weight_max'],0.00001)
	return space

def make_spikes_in(P):
	with nengo.Network() as model:
		signal = nengo.Node(
				output=lambda t: np.sin(2*np.pi*t/P['t_sample']))
				#rms=radius ensures that all max/min of signal bound max/min of eval_points?
				# output=nengo.processes.WhiteSignal(P['t_sample'],high=40,rms=P['radius']))
		ens_in = nengo.Ensemble(1,
				dimensions=1,
				# radius=P['radius'],
				max_rates=nengo.dists.Uniform(P['min_LIF_rate'],P['max_LIF_rate']))
		probe_signal = nengo.Probe(signal) #get spikes from one neuron (one tuning curve)
		probe_in = nengo.Probe(ens_in.neurons,'spikes')
	with nengo.Simulator(model,dt=P['dt']) as sim:
		sim.run(P['t_sample'])
		eval_points, activities = nengo.utils.ensemble.tuning_curves(ens_in,sim)
	signal_in=sim.data[probe_signal]
	spikes_in=sim.data[probe_in]
	#return neuron's spike raster and neuron tuning curve and tuning curve info
	return spikes_in.ravel(), signal_in.ravel(), eval_points.ravel(), activities.ravel()

def make_synapses(P,bioneuron,weights):
	dist=None
	if P['synapse_dist'] == 'random':
		dist=np.random.uniform(0,1,size=P['n_synapses'])
	for i in dist:
		bioneuron.add_synapse(bioneuron.cell.apical(i),weights[i],P['tau'])

def make_bioneuron(P,weights):
	from neurons import Bahl
	bioneuron=Bahl()
	make_synapses(P,bioneuron,weights)
	bioneuron.start_recording()
	return bioneuron

def run_neuron(P,LIFdata,bioneuron):
	import sys
	neuron.init()
	for t in P['timesteps']: 
		sys.stdout.write("\r%d%%" %(100*t/(P['t_sample'])))
		sys.stdout.flush()
		if np.any(np.asfarray(np.where(LIFdata['spikes_in'])[0]) == t/P['dt']):
			# print 'input spike at t=%s, idx=%s' %(t*1000, t/P['dt'])
			for syn in bioneuron.synapses:
				syn.conn.event(t*1000)
		neuron.run(t*1000)

def plot_signals_spikes(P,LIFData,bioneuron,h,spikes_out):
	sns.set(context='poster')
	figure, (ax1,ax2,ax3) = plt.subplots(3,1)
	ax1.plot(np.array(bioneuron.t_record), np.array(bioneuron.v_record))
	ax1.set(xlabel='time (ms)', ylabel='voltage (mV)')
	ax2.plot(P['timesteps'],h,label='filter')
	ax2.set(xlabel='time (s)', ylabel='filter value')
	ax3.plot(P['timesteps'],LIFdata['signal_in'],label='input signal')
	ax3.plot(P['timesteps'],LIFdata['spikes_in'],label='input spikes')
	ax3.plot(P['timesteps'],spikes_out,label='output spikes')
	ax3.set(xlabel='time (s)')
	plt.legend()
	plt.show()

def make_tuning_curves(P,LIFdata,bioneuron):
	import seaborn as sns
	h=np.exp(-P['timesteps']/(P['tau_filter'])) #smooth spikes with exponential synaptic filter
	# h=h/np.sum(h) #normalize??
	spikes_out=np.zeros_like(P['timesteps'])
	for t in np.round(np.array(bioneuron.spikes),decimals=3):
		spikes_out[t/P['dt']/1000.]=1.0
	smoothed=np.array(np.convolve(spikes_out,h,mode='full')[:len(spikes_out)])
	X=np.arange(np.min(LIFdata['signal_in']),np.max(LIFdata['signal_in']),P['dx']) #eval points in X
	Hz=np.zeros_like(X) #firing rate for each eval point

	for xi in range(len(X)-1): #foreach eval_point
		ts=[] #find the time indices where the signal is between this and the next evalpoint
		for ti in range(len(P['timesteps'])):
			if X[xi] < LIFdata['signal_in'][ti] < X[xi+1]:
				ts.append(ti)
		if len(ts)>0:
			#average the smoothed spike value at each of these time indices
			Hz[xi]=np.average([smoothed[ti] for ti in ts])
			#convert units to Hz by dividing by the time window
			Hz[xi]=Hz[xi]/(P['timesteps'][ts[-1]]-P['timesteps'][ts[0]])

	# plot_signals_spikes(P,bioneuron,h,spikes_out)
	return X, Hz

def calculate_loss(P,LIFdata,X_NEURON,Hz_NEURON):
	#shape of activities and Hz is mismatched, so interpolate and slice activities for comparison
	from scipy.interpolate import interp1d
	f_NEURON_rate = interp1d(X_NEURON,Hz_NEURON)
	f_LIF_rate = interp1d(np.array(LIFdata['X_LIF']),np.array(LIFdata['Hz_LIF']))
	x_min=np.maximum(LIFdata['X_LIF'][0],X_NEURON[0])
	x_max=np.minimum(LIFdata['X_LIF'][-1],X_NEURON[-1])
	X=np.arange(x_min,x_max,P['dx'])
	loss=np.sqrt(np.average((f_NEURON_rate(X)-f_LIF_rate(X))**2))

	# sns.set(context='poster')
	# figure, ax1 = plt.subplots(1,1)
	# ax1.plot(X,f_NEURON_rate(X),label='bioneuron firing rate (Hz)')
	# ax1.plot(X,f_LIF_rate(X),label='LIF firing rate (Hz)')
	# ax1.set(xlabel='x',ylabel='firing rate (Hz)',title='loss=%0.3f' %loss)
	# plt.legend()
	# plt.show()

	return loss

def plot_loss(trials):
	sns.set(context='poster')
	figure1, ax1 = plt.subplots(1, 1)
	X=[t['tid'] for t in trials]
	Y=[t['result']['loss'] for t in trials]
	ax1.scatter(X,Y)
	ax1.set(xlabel='$t$',ylabel='loss')
	plt.show()
	figure1.savefig('hyperopt_result.png')

'''main---------------------------------------------------'''


def simulate(space):
	P=space['P']
	P['timesteps']=np.arange(0,P['t_sample'],P['dt'])
	with open("LIFdata.json") as data_file:    
		LIFdata = json.load(data_file)[0]
	weights=np.zeros(P['n_synapses'])
	for i in space.iterkeys():
		if i != 'P': 
			weights[int(i)]=space[i]
	bioneuron = make_bioneuron(P,weights)
	print 'Running NEURON'
	run_neuron(P,LIFdata,bioneuron)
	print 'Calculating tuning curve and loss ...'
	X_NEURON, Hz_NEURON = make_tuning_curves(P,LIFdata,bioneuron)
	loss=calculate_loss(P,LIFdata,X_NEURON,Hz_NEURON)
	return {'loss': loss, 'status': hyperopt.STATUS_OK}

def main():
	P=eval(open('parameters.txt').read())
	print 'Generating input spikes ...'
	spikes_in=[]
	LIFdata={}
	while np.sum(spikes_in)==0: #rerun nengo spike generator until it returns something
		spikes_in, signal_in, X_LIF, Hz_LIF = make_spikes_in(P)
	LIFdata['signal_in']=signal_in
	LIFdata['spikes_in']=spikes_in
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