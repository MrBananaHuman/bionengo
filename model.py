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

# nengo.rc.set("decoder_cache", "disable", "True")

def make_spikes_in(P):
	with nengo.Network() as model:
		signal = nengo.Node(output=lambda t: np.sin(2*np.pi*t/P['t_sample']))
		ens_in = nengo.Ensemble(1,dimensions=1) 
		probe_signal = nengo.Probe(signal) #get spikes from one neuron (one tuning curve)
		probe_in = nengo.Probe(ens_in.neurons,'spikes')
	with nengo.Simulator(model) as sim:
		sim.run(P['t_sample'])
		eval_points, activities = nengo.utils.ensemble.tuning_curves(ens_in,sim)
	spikes_in=sim.data[probe_in]
	return np.average(spikes_in,axis=1)/1000. #summed spike raster and neuron tuning curve
	# return spikes_in #spike raster for all neurons

def make_weights(P):
	weights=np.random.uniform(-1,1,size=P['n_synapses'])
	weights*=P['weight_scale']
	return weights

def make_synapses(P,bioneuron,weights):
	dist=None
	if P['synapse_dist'] == 'random':
		dist=np.random.uniform(0,1,size=P['n_synapses'])
	for i in dist:
		bioneuron.add_synapse(bioneuron.cell.apical(i),weights[i],P['tau'])

def make_bioneuron(P):
	from neurons import Bahl
	weights=make_weights(P)
	bioneuron=Bahl()
	make_synapses(P,bioneuron,weights)
	return bioneuron

def run(P,spikes_in,bioneuron):
	neuron.init()
	for t in np.arange(0,P['t_sample']*1000,P['dt']):
		if np.any(np.asfarray(np.where(spikes_in)[0]) == t):
			for syn in bioneuron.synapses:
				syn.conn.event(t)
		neuron.run(t+P['dt'])

def make_dataframe(P):
	timesteps=np.arange(0,P['t_sample'],P['dt'])
	columns=('time','voltage','spike','trial')
	dataframe = pd.DataFrame(columns=columns, index=np.arange(0,len(timesteps)*P['n_trials']))
	return dataframe

def update_dataframe(P,bioneuron,dataframe,trial):
	timesteps=np.array(bioneuron.t_record)
	start=trial*len(timesteps)
	i=0
	for t in timesteps:
		voltage=np.asfarray(bioneuron.v_record)[int(t/P['dt'])]
		spiked=1.0*np.any(np.array(bioneuron.spikes) == t)
		dataframe.loc[start+i]=[t,voltage,spiked,trial]
		i+=1
	return dataframe

def plot_voltage(bioneuron):
	sns.set(context='poster')
	figure, ax1 = plt.subplots(1,1)
	ax1.plot(np.array(bioneuron.t_record), np.array(bioneuron.v_record))
	ax1.set(xlabel='time (ms)', ylabel='voltage (mV)')
	# plt.show()

def get_tuning_curve(P,bioneuron,dataframe):
	import seaborn as sns
	import ipdb
	timesteps=np.array(bioneuron.t_record)/1000.
	h=np.exp(-timesteps/(P['tau_filter'])) #smooth spikes with exponential synaptic filter
	# h=h/np.sum(h) #normalize??
	avg_spikes=np.average([np.array(dataframe.query("trial==%s"%i)['spike'])
							for i in range(P['n_trials'])],axis=0)
	smoothed=np.array(np.convolve(avg_spikes,h,mode='full')[:len(avg_spikes)])
	signal_in=np.array([np.sin(2*np.pi*t/P['t_sample']) for t in timesteps])

	X=np.arange(np.min(signal_in),np.max(signal_in),P['dx']) #eval points in X
	Hz=np.zeros_like(X) #firing rate for each eval point
	for xi in range(len(X)-1): #foreach eval point
		ts=[]
		#find the time indices where the signal is between this and the next evalpoint
		for ti in range(len(timesteps)):
			if X[xi] < signal_in[ti] < X[xi+1]:
				ts.append(ti)
		#average the smoothed spike value at each of these time indices
		Hz[xi]=np.average([smoothed[ti] for ti in ts])
		#convert units to Hz by dividing by the time window
		Hz[xi]=Hz[xi]/(timesteps[ts[-1]]-timesteps[ts[0]])

	sns.set(context='poster')
	figure1, (ax1,ax2) = plt.subplots(2,1)
	ax1.plot(timesteps,signal_in,label='input signal')
	ax1.plot(timesteps,avg_spikes,label='output spikes')
	ax1.set(xlabel='time (s)', ylabel='firing rate (Hz)')
	ax2.scatter(X,Hz,label='trials=%s'%P['n_trials'])
	ax2.set(xlabel='x',ylabel='firing rate (Hz)')
	plt.legend()
	# figure3, ax3 = plt.subplots(1,1)
	# ax3.plot(timesteps,h,label='filter')
	# ax3.set(xlabel='time (s)', ylabel='filter value')
	plt.show()
	return X,Hz

def main():
	P=eval(open('parameters.txt').read())
	P['dt']=neuron.h.dt
	spikes_in=make_spikes_in(P)
	print 'input spikes at ', np.asfarray(np.where(spikes_in)[0])
	bioneuron=make_bioneuron(P)
	dataframe=make_dataframe(P)
	for i in range(P['n_trials']):
		bioneuron.start_recording()
		run(P,spikes_in,bioneuron)
		dataframe=update_dataframe(P,bioneuron,dataframe,i)
		# plot_voltage(bioneuron)
	X,Hz=get_tuning_curve(P,bioneuron,dataframe)
	# dataframe=make_dataframe(P,bioensemble)

if __name__=='__main__':
	main()