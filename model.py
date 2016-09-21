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
		signal = nengo.Node(output=lambda x: np.sin(2*np.pi*x/P['t_sample']))
		ens_in = nengo.Ensemble(5,dimensions=1)
		probe_signal = nengo.Probe(signal)
		probe_in = nengo.Probe(ens_in.neurons,'spikes')
	with nengo.Simulator(model) as sim:
		sim.run(P['t_sample'])
		eval_points, activities = nengo.utils.ensemble.tuning_curves(ens_in,sim)
	spikes_in=sim.data[probe_in]
	return np.average(spikes_in,axis=1)/1000. #summed spike raster
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
	for t in np.arange(0,P['t_run']*1000,P['dt']):
		if np.any(np.asfarray(np.where(spikes_in)[0]) == t):
			for syn in bioneuron.synapses:
				syn.conn.event(t)
		neuron.run(t+P['dt'])

def make_dataframe(P):
	timesteps=np.arange(0,P['t_run'],P['dt'])
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
	timesteps=np.array(bioneuron.t_record)
	h=np.exp(-timesteps/(P['tau_filter'])) #smooth spikes with exponential synaptic filter
	h=h/np.sum(h)
	avg_spikes=np.average([np.array(dataframe.query("trial==%s"%i)['spike'])
							for i in range(P['n_trials'])],axis=0)
	rates=np.array(np.convolve(avg_spikes,h,mode='full')[:len(avg_spikes)])

	#construct Hz vs X
	def signal_in(t):
		return np.sin(2*np.pi*t/(P['t_sample']*1000)) #period 2pi/t_sample, t in msp

	X=np.arange(np.min(signal_in(timesteps)),np.max(signal_in(timesteps)),P['dx'])
	Hz=np.zeros_like(X)
	for xi in range(len(X)-1):
		ts=[]
		for ti in range(len(timesteps)):
			if X[xi] < signal_in(timesteps[ti]) < X[xi+1]:
				ts.append(ti)
		Hz[xi]=np.average([rates[ti] for ti in ts])

	sns.set(context='poster')
	figure1, ax1 = plt.subplots(1,1)
	ax1.plot(timesteps,signal_in(timesteps),label='input signal')
	ax1.plot(timesteps,rates,label='output rate')
	ax1.set(xlabel='time (ms)', ylabel='firing rate (1/ms)')
	plt.legend()
	figure2, ax2 = plt.subplots(1,1)
	ax2.scatter(X,Hz)
	ax2.set(xlabel='x',ylabel='firing rate (1/ms)')
	plt.show()

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
	get_tuning_curve(P,bioneuron,dataframe)
	# dataframe=make_dataframe(P,bioensemble)

if __name__=='__main__':
	main()