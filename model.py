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

def make_search_space(P):
	space={'P':P}
	for i in range(P['n_synapses']): #add a quniform weight for each synapse
		space[i]=hyperopt.hp.quniform('weights[%s]'%i,P['weight_min'],P['weight_max'],0.00001)
	return space

def make_spikes_in(P):
	with nengo.Network() as model:
		signal = nengo.Node(
				# output=lambda t: np.sin(2*np.pi*t/P['t_sample']))
				#rms=radius ensures that all max/min of signal bound max/min of eval_points?
				output=nengo.processes.WhiteSignal(P['t_sample'],high=0.5/P['dt'],rms=P['radius']))
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
	return spikes_in, signal_in, eval_points, activities

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
	return bioneuron

def run_neuron(P,spikes_in,bioneuron):
	import sys
	neuron.init()
	for t in P['timesteps']: 
		sys.stdout.write("\r%d%%" %(100*t/(P['t_sample'])))
		sys.stdout.flush()
		if np.any(np.asfarray(np.where(spikes_in)[0]) == t/P['dt']):
			# print 'spike at t=%s' %(t/P['dt'])
			for syn in bioneuron.synapses:
				syn.conn.event(t*1000/P['dt'])
		neuron.run(t*1000)

def make_dataframe(P):
	# timesteps=np.arange(0,P['t_sample'],P['dt'])
	columns=('time','voltage','spike','trial')
	dataframe = pd.DataFrame(columns=columns,
							index=np.arange(0,len(P['timesteps'])*P['n_trials']))
	return dataframe

def update_dataframe(P,bioneuron,dataframe,trial):
	# timesteps=np.array(bioneuron.t_record)
	start=trial*len(P['timesteps'])
	i=0
	for t in P['timesteps']:
		voltage=np.asfarray(bioneuron.v_record)[int(t/P['dt'])]
		spiked=1.0*np.any(np.array(bioneuron.spikes) == t*1000)
		dataframe.loc[start+i]=[t,voltage,spiked,trial]
		i+=1
	return dataframe

def make_plots(P,bioneuron,X,Hz,h,avg_spikes,signal_in,eval_points,activities):
	sns.set(context='poster')
	figure, (ax2,ax4) = plt.subplots(2,1)
	# figure, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
	# ax1.plot(np.array(bioneuron.t_record), np.array(bioneuron.v_record))
	# ax1.set(xlabel='time (ms)', ylabel='voltage (mV)')
	ax2.plot(P['timesteps'],signal_in,label='input signal')
	ax2.plot(P['timesteps'],avg_spikes,label='output spikes')
	ax2.set(xlabel='time (s)', ylabel='firing rate (Hz)')
	# ax3.plot(P['timesteps'],h,label='filter')
	# ax3.set(xlabel='time (s)', ylabel='filter value')
	ax4.scatter(X,Hz,label='bioneuron firing rate (Hz)')
	ax4.plot(eval_points,activities,label='LIF firing rate (Hz)')
	ax4.set(xlabel='x',ylabel='firing rate (Hz)')
	plt.legend()
	plt.show()

def make_tuning_curves(P,bioneuron,dataframe,signal_in,eval_points,activities):
	import seaborn as sns
	import ipdb
	h=np.exp(-P['timesteps']/(P['tau_filter'])) #smooth spikes with exponential synaptic filter
	# h=h/np.sum(h) #normalize??
	avg_spikes=np.average([np.array(dataframe.query("trial==%s"%i)['spike'])
							for i in range(P['n_trials'])],axis=0)
	smoothed=np.array(np.convolve(avg_spikes,h,mode='full')[:len(avg_spikes)])
	X=np.arange(np.min(signal_in),np.max(signal_in),P['dx']) #eval points in X
	Hz=np.zeros_like(X) #firing rate for each eval point

	for xi in range(len(X)-1): #foreach eval_point
		ts=[] #find the time indices where the signal is between this and the next evalpoint
		for ti in range(len(P['timesteps'])):
			if X[xi] < signal_in[ti] < X[xi+1]:
				ts.append(ti)
		#average the smoothed spike value at each of these time indices
		Hz[xi]=np.average([smoothed[ti] for ti in ts])
		#convert units to Hz by dividing by the time window
		Hz[xi]=Hz[xi]/(P['timesteps'][ts[-1]]-P['timesteps'][ts[0]])

	ipdb.set_trace()
	make_plots(P,bioneuron,X,Hz,h,avg_spikes,signal_in,eval_points,activities)
	return X, Hz

def calculate_loss(P,X_NEURON,Hz_NEURON,X_LIF,Hz_LIF):
	#shape of activities and Hz is mismatched, so interpolate and slice activities for comparison
	from scipy.interpolate import interp1d
	f_NEURON_rate = interp1d(X_NEURON,Hz_NEURON)
	f_LIF_rate = interp1d(X_LIF,Hz_LIF)
	x_min=np.maximum(X_NEURON[0],X_LIF[0])
	x_max=np.minimum(X_NEURON[-1],X_LIF[-1])
	X=np.arange(x_min,x_max,P['dx'])
	loss=np.sqrt(np.average((f_NEURON_rate(X)-f_LIF_rate(X))**2))
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
	weights=np.zeros(P['n_synapses'])
	for i in space.iterkeys():
		if i != 'P': 
			weights[int(i)]=space[i]
	spikes_in, signal_in, eval_points, activities = make_spikes_in(P)
	print 'input spikes at t=', np.asfarray(np.where(spikes_in)[0])
	bioneuron = make_bioneuron(P,weights)
	dataframe = make_dataframe(P)
	for i in range(P['n_trials']):
		bioneuron.start_recording()
		run_neuron(P,spikes_in,bioneuron)
		dataframe=update_dataframe(P,bioneuron,dataframe,i)
	X, Hz = make_tuning_curves(P,bioneuron,dataframe,signal_in,eval_points,activities)
	loss=calculate_loss(P,X,Hz,eval_points,activities)
	return {'loss': loss, 'status': hyperopt.STATUS_OK}

def main():
	P=eval(open('parameters.txt').read())
	if P['optimization']=='hyperopt':
		search_space=make_search_space(P)
		trials=hyperopt.Trials()
		best=hyperopt.fmin(simulate,space=search_space,algo=hyperopt.tpe.suggest,
							max_evals=P['max_evals'],trials=trials)
		plot_loss(trials)

if __name__=='__main__':
	main()