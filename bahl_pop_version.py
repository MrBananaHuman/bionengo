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

def gen_spikes_in(P):
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

def gen_Bahl_pop(P):
	from neurons import Bahl
	import copy
	import ipdb

	#bioensemble={}
	bioensemble = []
	prototype = Bahl()
	for i in range(P['n_bioneurons']):
		# ipdb.set_trace()
		bioneuron = copy.deepcopy(prototype)
		bioneuron.cell.soma.v=np.random.uniform(-85,-50)
		# bioneuron.set_voltage(np.random.uniform(-85,-50))
		# bioneuron.start_recording()
		bioensemble.append(bioneuron)
		print bioensemble[i].cell.soma.v

	print 'after'
	for n in bioensemble:
		print n.cell.soma.v
	# ipdb.set_trace()
	return bioensemble

def gen_weights(P):
	weights=np.random.uniform(-1,1,size=(P['n_bioneurons'],P['n_synapses']))
	weights*=P['weight_scale']
	return weights

def gen_synapses(P,bioensemble,weights):
	dist=None
	if P['synapse_dist'] == 'random':
		dist=np.random.uniform(0,1,size=P['n_synapses'])
	for n in bioensemble.iterkeys():
		for i in dist:
			bioensemble[n].add_synapse(bioensemble[n].cell.apical(i),weights[n][i],P['tau'])

# doesn't seem to inject current through synapses when events are added pre-simulation
# def drive_synapses(spikes_in,bioensemble):
# 	for n in bioensemble.iterkeys():
# 		for syn in bioensemble[n].synapses:
# 			for t_spike in np.where(spikes_in)[0]:
# 				syn.conn.event(t_spike)

def run(P,spikes_in,bioensemble):
	neuron.init()
	for t in np.arange(0,P['t_run']*1000,P['dt']):
		if np.any(np.asfarray(np.where(spikes_in)[0]) == t):
			for n in bioensemble.iterkeys():
				for syn in bioensemble[n].synapses:
					syn.conn.event(t)
		neuron.run(t+P['dt'])

def make_dataframe(P,bioensemble):
	timesteps=np.array(bioensemble[0].t_record)
	columns=('time','neuron','voltage','spike')
	dataframe = pd.DataFrame(columns=columns, index=np.arange(0,len(timesteps)*P['n_bioneurons']))
	i=0
	for t in timesteps:
		for i in bioensemble.iterkeys():
			voltage=np.asfarray(bioensemble[i].v_record)[int(t/P['dt'])]
			spiked=1.0*np.any(np.asfarray(np.where(np.array(bioensemble[0].spikes))[0]) == t)
			dataframe.loc[i]=[t,i,voltage,spiked]
			i+=1
	return dataframe

def plot_voltage(bioensemble):
	sns.set(context='poster')
	figure, ax1 = plt.subplots(1,1)
	for n in bioensemble.iterkeys():
		# print n, np.array(bioensemble[n].t_record), np.array(bioensemble[n].v_record)
		ax1.plot(np.array(bioensemble[n].t_record), np.array(bioensemble[n].v_record),label='%s'%n)
	ax1.set(xlabel='time (ms)', ylabel='voltage (mV)')
	plt.legend()
	# plt.show()

def get_tuning_curve(P,bioensemble,spikes_in):
	timesteps=np.array(bioensemble[0].t_record)
	signal_in=np.sin(2*np.pi*timesteps/(P['t_sample']*1000))
	h=np.exp(-timesteps/(P['tau_filter']*1000)) #smooth spikes with exponential synaptic filter
	h=h/np.sum(h)
	all_spikes=[]
	for i in bioensemble.iterkeys():
		my_spikes=[]
		for t in timesteps:
			spiked=1.0*np.any(np.array(bioensemble[i].spikes) == t)
			my_spikes.append(spiked)
		all_spikes.append(my_spikes)
	all_spikes=np.average(all_spikes,axis=0).T
	rates=np.array(np.convolve(all_spikes,h,mode='full')[:len(all_spikes)])

	X=[]
	HZ_tmp=[]
	for t in range(len(timesteps)):
		x=signal_in[t]
		hz=rates[t]
		if np.any(np.array(X) == x) == False:
			X.append(x)
			HZ_tmp.append([hz])
		else:
			idx=np.where(np.array(X)==x)[0]
			HZ_tmp[idx].append(hz)
	HZ=[]
	for hz_list in HZ_tmp:
		HZ.append(np.average(hz_list))
	X,HZ=np.array(X),np.array(HZ)
	print X,HZ

	sns.set(context='poster')
	figure1, ax1 = plt.subplots(1,1)
	ax1.plot(timesteps,signal_in,label='input signal')
	ax1.plot(timesteps,rates,label='output rate')
	ax1.set(xlabel='time (ms)', ylabel='firing rate (1/ms)')
	plt.legend()
	figure2, ax2 = plt.subplots(1,1)
	ax2.scatter(X,HZ,label='average')
	ax2.set(xlabel='x',ylabel='firing rate (1/ms)')
	plt.legend()
	plt.show()

def main():
	P=eval(open('parameters.txt').read())
	P['dt']=neuron.h.dt
	spikes_in=gen_spikes_in(P)
	print 'input spikes at ', np.asfarray(np.where(spikes_in)[0])
	bioensemble=gen_Bahl_pop(P)
	weights=gen_weights(P)
	gen_synapses(P,bioensemble,weights)
	run(P,spikes_in,bioensemble)
	# dataframe=make_dataframe(P,bioensemble)
	plot_voltage(bioensemble)
	get_tuning_curve(P,bioensemble,spikes_in)

if __name__=='__main__':
	main()