'''
Peter Duggins
September 2016
bionengo - interface NEURON and Bahr2 neurons with nengo
'''

import nengo
import numpy as np
from neurons import Bahl
from synapses import ExpSyn
import neuron
import matplotlib.pyplot as plt

# nengo.rc.set("decoder_cache", "disable", "True")

def gen_spikes_in(P):
	with nengo.Network() as model:
		signal = nengo.Node(output=lambda x: np.sin(x))
		ens_in = nengo.Ensemble(100,dimensions=1)
		probe_signal = nengo.Probe(signal)
		probe_in = nengo.Probe(ens_in.neurons,'spikes')
	with nengo.Simulator(model) as sim:
		sim.run(P['t_sample'])
		eval_points, activities = nengo.utils.ensemble.tuning_curves(ens_in,sim)
	spikes_in=sim.data[probe_in]
	return np.sum(spikes_in,axis=1)/1000. #spike summed raster
	# return spikes_in #spike raster for all neurons

def gen_Bahl_pop(P):
	bioensemble={}
	for i in range(P['popsize']):
		bioneuron=Bahl()
		bioneuron.set_voltage(np.random.uniform(-65,-75))
		bioensemble[i]=bioneuron
	return bioensemble

def gen_weights(P):
	return np.random.uniform(100,101,size=(P['popsize'],P['n_synapses']))

def gen_synapses(P,bioensemble,weights):
	dist=None
	if P['synapse_dist'] == 'random':
		dist=np.random.uniform(0,1,size=P['n_synapses'])
	for n in bioensemble.iterkeys():
		for i in dist:
			bioensemble[n].add_synapse(bioensemble[n].cell.soma(i),weights[n][i],P['tau']) #apical

def drive_synapses(spikes_in,bioensemble):
	for n in bioensemble.iterkeys():
		for syn in bioensemble[n].synapses:
			for t_spike in np.where(spikes_in)[0]:
				syn.conn.event(t_spike)
			print dir(syn.conn.event)


def main():
	P=eval(open('parameters.txt').read())
	spikes_in=gen_spikes_in(P)
	bioensemble=gen_Bahl_pop(P)
	weights=gen_weights(P)
	gen_synapses(P,bioensemble,weights)
	drive_synapses(spikes_in,bioensemble)
	neuron.init()
	neuron.run(P['t_sample']*1000)
	plt.plot(bioensemble[0].t_record, bioensemble[0].v_record)
	plt.show()

if __name__=='__main__':
	main()