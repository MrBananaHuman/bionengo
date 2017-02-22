import numpy as np
import nengo
import neuron
import hyperopt
import timeit
import json
import copy
import ipdb
import os
import sys
import gc
import matplotlib.pyplot as plt
import seaborn
from synapses import ExpSyn
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot
from pathos.multiprocessing import ProcessingPool as Pool

class Bahl():
	def __init__(self,P,bias):
		neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
		self.cell = neuron.h.Bahl()
		self.bias = bias
		self.bias_current = neuron.h.IClamp(self.cell.soma(0.5))
		self.bias_current.delay = 0
		self.bias_current.dur = 1e9  # TODO; limits simulation time
		self.bias_current.amp = self.bias
		self.synapses={}
		self.netcons={}
	def make_synapses(self,P,my_weights,my_locations):
		for inpt in P['ens_post']['inpts'].iterkeys():
			ens_pre_info=P['ens_post']['inpts'][inpt]
			self.synapses[inpt]=np.empty((ens_pre_info['pre_neurons'],P['n_syn']),dtype=object)
			for pre in range(ens_pre_info['pre_neurons']):
				for syn in range(P['n_syn']):
					section=self.cell.apical(my_locations[inpt][pre][syn])
					weight=my_weights[inpt][pre][syn]
					synapse=ExpSyn(section,weight,P['tau'])
					self.synapses[inpt][pre][syn]=synapse	
	def start_recording(self):
		self.v_record = neuron.h.Vector()
		self.v_record.record(self.cell.soma(0.5)._ref_v)
		self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
		self.t_record = neuron.h.Vector()
		self.t_record.record(neuron.h._ref_t)
		self.spikes = neuron.h.Vector()
		self.ap_counter.record(neuron.h.ref(self.spikes))
	def event_step(self,t_neuron,inpt,pre):
		for syn in self.synapses[inpt][pre]: #for each synapse in this connection
			syn.spike_in.event(t_neuron) #add a spike at time (ms)

def make_pre_ideal_spikes(P,network):
	bio_dict={}
	#opt_net=copy.copy(network)
	for ens in opt_net.ensembles:
		if not isinstance(ens.neuron_type,BahlNeuron): continue #only bioensembles selected
		ens.neuron_type=nengo.LIF()
		bio_dict[ens.label]={}
		with opt_net: bio_dict[ens.label]['probe']=nengo.Probe(ens,synapse=ens.synapse)
		bio_dict[ens.label]['inputs']={}
		for conn in opt_net.connections:
			if isinstance(conn.post_obj,BahlNeuron):
				bio_dict[ens.label]['inputs'][conn.pre_obj.label]={}
				with opt_net: 
					bio_dict[ens.label]['inputs'][conn.pre_obj.label]['probe']=
							nengo.Probe(conn.pre_obj.neurons,'spikes')
	#rebuild network?
	#define input signals and connect to inputs
	with nengo.Simulator(opt_net,dt=P['dt_nengo']) as opt_sim:
		opt_sim.run(P['optimize']['t_final'])
	for bio in bio_dict.iterkeys():
		try: 
			os.makedirs(bio)
			os.chdir(bio)
		except OSError:
			os.chdir(bio)
		bio_dict[bio]['ideal_spikes']=opt_sim.data[bio_dict[bio]['probe']]
		for inpt in bio_dict[bio]['inputs'].iterkeys():
			bio_dict[bio]['inputs'][inpt]['pre_spikes']=
					opt_sim.data[bio_dict[bio]['inputs'][inpt]['probe']]
			np.savez('spikes_from_%s_to_%s.npz'%(inpt,bio),spikes=
					bio_dict[bio]['inputs'][inpt]['pre_spikes'])
		np.savez('spikes_ideal_%s.npz'%bio,spikes=bio_dict[bio]['ideal_spikes'])
		os.chdir('..')