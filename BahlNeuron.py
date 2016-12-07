import numpy as np
import nengo
import neuron
from nengo.builder import Builder, Operator, Signal
import copy
import ipdb
import json

class BahlNeuron(nengo.neurons.NeuronType):
	'''compartmental neuron from Bahl et al 2012'''

	probeable=('spikes','voltage')
	def __init__(self,P):
		super(BahlNeuron,self).__init__()
		self.P=P
		self.bioneuron_directory=P['bioneuron_directory']
		if self.bioneuron_directory is not None:
			with open(self.bioneuron_directory+'filenames.txt','r') as df:
				self.filenames=json.load(df)
		else:
			self.filenames=None
		self.optimized=None

	class Bahl():
		def __init__(self):
			neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc') #todo
			# neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc') #todo
			self.cell = neuron.h.Bahl()
			self.synapses = {} #index=input neuron, value=synapses
			self.vecstim = {} #index=input neuron, value=VecStim object (input spike times)
			self.netcons = {} #index=input neuron, value=NetCon Object (connection b/w VecStim and Syn)

		def start_recording(self):
			self.v_record = neuron.h.Vector()
			self.v_record.record(self.cell.soma(0.5)._ref_v)
			self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
			self.t_record = neuron.h.Vector()
			self.t_record.record(neuron.h._ref_t)
			self.spikes = neuron.h.Vector()
			self.ap_counter.record(neuron.h.ref(self.spikes))
			self.nengo_spike_times=[]
			self.nengo_voltages=[]

		def add_bias(self,bias):
			self.bias = bias
			self.bias_current = neuron.h.IClamp(self.cell.soma(0.5))
			self.bias_current.delay = 0
			self.bias_current.dur = 1e9 #todo
			self.bias_current.amp = self.bias

		def add_connection(self,idx):
			self.synapses[idx]=[] #list of each synapse in this connection
			self.vecstim[idx]={'vstim':[],'vtimes':[]} #list of input spike times from this neuron
			self.netcons[idx]=[] #list of netcon objects between input vecstim and synapses for this nrn

		def add_synapse(self,idx,syn_type,section,weight,tau,tau2=0.005):
			from synapses import ExpSyn, Exp2Syn
			if syn_type == 'ExpSyn':
				self.synapses[idx].append(ExpSyn(section,weight,tau))
			elif syn_type == 'Exp2Syn':
				self.synapses[idx].append(Exp2Syn(section,weight,tau,tau2))

	def create(self,idx):
		self.bio_idx=idx
		self.bahl=None
		return copy.copy(self)

	def save_optimization(self):
		self.indices=[]
		self.weights=[]
		self.locations=[]
		self.biases=[]
		self.signal_in=[]
		self.bio_spikes=[]
		self.bio_rates=[]
		self.ideal_spikes=[]
		self.ideal_rates=[]
		self.losses=[]
		for j in range(len(self.filenames)): #for each bioneuron
			with open(self.filenames[j],'r') as data_file:
				bioneuron_info=json.load(data_file)
			self.indices.append(bioneuron_info['bio_idx'])
			self.weights.append(bioneuron_info['weights'])
			self.locations.append(bioneuron_info['locations'])
			self.biases.append(bioneuron_info['bias'])
			self.signal_in.append(bioneuron_info['signal_in'])
			self.bio_spikes.append(bioneuron_info['bio_spikes'])
			self.bio_rates.append(bioneuron_info['bio_rates'])
			self.ideal_spikes.append(bioneuron_info['ideal_spikes'])
			self.ideal_rates.append(bioneuron_info['ideal_rates'])
			self.losses.append(bioneuron_info['loss'])
		self.indices=np.array(self.indices)
		self.weights=np.array(self.weights)
		self.locations=np.array(self.locations)
		self.biases=np.array(self.biases)
		self.signal_in=np.array(self.signal_in)[0]
		self.bio_spikes=np.array(self.bio_spikes)
		self.bio_rates=np.array(self.bio_rates)
		self.ideal_spikes=np.array(self.ideal_spikes)
		self.ideal_rates=np.array(self.ideal_rates)
		self.losses=np.array(self.losses)
		np.savez(self.bioneuron_directory+'biodata.npz',
				indices=self.indices,weights=self.weights,locations=self.locations,
				biases=self.biases,signal_in=self.signal_in,bio_spikes=self.bio_spikes,
				bio_rates=self.bio_rates,ideal_spikes=self.ideal_spikes,ideal_rates=self.ideal_rates,
				losses=self.losses)

	def rates(self, x, gain, bias): #todo: remove this without errors
		return x #CustomSolver class calculates A from Y

	def gain_bias(self, max_rates, intercepts): #todo: remove this without errors
		return np.ones(len(max_rates)),np.ones(len(max_rates))

	def step_math(self,dt,spiked,neurons,voltage,time):
		# if time==dt: neuron.init() #todo: prettier way to initialize first timestep without segfault
		desired_t=time*1000
		# ipdb.set_trace()
		'''runs NEURON only for 1st bioensemble in model...OK because transmit_spikes happend already?'''
		neuron.run(desired_t) 
		new_spiked=[]
		new_voltage=[]
		for nrn in neurons:
			spike_times=np.round(np.array(nrn.bahl.spikes),decimals=3)
			count=np.sum(spike_times>(time-dt)*1000)
			new_spiked.append(count)
			nrn.bahl.nengo_spike_times.extend(
					spike_times[spike_times>(time-dt)*1000])
			volt=np.array(nrn.bahl.v_record)[-1] #fails if neuron.init() not called at right times
			nrn.bahl.nengo_voltages.append(volt)
			new_voltage.append(volt)
		spiked[:]=np.array(new_spiked)/dt
		voltage[:]=np.array(new_voltage)
		# ipdb.set_trace()

'''
Builder #############################################################################3
'''

class SimBahlNeuron(Operator):
	def __init__(self,neurons,output,voltage,states):
		super(SimBahlNeuron,self).__init__()
		self.neurons=neurons
		self.output=output
		self.voltage=voltage
		self.time=states[0]
		self.reads = [states[0]]
		self.sets=[output,voltage]
		self.updates=[]
		self.incs=[]
		self.neurons.neurons=[self.neurons.create(i) for i in range(output.shape[0])]
		self.P=self.neurons.P
		self.neurons.father_op=self
		neuron.h.dt = self.P['dt_neuron']*1000
		neuron.init()

	def make_step(self,signals,dt,rng):
		output=signals[self.output]
		voltage=signals[self.voltage]
		time=signals[self.time]
		def step_nrn():
			#one bahlneuron object runs step_math, but arg is all cells - what/where returns?
			self.neurons.step_math(dt,output,self.neurons.neurons,voltage,time)
		return step_nrn


class TransmitSpikes(Operator):
	def __init__(self,spikes,bahl_op,states):
		self.spikes=spikes
		self.neurons=bahl_op.neurons.neurons
		self.time=states[0]
		self.reads=[spikes,states[0]]
		self.updates=[]
		self.sets=[]
		self.incs=[]
		neuron.init()

	def make_step(self,signals,dt,rng):
		spikes=signals[self.spikes]
		time=signals[self.time]
		def step():
			# ipdb.set_trace()
			for n in range(spikes.shape[0]): #for each input neuron
				if spikes[n] > 0: #if this neuron spiked at this time, then
					for nrn in self.neurons: #for each bioneuron
						for syn in nrn.bahl.synapses[n]: #for each synapse conn. to input
							syn.spike_in.event(1.0*time*1000) #add a spike at time t (ms)
		return step


def load_weights(P,bahl_op):
	import ipdb
	for j in range(len(bahl_op.neurons.filenames)): #for each bioneuron
		bahl=BahlNeuron.Bahl()
		with open(bahl_op.neurons.filenames[j],'r') as data_file:
			bioneuroens_pre_neuronsfo=json.load(data_file)
		ens_pre_neuronsputs=len(np.array(bioneuroens_pre_neuronsfo['weights']))
		n_synapses=len(np.array(bioneuroens_pre_neuronsfo['weights'])[0])
		for n in range(ens_pre_neuronsputs):
			bahl.add_bias(bioneuroens_pre_neuronsfo['bias'])
			bahl.add_connection(n)
			for i in range(n_synapses): #for each synapse connected to input neuron
				section=bahl.cell.apical(np.array(bioneuroens_pre_neuronsfo['locations'])[n][i])
				weight=np.array(bioneuroens_pre_neuronsfo['weights'])[n][i]
				bahl.add_synapse(n,bahl_op.P['synapse_type'],section,
										weight,P['synapse_tau'],None)
		bahl.start_recording()
		bahl_op.neurons.neurons[j].bahl=bahl

@Builder.register(BahlNeuron)
def build_bahlneuron(model,neuron_type,ens):
	model.sig[ens]['voltage'] = Signal(np.zeros(ens.ensemble.n_neurons),
						name='%s.voltage' %ens.ensemble.label)
	op=SimBahlNeuron(neurons=neuron_type,
						output=model.sig[ens]['out'],voltage=model.sig[ens]['voltage'],
						states=[model.time])
	model.add_op(op)

@Builder.register(nengo.Ensemble)
def build_ensemble(model,ens):
	nengo.builder.ensemble.build_ensemble(model,ens)

@Builder.register(nengo.Connection)
def build_connection(model,conn):
	import copy
	use_nrn = (
		isinstance(conn.post, nengo.Ensemble) and
		isinstance(conn.post.neuron_type, BahlNeuron))
	if use_nrn: #bioneuron connection
		rng = np.random.RandomState(model.seeds[conn])
		model.sig[conn]['in']=model.sig[conn.pre]['out']
		P=copy.copy(conn.post.neuron_type.P)
		bahl_op=conn.post.neuron_type.father_op

		if bahl_op.neurons.bioneuron_directory == None:
			P['ens_pre_neurons']=conn.pre.n_neurons
			P['ens_pre_dim']=conn.pre.dimensions
			P['ens_pre_min_rate']=conn.pre.max_rates.low
			P['ens_pre_max_rate']=conn.pre.max_rates.high
			P['ens_pre_seed']=conn.pre.seed
			P['ens_pre_type']=str(conn.pre.neuron_type)
			if isinstance(conn.pre.neuron_type,BahlNeuron):
				P['ens_pre_label']=conn.pre.label+'/'
			P['ens_label']=conn.post.label+'/'
			P['ens_ideal_neurons']=conn.post.n_neurons
			P['ens_ideal_dim']=conn.post.dimensions
			P['ens_ideal_seed']=conn.post.seed
			P['ens_ideal_min_rate']=conn.post.max_rates.low
			P['ens_ideal_max_rate']=conn.post.max_rates.high
			from optimize_bioneuron import optimize_bioneuron
			bahl_op.neurons.bioneuron_directory=optimize_bioneuron(P)
			filenames=bahl_op.neurons.bioneuron_directory+'filenames.txt'
			with open(filenames,'r') as df:
				bahl_op.neurons.filenames=json.load(df)

		load_weights(P,bahl_op) 	#load weights into these operators
		conn.post.neuron_type.save_optimization()	#save contents of filenames to the neuron
		model.add_op(TransmitSpikes(model.sig[conn]['in'],bahl_op,states=[model.time])) #setup spike transmission

	else: #normal connection
		return nengo.builder.connection.build_connection(model, conn)



class CustomSolver(nengo.solvers.Solver):
	import json
	import ipdb
	import copy

	def __init__(self,P,conn):
		self.P=P
		self.conn=conn #only used to grab the SimBahlOp operator
		self.weights=False #decoders not weights
		self.bahl_op=None
		self.A=None
		self.Y=None
		self.decoders=None

	def __call__(self,A,Y,rng=None,E=None): #function that gets called by the builder
		'''preloaded spike approach: same as below, but saved from optimize_bioneuron'''
		self.activities=self.conn.post.neuron_type.bio_rates
		self.upsilon=self.conn.post.neuron_type.signal_in
		self.solver=nengo.solvers.LstsqL2()
		self.decoders,self.info=self.solver(self.activities.T,self.upsilon)
		return self.decoders, dict()

	 	'''spike-approach: feed an N-dim white noise signal, through a spiking pre LIF population,
 		collect spikes from the bioneuron population, then construct multidimensional A and Y
 		NOTE: currently breaks bahlneuron_test simulation because NEURON references are not
 		properly cleared, or something'''

		# '''1. Generate white noise signal'''
		# print 'Computing A and Y...'
		# from optimize_bioneuron import make_signal
		# #todo: pass P dictionary around
		# P=self.P
		# raw_signal=make_signal(P)
		# self.bahl_op=self.conn.post.neuron_type.father_op

		# ''' 2. Pass signal through pre LIF population, generate spikes'''
		# import nengo
		# import numpy as np
		# import pandas as pd
		# import ipdb
		# with nengo.Network() as decode_model:
		# 	signal_decode = nengo.Node(lambda t: raw_signal[:,int(t/P['dt_nengo'])]) #all dim, index at t
		# 	pre_decode = nengo.Ensemble(n_neurons=P['ens_pre_neurons'],
		# 							dimensions=P['dim'],
		# 							seed=P['ens_pre_seed'])
		# 	nengo.Connection(signal_decode,pre_decode,synapse=None)
		# 	probe_signal_decode = nengo.Probe(signal_decode)
		# 	probe_pre_decode = nengo.Probe(pre_decode.neurons,'spikes')
		# with nengo.Simulator(decode_model,dt=P['dt_nengo']) as sim_decode:
		# 	sim_decode.run(P['t_train'])
		# signal_in=sim_decode.data[probe_signal_decode]
		# spikes_in=sim_decode.data[probe_pre_decode]

		# '''3. New bioneurons, send spikes to bioneurons, collect spikes from bioneurons'''
		# from optimize_bioneuron import make_bioneuron,connect_bioneuron,run_bioneuron
		# bioneurons=[]
		# for b in range(P['n_bio']):
		# 	with open(self.bahl_op.neurons.filenames[b],'r') as data_file:
		# 		bioneuroens_pre_neuronsfo=json.load(data_file)
		# 	bias=bioneuroens_pre_neuronsfo['bias']
		# 	weights=np.zeros((P['ens_pre_neurons'],P['n_syn']))
		# 	locations=np.zeros((P['ens_pre_neurons'],P['n_syn']))
		# 	for n in range(P['ens_pre_neurons']):
		# 		for i in range(P['n_syn']):
		# 			weights[n][i]=np.array(bioneuroens_pre_neuronsfo['weights'])[n][i]
		# 			locations[n][i]=np.array(bioneuroens_pre_neuronsfo['locations'])[n][i]
		# 	bioneuron = make_bioneuron(P,weights,locations,bias)
		# 	connect_bioneuron(P,spikes_in,bioneuron)
		# 	bioneurons.append(bioneuron)
		# run_bioneuron(P)

		# '''4. Collect spikes from bioneurons'''
		# from optimize_bioneuron import get_rates
		# bio_rates=[]
		# for nrn in bioneurons:
		# 	bio_spike, bio_rate=get_rates(P,np.round(np.array(nrn.spikes),decimals=3))
		# 	bio_rates.append(bio_rate)

		# '''5. Assemble A and Y'''
		# # ipdb.set_trace()
		# self.activities=np.array(bio_rates)
		# self.upsilon=signal_in
		# self.solver=nengo.solvers.LstsqL2()
		# self.decoders,self.info=self.solver(self.activities.T,self.upsilon)

		# '''6. Reset NEURON'''
		# print 'done'
		# # neuron.init()
		# # print 'after'
		# ipdb.set_trace()
		# # self.bahl_op.neurons.neurons=None
		# # self.bahl_op.neurons.neurons=[self.bahl_op.neurons.create(i) for i in range(P['n_bio'])]
		# # # for nrn in bioneurons:
		# # 	# for n in range(P['ens_pre_neurons']):
		# # 	# 	del(nrn.synapses)
		# # 	# 	del(nrn.vecstim)
		# # 	# 	del(nrn.netcons)
		# # 	# 	nrn.synapses={}
		# # 	# 	nrn.vecstim={}
		# # 	# 	nrn.netcons={}
		# # load_weights(P,self.bahl_op)
		# # neuron.init()

		# return self.decoders, dict()