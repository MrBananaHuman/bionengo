import numpy as np
import nengo
import neuron
from nengo.builder import Builder, Operator, Signal
import copy
import ipdb
import json
import os

'''NEURON model class ###################################################################'''
'''################## ###################################################################'''

class BahlNeuron(nengo.neurons.NeuronType):
	'''compartmental neuron from Bahl et al 2012'''

	probeable=('spikes','voltage')
	def __init__(self,P,label):
		super(BahlNeuron,self).__init__()
		self.P=P
		self.label=label
		if self.P['load_weights']==True: self.best_hyperparam_files=True
		else: self.best_hyperparam_files=None

	def create(self,bio_idx):
		return self.Bahl(bio_idx)

	class Bahl():
		import numpy as np
		import neuron
		from synapses import ExpSyn, Exp2Syn
		import os
		def __init__(self,bio_idx):
			# neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc') #todo: hardcoded path
			neuron.h.load_file('/home/pduggins/bionengo/NEURON_models/bahl.hoc') #todo: hardcoded path
			# neuron.h.load_file('/home/pduggins/bionengo/NEURON_models/bahl2.hoc') #todo: hardcoded path
			self.bio_idx=bio_idx
			self.bias = None
			self.synapses = {}
			self.netcons = {}
		def add_cell(self): 
			self.cell = neuron.h.Bahl()
			# self.cell = neuron.h.Bahl2()
		def start_recording(self):
			self.bias_current = neuron.h.IClamp(self.cell.soma(0.5))
			self.bias_current.delay = 0
			self.bias_current.dur = 1e9  #todo: limits simulation time
			self.bias_current.amp = self.bias
			self.v_record = neuron.h.Vector()
			self.v_record.record(self.cell.soma(0.5)._ref_v)
			self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
			self.t_record = neuron.h.Vector()
			self.t_record.record(neuron.h._ref_t)
			self.spikes = neuron.h.Vector()
			self.ap_counter.record(neuron.h.ref(self.spikes))
			self.spikes_last=[]

	def rates(self, x, gain, bias): #todo: remove this without errors
		return x

	def gain_bias(self, max_rates, intercepts): #todo: remove this without errors
		return np.ones(len(max_rates)),np.ones(len(max_rates))

	def step_math(self,dt,spiked,neurons,voltage,time):
		desired_t=time*1000
		neuron.run(desired_t) 
		new_spiked=[]
		new_voltage=[]
		for nrn in neurons:
			spike_times=np.array(nrn.spikes)
			spike_times_last=np.array(nrn.spikes_last)
			count=len(spike_times)-len(spike_times_last)
			new_spiked.append(count)
			volt=np.array(nrn.v_record)[-1] #fails if neuron.init() not called at right times
			new_voltage.append(volt)
			nrn.spikes_last=spike_times
		spiked[:]=np.array(new_spiked)/dt
		voltage[:]=np.array(new_voltage)


'''Builder #############################################################################'''
'''####### #############################################################################'''

class SimBahlNeuron(Operator):
	def __init__(self,neurons,output,voltage,states):
		super(SimBahlNeuron,self).__init__()
		self.neurons=neurons
		self.neurons.father_op=self
		self.output=output
		self.voltage=voltage
		self.time=states[0]
		self.reads = [states[0]]
		self.sets=[output,voltage]
		self.updates=[]
		self.incs=[]
		self.P=self.neurons.P
		self.label=self.neurons.label
		self.ens_atributes=None #stores nengo information about the ensemble
		self.inputs={}
		self.neurons.neurons=[self.neurons.create(i) for i in range(self.P[self.label]['n_neurons'])] #range(output.shape[0]) for full
		self.best_hyperparam_files=self.neurons.best_hyperparam_files

	def make_step(self,signals,dt,rng):
		output=signals[self.output]
		voltage=signals[self.voltage]
		time=signals[self.time]
		def step_nrn():
			#one bahlneuron object runs step_math, but arg is all cells - what/where returns?
			self.neurons.step_math(dt,output,self.neurons.neurons,voltage,time)
		return step_nrn

	def init_cells(self):
		for bioneuron in self.neurons.neurons:
			if not hasattr(bioneuron,'cell'):
				bioneuron.add_cell()

	def init_connections(self):
		from synapses import ExpSyn
		for bionrn in range(len(self.neurons.neurons)):
			bioneuron=self.neurons.neurons[bionrn]
			bioneuron.bias=np.load(self.best_hyperparam_files[bionrn]+'/bias.npz')['bias']
			for inpt in self.inputs.iterkeys():
				pre_neurons=self.inputs[inpt]['pre_neurons']
				pre_synapses=self.ens_atributes['n_syn']
				bioneuron.synapses[inpt]=np.empty((pre_neurons,pre_synapses),dtype=object)
				weights=np.load(self.best_hyperparam_files[bionrn]+'/'+inpt+'_weights.npz')['weights']
				locations=np.load(self.best_hyperparam_files[bionrn]+'/'+inpt+'_locations.npz')['locations']
				for pre in range(pre_neurons):
					for syn in range(pre_synapses):	
						section=bioneuron.cell.apical(locations[pre][syn])
						weight=weights[pre][syn]
						synapse=ExpSyn(section,weight,self.ens_atributes['tau'])
						bioneuron.synapses[inpt][pre][syn]=synapse
			bioneuron.start_recording()

class TransmitSpikes(Operator):
	def __init__(self,ens_pre_label,spikes,bahl_op,states):
		self.ens_pre_label=ens_pre_label
		self.spikes=spikes
		self.neurons=bahl_op.neurons.neurons
		self.time=states[0]
		self.reads=[spikes,states[0]]
		self.updates=[]
		self.sets=[]
		self.incs=[]

	def make_step(self,signals,dt,rng):
		spikes=signals[self.spikes]
		time=signals[self.time]
		def step():
			'event-based method'
			for n in range(spikes.shape[0]): #for each input neuron
				if spikes[n] > 0: #if input neuron spiked
					for nrn in self.neurons: #for each bioneuron
						for syn in nrn.synapses[self.ens_pre_label][n]: #for each synapse conn. to input
							syn.spike_in.event(1.0*time*1000) #add a spike at time (ms)
		return step

@Builder.register(BahlNeuron)
def build_bahlneuron(model,neuron_type,ens):
	model.sig[ens]['voltage'] = Signal(np.zeros(ens.ensemble.n_neurons),
						name='%s.voltage' %ens.ensemble.label)
	op=SimBahlNeuron(neurons=neuron_type,output=model.sig[ens]['out'],
						voltage=model.sig[ens]['voltage'],states=[model.time])
	model.add_op(op)

@Builder.register(nengo.Ensemble)
def build_ensemble(model,ens):
	nengo.builder.ensemble.build_ensemble(model,ens)

@Builder.register(nengo.Connection)
def build_connection(model,conn):
	use_nrn = (
		isinstance(conn.post, nengo.Ensemble) and
		isinstance(conn.post.neuron_type, BahlNeuron))
	if use_nrn: #bioneuron connection
		rng = np.random.RandomState(model.seeds[conn])
		model.sig[conn]['in']=model.sig[conn.pre]['out']
		P=copy.copy(conn.post.neuron_type.P)
		bahl_op=conn.post.neuron_type.father_op
		if bahl_op.ens_atributes==None:
			bahl_op.ens_atributes={}
			bahl_op.ens_atributes['label']=conn.post.label
			bahl_op.ens_atributes['neurons']=conn.post.n_neurons
			bahl_op.ens_atributes['dim']=conn.post.dimensions
			bahl_op.ens_atributes['min_rate']=conn.post.max_rates.low
			bahl_op.ens_atributes['max_rate']=conn.post.max_rates.high
			bahl_op.ens_atributes['radius']=conn.post.radius
			bahl_op.ens_atributes['seed']=conn.post.seed
			bahl_op.ens_atributes['n_syn']=P[bahl_op.label]['n_syn']
			bahl_op.ens_atributes['evals']=P[bahl_op.label]['evals']
			bahl_op.ens_atributes['tau']=P[bahl_op.label]['tau']
		#if there's no saved information about this input, save the connection info and later optimize
		if conn.pre.label not in bahl_op.inputs:
			bahl_op.inputs[conn.pre.label]={}
			bahl_op.inputs[conn.pre.label]['pre_label']=conn.pre.label
			bahl_op.inputs[conn.pre.label]['pre_neurons']=conn.pre.n_neurons
			bahl_op.inputs[conn.pre.label]['pre_dim']=conn.pre.dimensions
			bahl_op.inputs[conn.pre.label]['pre_min_rate']=conn.pre.max_rates.low
			bahl_op.inputs[conn.pre.label]['pre_max_rate']=conn.pre.max_rates.high
			# bahl_op.inputs[conn.pre.label]['pre_radius']=conn.pre.radius
			# bahl_op.inputs[conn.pre.label]['pre_seed']=conn.pre.seed
			# bahl_op.inputs[conn.pre.label]['pre_type']=str(conn.pre.neuron_type)
			# bahl_op.inputs[conn.pre.label]['transform']=conn.transform
			# bahl_op.inputs[conn.pre.label]['synapse']=conn.synapse
			#function?
	else: #normal connection
		return nengo.builder.connection.build_connection(model, conn)

'''Helper Functions #############################################################################'''
'''################ #############################################################################'''

def pre_build_func(network,dt): #or take P instead of model
	#this function get called in simulator.py after network is defined and operators are created
	#but before decoders are calculated.
	#Replaces all bioneurons in network with LIF neurons, runs the model with space-covering inputs,
	#collects spikes into the bioneurons (pres) and spikes out of the LIF neurons (ideal)
	bio_dict={}
	opt_net=copy.copy(network)
	for ens in opt_net.ensembles:
		if not isinstance(ens.neuron_type,BahlNeuron): continue #only bioensembles selected
		P=ens.neuron_type.P
		bio_dict[ens.label]={}
		with opt_net:
			bio_dict[ens.label]['probe']=nengo.Probe(ens.neurons,'spikes') #probe output spikes
		bio_dict[ens.label]['inputs']={}
		for conn in opt_net.connections:
			if conn.post == ens:
				bio_dict[ens.label]['inputs'][conn.pre_obj.label]={}
				with opt_net: 
					bio_dict[ens.label]['inputs'][conn.pre_obj.label]['probe']=\
							nengo.Probe(conn.pre_obj.neurons,'spikes')
		with opt_net:
			ens.neuron_type=nengo.LIF() #replace bioensemble with param-identical LIF ensemble
	#rebuild network?
	#define input signals and connect to inputs?
	with nengo.Simulator(opt_net,dt=dt) as opt_sim:
		opt_sim.run(P['optimize']['t_final'])
	for bio in bio_dict.iterkeys():
		try: 
			os.makedirs(bio)
			os.chdir(bio)
		except OSError:
			os.chdir(bio)
		bio_dict[bio]['ideal_spikes']=opt_sim.data[bio_dict[bio]['probe']]
		for inpt in bio_dict[bio]['inputs'].iterkeys():
			bio_dict[bio]['inputs'][inpt]['pre_spikes']=\
					opt_sim.data[bio_dict[bio]['inputs'][inpt]['probe']]
			np.savez('spikes_from_%s_to_%s.npz'%(inpt,bio),spikes=\
					bio_dict[bio]['inputs'][inpt]['pre_spikes'])
		np.savez('spikes_ideal_%s.npz'%bio,spikes=bio_dict[bio]['ideal_spikes'])
		os.chdir('..')
	#replace bioensembles with BahlNeuron neuron_type
	for ens in opt_net.ensembles:
		if ens.label not in bio_dict: continue #only bioensembles selected
		with opt_net:
			ens.neuron_type=BahlNeuron(P,ens.label)
	print 'Pre and ideal spikes generated...'


def post_build_func(model,network):
	#this function get called in simulator.py after models are built but before signals are created
	count_bioensembles=0
	for op in model.operators:
		if isinstance(op, SimBahlNeuron):
			count_bioensembles+=1
			# print op.ens_atributes['label']
			# print 'Initializing cells'
			op.init_cells()
			# print 'Initializing connections'
			op.init_connections()
			dt_neuron=op.P['dt_neuron']*1000
			for conn in network.all_connections:
				if conn.pre_obj.label in op.inputs and conn.post_obj.label == op.ens_atributes['label']:
					# print 'Adding TransmitSpikes operator to model'
					model.add_op(TransmitSpikes(
						conn.pre_obj.label,model.sig[conn.pre]['out'],op,states=[model.time]))
	if count_bioensembles>0:
		neuron.h.dt = dt_neuron #fails if no bioensembles present in model
	else:
		neuron.h.dt = 0.0001*1000
	neuron.init()
	print 'NEURON initialized, beginning simulation...'