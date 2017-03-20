import numpy as np
import nengo
import neuron
from nengo.builder import Builder, Operator, Signal
import copy
import ipdb
import json
import os
import gc

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
		self.atrb={}
		self.inputs={}

	def create(self,bio_idx):
		return self.Bahl(bio_idx)

	class Bahl():
		import numpy as np
		import neuron
		from synapses import ExpSyn
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
		def add_bias(self):
			self.bias_current = neuron.h.IClamp(self.cell.soma(0.5))
			self.bias_current.delay = 0
			self.bias_current.dur = 1e9  #todo: limits simulation time
			self.bias_current.amp = self.bias		
		def start_recording(self):
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
		# print 'step math', self.father_op.label
		t_neuron=time*1000
		neuron.run(t_neuron) 
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
		self.inputs={}
		self.neurons.neurons=[self.neurons.create(i) for i in range(self.P[self.label]['n_neurons'])] #range(output.shape[0]) for full
		self.ens_atributes=self.neurons.atrb
		self.inputs=self.neurons.inputs
		self.best_hyperparam_files=self.neurons.best_hyperparam_files
		self.targets=self.neurons.targets
		self.activities=self.neurons.activities

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
			if self.P['optimize_bias']==True:
				bioneuron.bias=np.load(self.best_hyperparam_files[bionrn]+'/bias.npz')['bias']
				bioneuron.add_bias()
			for inpt in self.inputs.iterkeys():
				pre_neurons=self.inputs[inpt]['pre_neurons']
				pre_synapses=self.ens_atributes['n_syn']
				bioneuron.synapses[inpt]=np.empty((pre_neurons,pre_synapses),dtype=object)
				weights=np.load(self.best_hyperparam_files[bionrn]+'/'+inpt+'_weights.npz')['weights']
				locations=np.load(self.best_hyperparam_files[bionrn]+'/'+inpt+'_locations.npz')['locations']
				# ipdb.set_trace()
				for pre in range(pre_neurons):
					for syn in range(pre_synapses):	
						section=bioneuron.cell.apical(locations[pre][syn])
						weight=weights[pre][syn]
						synapse=ExpSyn(section,weight,self.ens_atributes['tau'])
						bioneuron.synapses[inpt][pre][syn]=synapse
				# print '\nBioneuron:', bionrn
				# print 'Loss:',np.load(self.best_hyperparam_files[bionrn]+'/loss.npz')['loss']
				# print inpt
				# print 'Weights: mean=',np.average(weights),'std=',np.std(weights)
				# print 'Locations',np.average(locations)
			bioneuron.start_recording()
			# print 'Bias:', bioneuron.bias

class TransmitSpikes(Operator):
	def __init__(self,ens_pre_label,spikes,bahl_op,states):
		self.ens_pre_label=ens_pre_label
		self.spikes=spikes
		self.bahl_op=bahl_op
		self.neurons=bahl_op.neurons.neurons
		self.time=states[0]
		self.reads=[spikes,states[0]]
		self.updates=[]
		self.sets=[]
		self.incs=[]
		if self.bahl_op.ens_atributes['label'] == 'ens_bio':
			self.trained_voltages=[]
			self.trained_spikes=[]
			for file in self.bahl_op.best_hyperparam_files:
				spikes_rates_bio_ideal=np.load(file+'/spikes_rates_bio_ideal.npz')
				self.trained_voltages.append(spikes_rates_bio_ideal['voltages'])
				self.trained_spikes.append(spikes_rates_bio_ideal['spikes_bio'])
			self.trained_voltages=np.array(self.trained_voltages).T
			self.trained_spikes=np.array(self.trained_spikes).T
			self.ideal_spikes=np.load(self.bahl_op.P['directory']+self.bahl_op.ens_atributes['label']+'/spikes_from_%s_to_%s.npz'%('ens_bio','ens_bio'))['spikes']
			self.spikes_pre_to_bio=np.load(self.bahl_op.P['directory']+self.bahl_op.ens_atributes['label']+'/spikes_from_%s_to_%s.npz'%('pre','ens_bio'))['spikes']
		# ipdb.set_trace()
	def make_step(self,signals,dt,rng):
		spikes=signals[self.spikes]
		time=signals[self.time]
		def step():
			'event-based method'
			tback=time-dt
			t_neuron=tback*1000
			# print 'make_step', 'ens', self.bahl_op.ens_atributes['label'], 'pre', self.ens_pre_label

			if self.bahl_op.ens_atributes['label'] == 'ens_bio' and self.ens_pre_label == 'ens_bio':
				try:
					# print time, round(time/dt)
					# print 'ideal bio-bio spikes', self.ideal_spikes[round(time/dt)]
					# if self.trained_spikes[round(time/dt)].sum() != spikes.sum():
					# print 'trained bio-bio spikes', self.trained_spikes[round(time/dt)]
					# print 'actual bio-bio spikes', spikes
					# print 'trained bio voltages', self.trained_voltages[round(time/self.bahl_op.P['dt_neuron'])]
					# print 'actual bio voltages', [np.array(nrn.v_record)[-1] for nrn in self.neurons]
					for n in range(spikes.shape[0]): #for each input neuron
						my_spikes=spikes[n] #actual spikes
						if my_spikes > 0: #if input neuron spiked
							for nrn in self.neurons: #for each bioneuron
								for syn in nrn.synapses[self.ens_pre_label][n]: #for each synapse conn. to input
									syn.spike_in.event(t_neuron) #add a spike at time (ms)
				except: print 'out of range'

			elif self.bahl_op.ens_atributes['label'] == 'ens_bio' and self.ens_pre_label == 'pre':
				try:
					# print 'ideal pre-bio spikes', self.spikes_pre_to_bio[round(time/dt)]
					# print 'actual pre-bio spikes', spikes
					for n in range(spikes.shape[0]): #for each input neuron
						# if spikes[n] > 0: #if input neuron spiked
						if self.spikes_pre_to_bio[round(time/dt),n] > 0: #if input neuron spiked
							for nrn in self.neurons: #for each bioneuron
								for syn in nrn.synapses[self.ens_pre_label][n]: #for each synapse conn. to input
									syn.spike_in.event(t_neuron) #add a spike at time (ms)
				except: print 'out of range'

			else:
				for n in range(spikes.shape[0]): #for each input neuron
					if spikes[n] > 0: #if input neuron spiked
						for nrn in self.neurons: #for each bioneuron
							for syn in nrn.synapses[self.ens_pre_label][n]: #for each synapse conn. to input
								syn.spike_in.event(t_neuron) #add a spike at time (ms)
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
	from nengo.builder.connection import BuiltConnection
	use_nrn = (
		isinstance(conn.post, nengo.Ensemble) and
		isinstance(conn.post.neuron_type, BahlNeuron))
	if use_nrn: #bioneuron connection
		rng = np.random.RandomState(model.seeds[conn])
		model.sig[conn]['in']=model.sig[conn.pre]['out']
		# bahl_op=conn.post.neuron_type.father_op
		# bahl_op.ens_atributes=conn.post.neuron_type.atrb
		# bahl_op.inputs=conn.post.neuron_type.inputs
		from nengo.dists import get_samples	
		from nengo.builder.connection import build_decoders
		transform = get_samples(conn.transform, conn.size_out, d=conn.size_mid, rng=rng)
		eval_points, weights, solver_info = build_decoders(model, conn, rng, transform)
		model.params[conn] = BuiltConnection(eval_points=eval_points,
                                         solver_info=solver_info,
                                         transform=transform,
                                         weights=weights)

	else: #normal connection
		return nengo.builder.connection.build_connection(model, conn)

'''Helper Functions #############################################################################'''
'''################ #############################################################################'''

def pre_build_func(network,dt):
	'''called in simulator.py after network is defined and operators are created but before decoders are calculated.'''
	#first save all information about incoming connections to attributes on ens.neuron_type
	for ens in network.ensembles:
		if not isinstance(ens.neuron_type,BahlNeuron): continue #only bioensembles selected
		# ens.neuron_type.atrb={}
		# ens.neuron_type.inputs={}
		for conn in network.connections:
			if conn.post == ens:
				if len(ens.neuron_type.atrb) == 0:
					ens.neuron_type.atrb['label']=conn.post.label
					ens.neuron_type.atrb['neurons']=conn.post.n_neurons
					ens.neuron_type.atrb['dim']=conn.post.dimensions
					ens.neuron_type.atrb['min_rate']=conn.post.max_rates.low
					ens.neuron_type.atrb['max_rate']=conn.post.max_rates.high
					ens.neuron_type.atrb['radius']=conn.post.radius
					ens.neuron_type.atrb['seed']=conn.post.seed
					ens.neuron_type.atrb['n_syn']=ens.neuron_type.P[conn.post.label]['n_syn']
					ens.neuron_type.atrb['evals']=ens.neuron_type.P[conn.post.label]['evals']
					ens.neuron_type.atrb['tau']=ens.neuron_type.P[conn.post.label]['tau']
				if conn.pre.label not in ens.neuron_type.inputs:
					ens.neuron_type.inputs[conn.pre.label]={}
					ens.neuron_type.inputs[conn.pre.label]['pre_label']=conn.pre.label
					ens.neuron_type.inputs[conn.pre.label]['pre_neurons']=conn.pre.n_neurons
					ens.neuron_type.inputs[conn.pre.label]['pre_dim']=conn.pre.dimensions
					ens.neuron_type.inputs[conn.pre.label]['pre_min_rate']=conn.pre.max_rates.low
					ens.neuron_type.inputs[conn.pre.label]['pre_max_rate']=conn.pre.max_rates.high

	print 'Generating pre spikes, ideal spikes, target values...'
	bio_dict={}
	lif_net=network.copy()
	del lif_net.probes[:]
	#Replaces all bioneurons in network with LIF neurons, runs the model with space-covering inputs,
	#collects spikes into the bioneurons (pres) and spikes out of the LIF neurons (ideal)
	for ens in lif_net.ensembles:
		if not isinstance(ens.neuron_type,BahlNeuron): continue #only bioensembles selected
		P=ens.neuron_type.P
		bio_dict[ens.label]={}
		with lif_net:
			ens.neuron_type=nengo.LIF() #replace bioensemble with param-identical LIF ensemble
			bio_dict[ens.label]['ensemble']=ens
			bio_dict[ens.label]['probe_ideal_spikes']=nengo.Probe(ens.neurons,'spikes') #probe output spikes
		bio_dict[ens.label]['inputs']={}
		for conn in lif_net.connections:
			if conn.post == ens:
				conn.solver=nengo.solvers.LstsqL2()
				bio_dict[ens.label]['inputs'][conn.pre_obj.label]={}
				with lif_net: 
					bio_dict[ens.label]['inputs'][conn.pre_obj.label]['probe_pre_spikes']=\
							nengo.Probe(conn.pre_obj.neurons,'spikes')

	target_net=network.copy()
	#replace all neurons in target_net with direct neurons to simulate target values (analytic solution)
	del target_net.probes[:]
	with target_net:
		for ens in target_net.ensembles:
			if P['target_method'] == 'LIF':
				ens.neuron_type=nengo.LIF()	
			elif P['target_method'] == 'LIFrate':
				ens.neuron_type=nengo.LIFRate()	
			elif P['target_method'] == 'direct':
				ens.neuron_type=nengo.Direct()	
			if ens.label in bio_dict:
				bio_dict[ens.label]['probe_ideal_output']=nengo.Probe(ens,synapse=P['kernel']['tau']) #probe ideal output

	# print 'before sim.run in pre_build_func'
	with nengo.Simulator(lif_net,dt=dt) as lif_sim:
		lif_sim.run(P['train']['t_final'])
	with nengo.Simulator(target_net,dt=dt) as target_sim:
		target_sim.run(P['train']['t_final'])

	#save probe data in .npz files, which are loaded during training and decoder calculation
	# print 'decoders in pre_build_func'
	for bio in bio_dict.iterkeys():
		# print bio
		try: 
			os.makedirs(bio)
			os.chdir(bio)
		except OSError:
			os.chdir(bio)
		bio_dict[bio]['encoders']=lif_sim.data[bio_dict[bio]['ensemble']].encoders
		bio_dict[bio]['ideal_spikes']=lif_sim.data[bio_dict[bio]['probe_ideal_spikes']]
		bio_dict[bio]['ideal_output']=target_sim.data[bio_dict[bio]['probe_ideal_output']]
		for conn in lif_net.connections:
			if conn.pre.label in bio_dict[bio]['inputs'] and conn.post_obj.label == bio:
				bio_dict[bio]['inputs'][conn.pre.label]['decoders']=lif_sim.data[conn].weights.T
		for inpt in bio_dict[bio]['inputs'].iterkeys():
			bio_dict[bio]['inputs'][inpt]['pre_spikes']=\
					lif_sim.data[bio_dict[bio]['inputs'][inpt]['probe_pre_spikes']]
			np.savez('spikes_from_%s_to_%s.npz'%(inpt,bio),spikes=\
					bio_dict[bio]['inputs'][inpt]['pre_spikes'])
			np.savez('decoders_from_%s_to_%s.npz'%(inpt,bio),decoders=\
					bio_dict[bio]['inputs'][inpt]['decoders'])
		np.savez('encoders_for_%s.npz'%bio,encoders=bio_dict[bio]['encoders'])		
		np.savez('spikes_ideal_%s.npz'%bio,spikes=bio_dict[bio]['ideal_spikes'])
		np.savez('output_ideal_%s.npz'%bio,values=bio_dict[bio]['ideal_output'])
		os.chdir('..')
	del lif_net, lif_sim, target_net, target_sim


	#run the optimization
	optimize_bioensembles(network)

def optimize_bioensembles(network):
	from bioneuron_train import train_hyperparams
	from bioneuron_helper import load_hyperparams
	for ens in network.ensembles:
		if not isinstance(ens.neuron_type,BahlNeuron): continue #only bioensembles selected
		print 'Optimizing synaptic weights into %s' %ens
		P=copy.copy(ens.neuron_type.P)
		# ipdb.set_trace()
		P['inpts']=ens.neuron_type.inputs
		P['atrb']=ens.neuron_type.atrb
		if ens.neuron_type.best_hyperparam_files != None and P['continue_optimization']==False:
			best_hyperparam_files, targets, activities = load_hyperparams(P) #don't train if filenames already exists
		else: 
			best_hyperparam_files, targets, activities = train_hyperparams(P)
		ens.neuron_type.best_hyperparam_files=best_hyperparam_files
		ens.neuron_type.targets=targets
		ens.neuron_type.activities=activities

def post_build_func(model,network):
	#this function get called in simulator.py after models are built but before signals are created
	for op in model.operators:
		if isinstance(op, SimBahlNeuron):
			op.init_cells()
			op.init_connections()
			neuron.h.dt=op.P['dt_neuron']*1000
			# t_transient=0.1*op.P['test']['t_final']*1000
			# neuron.init()
			# print 'Running transients for', op.ens_atributes['label']
			# neuron.run(t_transient) #run out transients (no inputs to model, lets voltages stabilize)
			# for bioneuron in op.neurons.neurons:
			# 	bioneuron.start_recording() #reset recording attributes in neuron
			for conn in network.all_connections:
				if conn.pre_obj.label in op.inputs and conn.post_obj.label == op.ens_atributes['label']:
					# print 'transmit spikes on connection', conn, 'with operator', op.ens_atributes['label']
					model.add_op(TransmitSpikes(
						conn.pre_obj.label,model.sig[conn.pre]['out'],op,states=[model.time]))
	neuron.init()
	print 'NEURON initialized, transients run, beginning simulation...'
