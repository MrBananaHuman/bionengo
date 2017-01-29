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
	def __init__(self,P,father_op_inputs=None):
		super(BahlNeuron,self).__init__()
		self.P=P
		self.father_op_inputs=father_op_inputs #for recreating during decoder calculation

	def create(self,bio_idx):
		return self.Bahl(bio_idx)

	class Bahl():
		import numpy as np
		import neuron
		from synapses import ExpSyn, Exp2Syn
		import os
		def __init__(self,bio_idx):
			# neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc')
			neuron.h.load_file('/home/pduggins/bionengo/bahl.hoc') #todo: hardcoded path
			self.bio_idx=bio_idx
			self.bias = None
			self.synapses = {}
			self.netcons = {}
		#creating cells in init() makes optimization run extra neurons;
		#instead call this in build_connection()
		def add_cell(self): 
			self.cell = neuron.h.Bahl()
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
		return x #CustomSolver class calculates A from Y

	def gain_bias(self, max_rates, intercepts): #todo: remove this without errors
		return np.ones(len(max_rates)),np.ones(len(max_rates))

	def step_math(self,dt,spiked,neurons,voltage,time):
		desired_t=time*1000
		#runs NEURON only for 1st bioensemble in model...OK because transmit_spikes happend already?
		neuron.run(desired_t) 
		new_spiked=[]
		new_voltage=[]
		for nrn in neurons:
			spike_times=np.array(nrn.spikes)
			spike_times_last=np.array(nrn.spikes_last)
			count=len(spike_times)-len(spike_times_last)
			# count=np.sum(spike_times>(time-dt)*1000)
			new_spiked.append(count)
			volt=np.array(nrn.v_record)[-1] #fails if neuron.init() not called at right times
			new_voltage.append(volt)
			nrn.spikes_last=spike_times
		spiked[:]=np.array(new_spiked)/dt
		voltage[:]=np.array(new_voltage)

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
		self.P=self.neurons.P
		self.neurons.father_op=self
		self.inputs={} #structured like {ens_in_label: {directory,filenames}}
		self.neurons.neurons=[self.neurons.create(i)for i in range(output.shape[0])]
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

	def init_cells(self):
		for bioneuron in self.neurons.neurons:
			if not hasattr(bioneuron,'cell'):
				bioneuron.add_cell()

	def init_connection(self,ens_pre_label,ens_pre_neurons,n_syn):
		for bioneuron in self.neurons.neurons:
			bioneuron.synapses[ens_pre_label]=np.empty((ens_pre_neurons,n_syn),dtype=object)
			bioneuron.netcons[ens_pre_label]=np.empty((ens_pre_neurons,n_syn),dtype=object)

	def load_weights(self,ens_pre_label):
		from synapses import ExpSyn
		for nrn in range(len(self.neurons.neurons)):
			bioneuron=self.neurons.neurons[nrn]
			with open(self.inputs[ens_pre_label]['filenames'][nrn]) as data_file:
				info=json.load(data_file)
			weights=np.array(info['weights'])
			locations=np.array(info['locations'])
			bioneuron.bias=info['bias']
			# print 'load_weights bias %s' %nrn, bioneuron.bias
			for n in range(weights.shape[0]):
				for s in range(weights.shape[1]):
					section=bioneuron.cell.apical(locations[n][s])
					weight=weights[n][s]
					synapse=ExpSyn(section,weight,self.P['tau'])
					bioneuron.synapses[ens_pre_label][n][s]=synapse
			bioneuron.start_recording()

	def save_optimization(self,conn_pre_label):
		indices=[]
		weights=[]
		locations=[]
		biases=[]
		signal_in=[]
		bio_spikes=[]
		bio_rates=[]
		ideal_spikes=[]
		ideal_rates=[]
		losses=[]
		filenames=self.inputs[conn_pre_label]['filenames']
		for file in filenames: #for each bioneuron
			with open(file,'r') as data_file:
				bioneuron_info=json.load(data_file)
			indices.append(bioneuron_info['bio_idx'])
			weights.append(bioneuron_info['weights'])
			locations.append(bioneuron_info['locations'])
			biases.append(bioneuron_info['bias'])
			signal_in.append(bioneuron_info['signal_in'])
			bio_spikes.append(bioneuron_info['bio_spikes'])
			bio_rates.append(bioneuron_info['bio_rates'])
			ideal_spikes.append(bioneuron_info['ideal_spikes'])
			ideal_rates.append(bioneuron_info['ideal_rates'])
			losses.append(bioneuron_info['loss'])
		self.inputs[conn_pre_label]['indices']=np.array(indices)
		self.inputs[conn_pre_label]['weights']=np.array(weights)
		self.inputs[conn_pre_label]['locations']=np.array(locations)
		self.inputs[conn_pre_label]['biases']=np.array(biases)
		self.inputs[conn_pre_label]['signal_in']=np.array(signal_in)[0]
		self.inputs[conn_pre_label]['bio_spikes']=np.array(bio_spikes)
		self.inputs[conn_pre_label]['bio_rates']=np.array(bio_rates)
		self.inputs[conn_pre_label]['ideal_spikes']=np.array(ideal_spikes)
		self.inputs[conn_pre_label]['ideal_rates']=np.array(ideal_rates)
		self.inputs[conn_pre_label]['losses']=np.array(losses)
		# print 'save optimization biases', self.inputs[conn_pre_label]['biases']
		np.savez(self.inputs[conn_pre_label]['directory']+'/'+'biodata.npz',
				indices=self.inputs[conn_pre_label]['indices'],
				weights=self.inputs[conn_pre_label]['weights'],
				locations=self.inputs[conn_pre_label]['locations'],
				biases=self.inputs[conn_pre_label]['biases'],
				signal_in=self.inputs[conn_pre_label]['signal_in'],
				bio_spikes=self.inputs[conn_pre_label]['bio_spikes'],
				bio_rates=self.inputs[conn_pre_label]['bio_rates'],
				ideal_spikes=self.inputs[conn_pre_label]['ideal_spikes'],
				ideal_rates=self.inputs[conn_pre_label]['ideal_rates'],
				losses=self.inputs[conn_pre_label]['losses'])

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
		neuron.h.dt = bahl_op.P['dt_neuron']*1000
		neuron.init()

	def make_step(self,signals,dt,rng):
		spikes=signals[self.spikes]
		time=signals[self.time]
		def step():
			'event-based method'
			for n in range(spikes.shape[0]): #for each input neuron
				if spikes[n] > 0: #if this neuron spiked at this time, then
					for nrn in self.neurons: #for each bioneuron
						for syn in nrn.synapses[self.ens_pre_label][n]: #for each synapse conn. to input
							syn.spike_in.event(1.0*time*1000) #add a spike at time t (ms) #ROUND??
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
	import copy
	import os
	use_nrn = (
		isinstance(conn.post, nengo.Ensemble) and
		isinstance(conn.post.neuron_type, BahlNeuron))
	if use_nrn: #bioneuron connection
		rng = np.random.RandomState(model.seeds[conn])
		model.sig[conn]['in']=model.sig[conn.pre]['out']
		P=copy.copy(conn.post.neuron_type.P)
		bahl_op=conn.post.neuron_type.father_op

		#if you're recreating bioneurons in decoder calculation, grab the bahl_op.inputs info
		if conn.post.neuron_type.father_op_inputs != None:
			# print 'loading optimized weights for %s' %conn.pre.label
			directory=conn.post.neuron_type.father_op_inputs[conn.pre.label]['directory']
			filenames_dir=directory+'filenames.txt'
			with open(filenames_dir,'r') as df:
				bahl_op.inputs[conn.pre.label]={'directory':directory,'filenames':json.load(df)}
			# print 'directory',directory


		#if there's no saved information about this input connection, do an optimization
		elif conn.pre.label not in bahl_op.inputs:
			from optimize_bioneuron import optimize_bioneuron
			# from optimize_recurrent import optimize_recurrent
			P['ens_pre_neurons']=conn.pre.n_neurons
			P['ens_pre_dim']=conn.pre.dimensions
			P['ens_pre_min_rate']=conn.pre.max_rates.low
			P['ens_pre_max_rate']=conn.pre.max_rates.high
			P['ens_pre_radius']=conn.pre.radius
			P['ens_pre_seed']=conn.pre.seed
			P['ens_pre_type']=str(conn.pre.neuron_type)
			P['ens_pre_label']=conn.pre.label
			P['ens_label']=conn.post.label
			P['ens_ideal_neurons']=conn.post.n_neurons
			P['ens_ideal_dim']=conn.post.dimensions
			P['ens_ideal_seed']=conn.post.seed
			P['ens_ideal_min_rate']=conn.post.max_rates.low
			P['ens_ideal_max_rate']=conn.post.max_rates.high
			P['ens_ideal_radius']=conn.post.radius
			P['conn_transform']=conn.transform
			#biases either equal None (first optimization) or are set with load_weights after first
			P['biases']=[bahl_op.neurons.neurons[n].bias for n in range(len(bahl_op.neurons.neurons))]
			P['input_path']=P['directory']+P['ens_label']+'/'+P['ens_pre_label']+'/'
			if os.path.exists(P['input_path']):
				directory=P['input_path']
			else:
				# directory=optimize_recurrent(P)
				directory=optimize_bioneuron(P)
			filenames_dir=directory+'filenames.txt'
			with open(filenames_dir,'r') as df:
				bahl_op.inputs[conn.pre.label]={'directory':directory,'filenames':json.load(df)}

		#load weights and locations into the neurons contained in bahl_op.neurons,
		#save contents of filenames to operator.inputs, and setup spike transmission
		bahl_op.init_cells()
		bahl_op.init_connection(conn.pre.label,conn.pre.n_neurons,P['n_syn'])
		bahl_op.load_weights(conn.pre.label) 
		bahl_op.save_optimization(conn.pre.label)
		# for key in bahl_op.inputs.iterkeys():
		# 	print 'key', key
		# 	bahlop_weights=bahl_op.inputs[key]['weights']
		# 	syn_weights=[]
		# 	for nrn in bahl_op.neurons.neurons:
		# 		for inpt in nrn.synapses[key]:
		# 			for syn in inpt:
		# 				syn_weights.append(syn.weight)
		# 	print 'bahl_op_weights', bahlop_weights.sum()
		# 	print 'syn_weights',np.array(syn_weights).sum()
		model.add_op(TransmitSpikes(conn.pre.label,model.sig[conn]['in'],bahl_op,states=[model.time]))

	else: #normal connection
		return nengo.builder.connection.build_connection(model, conn)